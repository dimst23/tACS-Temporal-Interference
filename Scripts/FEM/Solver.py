from __future__ import absolute_import
import os
import gc
import sys
import yaml
import meshio
import numpy as np

#### SfePy libraries
from sfepy.base.base import output, IndexedStruct, Struct
from sfepy.discrete import (FieldVariable, Material, Integral, Integrals, Equation, Equations, Problem, Function)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import PETScKrylovSolver
from sfepy.solvers.nls import Newton
#### SfePy libraries

import Meshing.modulation_envelope as mod_env

class Solver:
    def __init__(self, settings_file, settings_header, mesh):
        self.settings = settings_file
        self.settings_header = settings_header
        self.mesh = mesh
        self.linear_solver = None
        self.non_linear_solver = None
        self.conductivities = {}
        self.overall_volume = None

        # Read from settings
        self.material_conductivity = None
        self.domain = None
        self.selected_model = None
        self.essential_boundaries = {}
        self.field_variables = {}
        self.fields = {}
        print()

    def load_mesh(self, file_name=None):
        if file_name is None:
            mesh = meshio.read(self.settings[self.settings_header][self.selected_model])
        else:
            mesh = meshio.read(file_name)
        loaded_mesh = Mesh.from_data('mesh_name', mesh.points, None, [mesh.cells[0][1]], mesh.cell_data['cell_scalars'], ['3_4'])
        self.domain = FEDomain('domain', loaded_mesh)

    def define_essential_boundary(self, region_name: str, group_id: int, field_variable: str, field_value: float):
        # TODO: Add a check to see if the provided potential variable is a defined potential
        # TODO: Check if the temporary domain can be deleted after using it in EBC
        # TODO: Check if the provided `groupid` is valid
        # TODO: Do not run if there are no field variables
        temporary_domain = self.domain.create_region(region_name, 'vertices of group ' + str(group_id), 'facet', add_to_regions=False)
        self.essential_boundaries[region_name] = {
            'boundary_region': EssentialBC('base_vcc', temporary_domain, {field_variable + '.all' : field_value}),
        }

    def define_field_variable(self, var_name: str, field_name: str):
        # TODO: Check if the provided field exists
        # TODO: 
        self.field_variables[var_name] = {
            'unknown': FieldVariable(var_name, 'unknown', field=field_name),
            'test': FieldVariable(var_name + 'test', 'test', field=field_name, primary_var_name=var_name),
        }

    def create_field(self, field_name: str):
        # TODO: Check if there are any field created
        # TODO: Add the ability to select the region of the field
        self.fields[field_name] = Field.from_args(field_name, dtype=np.float64, shape=(1, ), region=self.overall_volume, approx_order=1)

    def run(self):
        problem = Problem('temporal_interference', equations=equations)
        problem.set_bcs(ebcs=Conditions([bc_base_vcc, bc_base_gnd, bc_df_vcc, bc_df_gnd]))
        problem.set_solver(solver)
        # Solve the problem
        state = problem.solve(post_process_hook=post_process)

    def solver_setup(self, max_iterations=250, relative_tol=1e-7, absolute_tol=1e-3):
        ls_status = IndexedStruct()
        self.linear_solver = PETScKrylovSolver({
            'ksp_max_it': max_iterations,
            'ksp_rtol': relative_tol,
            'ksp_atol': absolute_tol,
            'ksp_type': 'cg',
            'pc_type': 'hypre',
            'pc_hypre_type': 'boomeramg',
            'pc_hypre_boomeramg_coarsen_type': 'HMIS',
        }, status=ls_status)

        nls_status = IndexedStruct()
        self.non_linear_solver = Newton({
            'i_max': 1,
            'eps_a': absolute_tol,
        }, lin_solver=self.linear_solver, status=nls_status)

    def __generate_equations(self):
        integral = Integral('i1', order=2)

        laplace_base = Term.new('dw_laplace(conductivity.val, s_base, potential_base)', integral, region=self.overall_volume, conductivity=self.material_conductivity, potential_base=fld_potential_base, s_base=fld_s_base)
        laplace_df = Term.new('dw_laplace(conductivity.val, s_df, potential_df)', integral, region=self.overall_volume, conductivity=self.material_conductivity, potential_df=fld_potential_df, s_df=fld_s_df)

        eq_base = Equation('balance', laplace_base)
        eq_df = Equation('balance', laplace_df)

        return Equations([eq_base, eq_df])

    def __material_definition(self):
        if not self.conductivities:
            print('Conductivities are not assigned. Calling region assignment automatically.') # Warning log
            self.__assign_regions()
        self.material_conductivity = Material('conductivity', function=Function('get_conductivity', lambda ts, coors, mode=None, equations=None, term=None, problem=None, **kwargs: self.__get_conductivity(ts, coors, mode, equations, term, problem, conductivities=self.conductivities)))

    def __assign_regions(self):
        self.overall_volume = self.domain.create_region('Omega', 'all')

        for region in self.settings[self.settings_header][self.selected_model]['regions'].items():
            self.domain.create_region(region[0], 'cells of group ' + str(region[1]['id']))
            self.conductivities[region[0]] = region[1]['conductivity']

        for electrode in self.settings[self.settings_header]['electrodes']['10-20'].items():
            self.domain.create_region(electrode[0], 'cells of group ' + str(electrode[1]['id']))
            self.conductivities[electrode[0]] = self.settings[self.settings_header]['electrodes']['conductivity']

    def __get_conductivity(self, ts, coors, mode=None, equations=None, term=None, problem=None, conductivities=None):
        # Execute only once at the initialization
        if mode == 'qp':
            values = np.empty(int(coors.shape[0]/4)) # Each element corresponds to one coordinate of the respective tetrahedral edge

            # Save the conductivity values
            for domain in problem.domain.regions:
                if domain.name in conductivities.keys():
                    values[domain.entities[3]] = conductivities[domain.name]

            values = np.repeat(values, 4) # Account for the tetrahedral edges
            values.shape = (coors.shape[0], 1, 1)

            return {'val' : values}

    def __post_process(self, out, problem):
        e_field_base = problem.evaluate('-ev_grad.2.Omega(potential_base)', mode='qp')
        e_field_df = problem.evaluate('-ev_grad.2.Omega(potential_df)', mode='qp')

        # Calculate the maximum modulation envelope
        modulation_cells = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0])
        modulation_cells = np.repeat(modulation_cells, 4, axis=0).reshape((e_field_base.shape[0], 4, 1, 1))
        modulation_points = modulation_cells.flatten()[np.unique(problem.domain.mesh.get_conn('3_4').flatten(), return_index=True)[1]]

        # Calculate the directional modulation envelope
        modulation_x = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0], dir_vector=[1, 0, 0])
        modulation_y = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0], dir_vector=[0, 1, 0])
        modulation_z = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0], dir_vector=[0, 0, 1])

        modulation_x = np.repeat(modulation_x, 4, axis=0).reshape((e_field_base.shape[0], 4, 1, 1))
        modulation_y = np.repeat(modulation_y, 4, axis=0).reshape((e_field_base.shape[0], 4, 1, 1))
        modulation_z = np.repeat(modulation_z, 4, axis=0).reshape((e_field_base.shape[0], 4, 1, 1))

        # Save the output
        out['e_field_base'] = Struct(name='e_field_base', mode='cell', data=e_field_base, dofs=None)
        out['e_field_df'] = Struct(name='e_field_df', mode='cell', data=e_field_df, dofs=None)
        out['max_modulation'] = Struct(name='max_modulation', mode='cell', data=modulation_cells, dofs=None)
        out['max_modulation_pts'] = Struct(name='max_modulation_pts', mode='vertex', data=modulation_points, dofs=None)
        out['modulation_x'] = Struct(name='modulation_x', mode='cell', data=modulation_x, dofs=None)
        out['modulation_y'] = Struct(name='modulation_y', mode='cell', data=modulation_y, dofs=None)
        out['modulation_z'] = Struct(name='modulation_z', mode='cell', data=modulation_z, dofs=None)

        return out
