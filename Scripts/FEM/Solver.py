from __future__ import absolute_import
import os
import gc
import sys
import meshio
import numpy as np

#### SfePy libraries
from sfepy.base.base import Struct
from sfepy.discrete import (FieldVariable, Material, Integral, Equation, Equations, Problem, Function)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import PETScKrylovSolver
from sfepy.solvers.nls import Newton
#### SfePy libraries

import Meshing.modulation_envelope as mod_env

class Solver:
    def __init__(self, settings_file: dict, settings_header: str, electrode_system: str):
        self.__settings = settings_file
        self.__settings_header = settings_header
        if os.name == 'nt':
            self.__extra_path = '_windows'
        else:
            self.__extra_path = ''
        sys.path.append(os.path.realpath(settings_file[settings_header]['lib_path' + self.__extra_path]))

        self.__linear_solver = None
        self.__non_linear_solver = None
        self.__overall_volume = None
        self.conductivities = {}
        self.electrode_system = electrode_system

        # Read from settings
        self.__material_conductivity = None
        self.__selected_model = None
        self.domain = None
        self.problem = None
        self.essential_boundaries = []
        self.field_variables = {}
        self.fields = {}

    def load_mesh(self, model=None, connectivity='3_4'):
        if model is None:
            raise AttributeError('No model was selected.')
        mesh = meshio.read(self.__settings[self.__settings_header][model]['mesh_file' + self.__extra_path])
        self.__selected_model = model

        vertices = mesh.points
        vertex_groups = np.empty(vertices.shape[0])
        cells = mesh.cells[0][1]
        cell_groups = mesh.cell_data['cell_scalars'][0]

        for group in np.unique(cell_groups):
            roi_cells = np.unique(cells[np.where(cell_groups == group)[0]])
            vertex_groups[roi_cells] = group

        loaded_mesh = Mesh.from_data('mesh_name', vertices, vertex_groups, [cells], [cell_groups], [connectivity])
        self.domain = FEDomain('domain', loaded_mesh)

    def define_field_variable(self, var_name: str, field_name: str):
        # TODO: Check if the provided field exists
        if not self.__overall_volume:
            self.__assign_regions()
        if field_name not in self.fields.keys():
            self.fields[field_name] = Field.from_args(field_name, dtype=np.float64, shape=(1, ), region=self.__overall_volume, approx_order=1)

        self.field_variables[var_name] = {
            'unknown': FieldVariable(var_name, 'unknown', field=self.fields[field_name]),
            'test': FieldVariable(var_name + '_test', 'test', field=self.fields[field_name], primary_var_name=var_name),
        }

    def define_essential_boundary(self, region_name: str, group_id: int, field_variable: str, field_value: float):
        # TODO: Add a check to see if the provided potential variable is a defined potential
        # TODO: Do not run if there are no field variables
        if field_variable not in self.field_variables.keys():
            raise AttributeError('The field variable {}')
        temporary_domain = self.domain.create_region(region_name, 'vertices of group ' + str(group_id), 'facet', add_to_regions=False)
        self.essential_boundaries.append(EssentialBC(region_name, temporary_domain, {field_variable + '.all' : field_value}))

    def solver_setup(self, max_iterations=250, relative_tol=1e-7, absolute_tol=1e-3, verbose=False):
        self.__linear_solver = PETScKrylovSolver({
            'ksp_max_it': max_iterations,
            'ksp_rtol': relative_tol,
            'ksp_atol': absolute_tol,
            'ksp_type': 'cg',
            'pc_type': 'hypre',
            'pc_hypre_type': 'boomeramg',
            'pc_hypre_boomeramg_coarsen_type': 'HMIS',
            'verbose': 2 if verbose else 0,
        })

        self.__non_linear_solver = Newton({
            'i_max': 1,
            'eps_a': absolute_tol,
        }, lin_solver=self.__linear_solver)

    def run_solver(self, save_results: bool, post_process_calculation=True, output_dir=None, output_file_name=None):
        if not self.__non_linear_solver:
            raise AttributeError('The solver is not setup. Please set it up before calling run.')
        self.__material_definition()

        self.problem = Problem('temporal_interference', equations=self.__generate_equations())
        self.problem.set_bcs(ebcs=Conditions(self.essential_boundaries))
        self.problem.set_solver(self.__non_linear_solver)
        self.problem.setup_output(output_filename_trunk=output_file_name, output_dir=output_dir)

        if post_process_calculation:
            return self.problem.solve(post_process_hook=self.__post_process, save_results=save_results)
        return self.problem.solve(save_results=save_results)

    def set_custom_post_process(self, function):
        self.__post_process = function

    def clear_all(self):
        del self.domain
        del self.__overall_volume
        del self.essential_boundaries
        del self.field_variables
        del self.fields
        del self.problem
        gc.collect()

    def __generate_equations(self):
        # TODO: Add a check for the existence of the fields
        integral = Integral('i1', order=2)

        equations_list = []
        for field_variable in self.field_variables.items():
            term_arguments = {
                'conductivity': self.__material_conductivity,
                field_variable[0] + '_test': field_variable[1]['test'],
                field_variable[0]: field_variable[1]['unknown']
            }
            equation_term = Term.new('dw_laplace(conductivity.val, ' + field_variable[0] + '_test, ' + field_variable[0] + ')', integral, self.__overall_volume, **term_arguments)
            equations_list.append(Equation('balance', equation_term))

        return Equations(equations_list)

    def __material_definition(self):
        if not self.conductivities:
            self.__assign_regions()
        self.__material_conductivity = Material('conductivity', function=Function('get_conductivity', lambda ts, coors, mode=None, equations=None, term=None, problem=None, **kwargs: self.__get_conductivity(ts, coors, mode, equations, term, problem, conductivities=self.conductivities)))

    def __assign_regions(self):
        self.__overall_volume = self.domain.create_region('Omega', 'all')

        for region in self.__settings[self.__settings_header][self.__selected_model]['regions'].items():
            self.domain.create_region(region[0], 'cells of group ' + str(region[1]['id']))
            self.conductivities[region[0]] = region[1]['conductivity']

        for electrode in self.__settings[self.__settings_header]['electrodes'][self.electrode_system].items():
            self.domain.create_region(electrode[0], 'cells of group ' + str(electrode[1]['id']))
            self.conductivities[electrode[0]] = self.__settings[self.__settings_header]['electrodes']['conductivity']

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

    def __post_process(self, out, problem, state, extend=False):
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
