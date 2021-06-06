from __future__ import absolute_import
import os
import gc
import sys
import numpy as np
from numpy.core.fromnumeric import mean
import pyvista as pv

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
    def __init__(self, settings_file: dict, settings_header: str, electrode_system: str, units: str = 'mm'):
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
        self.__overall_surface = None
        self.__fields_to_calculate = []
        self.__electrode_currents = {}
        self.conductivities = {}
        self.electrode_system = electrode_system
        self.units = units

        # Read from settings
        self.__material_conductivity = None
        self.__selected_model = None
        self.domain = None
        self.problem = None
        self.essential_boundary_ids = {}
        self.essential_boundaries = []
        self.field_variables = {}
        self.fields = {}

    def load_mesh(self, model=None, connectivity='3_4', id_array_name='cell_scalars'):
        if model is None:
            raise AttributeError('No model was selected.')
        mesh = pv.UnstructuredGrid(self.__settings[self.__settings_header][model]['mesh_file' + self.__extra_path])
        self.__selected_model = model

        vertices = mesh.points
        vertex_groups = np.empty(vertices.shape[0])
        cells = mesh.cell_connectivity.reshape((-1, 4)) # TODO: Generalize for higher order connectivity
        cell_groups = mesh.cell_arrays[id_array_name]

        for group in np.unique(cell_groups):
            roi_cells = np.unique(cells[np.where(cell_groups == group)[0]])
            vertex_groups[roi_cells] = group

        loaded_mesh = Mesh.from_data('model_mesh', vertices, vertex_groups, [cells], [cell_groups], [connectivity])
        self.domain = FEDomain('model_domain', loaded_mesh)

        del mesh

    def define_field_variable(self, field_var_name: str, field_name: str):
        if not self.__overall_volume:
            self.__assign_regions()
        if field_name not in self.fields.keys():
            self.fields[field_name] = Field.from_args(field_name, dtype=np.float64, shape=(1, ), region=self.__overall_volume, approx_order=1)

        self.field_variables[field_var_name] = {
            'unknown': FieldVariable(field_var_name, 'unknown', field=self.fields[field_name]),
            'test': FieldVariable(field_var_name + '_test', 'test', field=self.fields[field_name], primary_var_name=field_var_name),
        }

    def define_essential_boundary(self, region_name: str, group_id: int, field_variable: str, potential: float = None, current: float = None):
        assert field_variable in self.field_variables.keys(), 'The field variable {} is not defined'.format(field_variable)
        assert (potential is not None) ^ (current is not None), 'Only potential or current value shall be provided.'

        if current is not None:
            try:
                self.__electrode_currents[field_variable][region_name] = current
            except KeyError:
                self.__electrode_currents[field_variable] = {region_name: current}
            potential = 1 if current > 0 else 0

        self.essential_boundary_ids[region_name] = group_id
        temporary_domain = self.domain.create_region(region_name, 'vertices of group ' + str(group_id), 'facet', add_to_regions=False)
        self.essential_boundaries.append(EssentialBC(region_name, temporary_domain, {field_variable + '.all' : potential}))

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

    def run_solver(self, save_results: bool, post_process_calculation=True, field_calculation: list = ['E'], output_dir=None, output_file_name=None):
        if not self.__non_linear_solver:
            raise AttributeError('The solver is not setup. Please set it up before calling run.')
        self.__material_definition()

        if field_calculation:
            self.__fields_to_calculate = field_calculation

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
        self.__overall_surface = self.domain.create_region('Gamma', 'all', 'facet')

        for region in self.__settings[self.__settings_header][self.__selected_model]['regions'].items():
            self.domain.create_region(region[0], 'cells of group ' + str(region[1]['id']))
            self.domain.create_region(region[0] + '_Gamma', 'vertices of group ' + str(region[1]['id']), 'facet')
            self.conductivities[region[0]] = region[1]['conductivity']

        for electrode in self.__settings[self.__settings_header]['electrodes'][self.electrode_system].items():
            self.domain.create_region(electrode[0], 'cells of group ' + str(electrode[1]['id']))
            self.domain.create_region(electrode[0] + '_Gamma', 'vertices of group ' + str(electrode[1]['id']), 'facet')
            self.conductivities[electrode[0]] = self.__settings[self.__settings_header]['electrodes']['conductivity']

    def __get_conductivity(self, ts, coors, mode=None, equations=None, term=None, problem=None, conductivities=None):
        # Execute only once at the initialization
        if mode == 'qp':
            values = np.empty(int(coors.shape[0]/4)) # Each element corresponds to one coordinate of the respective tetrahedral edge
            num_nodes, dim = coors.shape

            # Save the conductivity values
            for domain in problem.domain.regions:
                if domain.name in conductivities.keys():
                    values[domain.entities[3]] = conductivities[domain.name]

            # Values used in a matrix format for the material
            tmp_fun = lambda x, dim: x*np.eye(dim) # Required for the diffusion velocity in the current density calculation

            values = np.repeat(values, 4) # Account for the tetrahedral edges
            mat_vec = np.array(list(map(tmp_fun, values, np.repeat(dim, num_nodes))))
            values.shape = (coors.shape[0], 1, 1)

            return {'val' : values, 'mat_vec': mat_vec}

    def __post_process(self, out, problem, state, extend=False):
        electrode_conductivity = self.__settings[self.__settings_header]['electrodes']['conductivity']
        electrode_material = Material('electrode', kind='stationary', values={'mat_vec': electrode_conductivity*np.eye(3)})
        
        # Save the output
        for field_variable_name in self.field_variables.keys():
            if self.units == 'mm':
                distance_unit_multiplier = 1000
            else:
                distance_unit_multiplier = 1
            
            if self.__electrode_currents:
                mean_current = 0
                currents = list(self.__electrode_currents[field_variable_name].values())
                regions = list(self.__electrode_currents[field_variable_name].keys())
                assert np.sum(currents) == 0, 'The currents must sum to zero. The current sum is {}'.format(np.sum(currents))

                for region in regions:
                    flux = 1./distance_unit_multiplier * problem.evaluate('d_surface_flux.2.' + region + '_Gamma(electrode.mat_vec, ' + field_variable_name + ')', mode='eval', copy_materials=False, electrode=electrode_material)
                    area = 1./(distance_unit_multiplier**2) * problem.evaluate('d_surface.2.' + region + '_Gamma(' + field_variable_name + ')', mode='eval')
                    mean_current += np.abs(flux*area)
                mean_current = mean_current/len(regions)
                distance_unit_multiplier *= np.abs(currents[0]*0.001)/mean_current # Current is always in mA

            if 'E' in np.char.upper(self.__fields_to_calculate):
                output_var_name = 'e_field_(' + field_variable_name + ')'
                e_field = distance_unit_multiplier * problem.evaluate('-ev_grad.2.Omega(' + field_variable_name + ')', mode='qp')
                out[output_var_name] = Struct(name=output_var_name, mode='cell', data=e_field, dofs=None)
            if 'J' in np.char.upper(self.__fields_to_calculate):
                output_var_name = 'j_field_(' + field_variable_name + ')'
                j_field = distance_unit_multiplier * problem.evaluate('ev_diffusion_velocity.2.Omega(conductivity.mat_vec, ' + field_variable_name + ')', mode='qp', copy_materials=False)
                out[output_var_name] = Struct(name=output_var_name, mode='cell', data=j_field, dofs=None)

        return out