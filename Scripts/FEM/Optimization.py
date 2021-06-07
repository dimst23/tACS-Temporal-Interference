import os
import gc
import numpy as np
import pyvista as pv
from datetime import datetime
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds
from geneticalgorithm import geneticalgorithm as ga

import FEM.Solver as solver
import Meshing.modulation_envelope as mod_env


class Optimization(solver.Solver):
    def __init__(self,settings_file: dict, settings_header: str, electrode_system: str, units: str = 'mm') -> None:
        super().__init__(settings_file, settings_header, electrode_system, units)
        self.__region_volumes = {}
        self.__AAL_regions = None
        self.__settings_file = settings_file
        self.__settings_header = settings_header

    def initialization(self, model_name, max_solver_iterations=250, solver_relative_tol=1e-7, solver_absolute_tol=1e-3):
        self.load_mesh(model_name)
        self.solver_setup(max_solver_iterations, solver_relative_tol, solver_absolute_tol)

        mesh = pv.UnstructuredGrid(self.__settings_file[self.__settings_header][model_name]['mesh_file'])
        cell_volumes = mesh.compute_cell_sizes().cell_arrays['Volume']
        self.__AAL_regions = mesh['AAL_regions']

        for region in np.unique(self.__AAL_regions):
            roi = np.where(self.__AAL_regions == region)[0]
            self.__region_volumes[int(region)] = np.sum(cell_volumes[roi])
        
        del mesh

    def objective_df(self, x, unwanted_regions=[], regions_of_interest=[]):
        self.essential_boundaries.clear()
        self.fields.clear()
        self.field_variables.clear()

        self.define_field_variable('potential_base', 'voltage')
        self.define_field_variable('potential_df', 'voltage')

        electrodes = np.int(x[:4]) # The first 4 indices are the electrode IDs
        electrode_names = [self._electrode_names[id] for id in electrodes]
        currents = np.round(x[4:], 3)

        self.define_essential_boundary(electrode_names[0], electrodes[0], 'potential_base', current=currents[0])
        self.define_essential_boundary(electrode_names[1], electrodes[1], 'potential_base', current=-currents[0])
        self.define_essential_boundary(electrode_names[2], electrodes[2], 'potential_df', current=currents[1])
        self.define_essential_boundary(electrode_names[3], electrodes[3], 'potential_df', current=-currents[1])

        solution = self.run_solver(save_results=False, post_process_calculation=True)

        e_field_base = solution['e_field_(potential_base)'].data[:, 0, :, 0]
        e_field_df = solution['e_field_(potential_df)'].data[:, 0, :, 0]
        modulation_values = mod_env.modulation_envelope(e_field_base, e_field_df)

        unwanted_regions = [-1, -2, -3, -7]
        regions_of_interest = [42]
        roi_region_sum = 0
        non_roi_region_sum = 0

        for region in np.unique(self.__AAL_regions):
            if region in unwanted_regions:
                continue
            roi = np.where(self.__AAL_regions == region)[0]

            if region in regions_of_interest:
                roi_region_sum += np.sum(modulation_values[roi])/self.__region_volumes[region]
            non_roi_region_sum += np.sum(modulation_values[roi])/self.__region_volumes[region]
        
        return -roi_region_sum/non_roi_region_sum

    def run_optimization(self, boundaries):
        # TODO: Make variable count automatic
        return ga(function=self.objective_function, dimension=4, variable_type='int', variable_boundaries=boundaries, function_timeout=500, convergence_curve=False)
