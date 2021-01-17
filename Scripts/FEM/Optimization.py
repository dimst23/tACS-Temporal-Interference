import os
import gc
import numpy as np
from datetime import datetime
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds
from geneticalgorithm import geneticalgorithm as ga

import FEM.Solver as solver
import Meshing.modulation_envelope as mod_env


class Optimization(solver.Solver):
    def initialization(self, model_name, max_solver_iterations=250, solver_relative_tol=1e-7, solver_absolute_tol=1e-3):
    #def initialization(self, calc_func, constraints_func, max_solver_iterations=250, solver_relative_tol=1e-7, solver_absolute_tol=1e-3):
        self.load_mesh(model_name)
        #self.define_field_variable('potential_base', 'voltage')
        #self.define_field_variable('potential_df', 'voltage')
        self.solver_setup(max_solver_iterations, solver_relative_tol, solver_absolute_tol)
        #self.calc_func = calc_func
        #self.constraints_func = constraints_func

    def objective_function(self, x):
        # TODO: Constraints and calculation are functions
        pen = 0
        if np.unique(np.round(x), return_counts=True)[1].size != 4:
            pen = 500 + 1000*(2 + x[0] - x[1] + x[2] - x[3])
            print("\nVariables (Penalty): {} {} {} {}\n".format(x[0], x[1], x[2], x[3]))

            with open('', "a") as fl:
                date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                fl.write("{};{} {} {} {};{};{}\n".format(date_time, *x, pen, 1))
            gc.collect()
            return round(pen, 5)

        self.essential_boundaries.clear()
        self.fields.clear()
        self.field_variables.clear()

        self.define_field_variable('potential_base', 'voltage')
        self.define_field_variable('potential_df', 'voltage')

        self.define_essential_boundary('Base_VCC', int(x[0]), 'potential_base', 150.0)
        self.define_essential_boundary('Base_GND', int(x[1]), 'potential_base', -150.0)
        self.define_essential_boundary('DF_VCC', int(x[2]), 'potential_df', 150.0)
        self.define_essential_boundary('DF_GND', int(x[3]), 'potential_df', -150.0)

        solution = self.run_solver(save_results=False, post_process_calculation=False)

        e_field_base = self.problem.evaluate('-ev_grad.2.Omega(potential_base)', mode='qp')
        e_field_df = self.problem.evaluate('-ev_grad.2.Omega(potential_df)', mode='qp')

        modulation_values = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0])
        # self.calc_func
        # self.constraints_func

        crds = self.domain.mesh.coors

        bounding_roi = {
            'x_min': -7.0,
            'x_max': 10.0,
            'y_min': -2.0,
            'y_max': 5.5,
            'z_min': -38.0,
            'z_max': -24.0
        }

        vert_x = np.logical_and(crds[:, 0] >= bounding_roi['x_min'], crds[:, 0] <= bounding_roi['x_max'])
        vert_y = np.logical_and(crds[:, 1] >= bounding_roi['y_min'], crds[:, 1] <= bounding_roi['y_max'])
        vert_z = np.logical_and(crds[:, 2] >= bounding_roi['z_min'], crds[:, 2] <= bounding_roi['z_max'])

        # Get the ROI vertex indices
        vert_id_roi = np.arange(crds.shape[0])
        roi_ids = (vert_x * vert_y * vert_z > 0)

        common_points = np.isin(self.domain.mesh.get_conn('3_4'), vert_id_roi[roi_ids])
        common_points = np.where(np.sum(common_points, axis=1) == 4)[0]

        roi_value = np.average(modulation_values[common_points])

        region_names = ['GM', 'WM']
        sourounding_cells = []
        for region in self.domain.regions:
            if region.name in region_names:
                sourounding_cells.extend(region.cells.tolist())
        sourounding_cells = np.array(sourounding_cells)
        s_cells = sourounding_cells[np.isin(sourounding_cells, common_points, invert=True)]
        sourounding_value = np.average(modulation_values[s_cells])

        mod_val = round(sourounding_value/roi_value, 5)
        print("\nVariables (Correct): {} {} {} {}\n".format(x[0], x[1], x[2], x[3]))

        with open('', "a") as fl:
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            fl.write("{};{} {} {} {};{};{}\n".format(date_time, *x, mod_val, 0))
        gc.collect()

        return mod_val

    def run_optimization(self, boundaries):
        # TODO: Make variable count automatic
        return ga(function=self.objective_function, dimension=4, variable_type='int', variable_boundaries=boundaries, function_timeout=500, convergence_curve=False)
