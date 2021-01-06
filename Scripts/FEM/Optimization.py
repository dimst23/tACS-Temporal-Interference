import os
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds
from geneticalgorithm import geneticalgorithm as ga

import FEM.Solver as solver
import Meshing.modulation_envelope as mod_env


class Optimization(solver.Solver):
    def initialization(self, calc_func, constraints_func, max_solver_iterations=250, solver_relative_tol=1e-7, solver_absolute_tol=1e-3):
        self.load_mesh()
        self.define_field_variable('potential_base', 'voltage')
        self.define_field_variable('potential_df', 'voltage')
        self.solver_setup(max_solver_iterations, solver_relative_tol, solver_absolute_tol)
        self.calc_func = calc_func
        self.constraints_func = constraints_func

    def objective_function(self):
        # TODO: Constraints and calculation are functions
        solution = self.run_solver(post_process_calculation=False)

        e_field_base = solution[1].evaluate('-ev_grad.2.Omega(potential_base)', mode='qp')
        e_field_df = solution[1].evaluate('-ev_grad.2.Omega(potential_df)', mode='qp')

        modulation_values = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0])
        self.calc_func
        self.constraints_func

    def run_optimization(self, boundaries):
        # TODO: Make variable count automatic
        model = ga(function=self.objective_function, dimension=19, variable_type='int', variable_boundaries=boundaries)
