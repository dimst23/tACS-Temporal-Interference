#!/usr/bin/env python

from __future__ import absolute_import
import sys
import yaml

from argparse import ArgumentParser, RawDescriptionHelpFormatter

#### Argument parsing
helps = {
	'settings-file' : "File having the settings to be loaded",
	'model' : "Name of the model. Selection from the settings file",
}

parser = ArgumentParser(description=__doc__,
						formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('--version', action='version', version='%(prog)s')
parser.add_argument('--settings-file', metavar='str', type=str,
					action='store', dest='settings_file',
					default=None, help=helps['settings-file'])
parser.add_argument('--model', metavar='str', type=str,
					action='store', dest='model',
					default='brain', help=helps['model'])
options = parser.parse_args()
#### Argument parsing

with open(options.settings_file) as stream:
	settings = yaml.safe_load(stream)

sys.path.append(settings['SfePy']['lib_path'])
import Meshing.modulation_envelope as mod_env

import numpy as np

#### SfePy libraries
from sfepy.base.base import output, IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Integrals, Equation, Equations, Problem, Function)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyUmfpack, PyAMGSolver
from sfepy.solvers.nls import Newton
#### SfePy libraries

def get_conductivity(ts, coors, mode=None, equations=None, term=None, problem=None, conductivities=None):
	"""[summary]

	Args:
		ts ([type]): [description]
		coors ([type]): [description]
		mode ([type], optional): [description]. Defaults to None.
		equations ([type], optional): [description]. Defaults to None.
		term ([type], optional): [description]. Defaults to None.
		problem ([type], optional): [description]. Defaults to None.
		conductivities ([type], optional): [description]. Defaults to None.

	Returns:
		[type]: [description]
	"""
	# Execute only once at the initialization
	if mode == 'qp':
		print(coors)
		print(coors.shape)
		values = np.empty(int(coors.shape[0]/4)) # Each element corresponds to one coordinate of the respective tetrahedral edge

		# Save the conductivity values
		for domain in problem.domain.regions:
			if domain.name in conductivities.keys():
				values[domain.entities[3]] = conductivities[domain.name]

		values = np.repeat(values, 4) # Account for the tetrahedral edges
		values.shape = (coors.shape[0], 1, 1)
		
		return {'val' : values}


mesh = Mesh.from_file(settings['SfePy'][options.model]['mesh_file'])
domain = FEDomain('domain', mesh)

conductivities = {} # Empty conductivity dictionaries

#### Region definition
overall_volume = domain.create_region('Omega', 'all')

for region in settings['SfePy']['regions'].items():
	domain.create_region(region[0], 'cells of group ' + str(region[1]['id']))
	conductivities[region[0]] = region[1]['conductivity']

for electrode in settings['SfePy']['electrodes'].items():
	if electrode[0] != 'conductivity':
		domain.create_region(electrode[0], 'cells of group ' + str(region[1]['id']))
	else:
		conductivities[electrode[0]] = electrode[1]['conductivity']

r_base_vcc = domain.create_region('Base_VCC', 'vertices of group 25', 'facet')
r_base_gnd = domain.create_region('Base_GND', 'vertices of group 12', 'facet')
r_df_vcc = domain.create_region('DF_VCC', 'vertices of group 23', 'facet')
r_df_gnd = domain.create_region('DF_GND', 'vertices of group 16', 'facet')
#### Region definition

#### Material definition
conductivity = Material('conductivity', function=Function('get_conductivity', lambda ts, coors, mode=None, equations=None, term=None, problem=None, **kwargs: get_conductivity(ts, coors, mode, equations, term, problem, conductivities=conductivities)))
#conductivity = Material('conductivities', function=Function(domain, settings['SfePy'][options.model]['conductivities']))
#### Material definition

#### Essential boundary conditions
bc_base_vcc = EssentialBC('base_vcc', r_base_vcc, {'potential_base.all' : 150.0})
bc_base_gnd = EssentialBC('base_gnd', r_base_gnd, {'potential_base.all' : -150.0})

bc_df_vcc = EssentialBC('df_vcc', r_df_vcc, {'potential_df.all' : 150.0})
bc_df_gnd = EssentialBC('df_gnd', r_df_gnd, {'potential_df.all' : -150.0})
#### Essential boundary conditions

#### Field definition
field_potential = Field.from_args('voltage', dtype=np.float64, shape=(1, ), region=overall_volume, approx_order=1)

fld_potential_base = FieldVariable('potential_base', 'unknown', field=field_potential)
fld_s_base = FieldVariable('s_base', 'test', field=field_potential, primary_var_name='potential_base')

fld_potential_df = FieldVariable('potential_df', 'unknown', field=field_potential)
fld_s_df = FieldVariable('s_df', 'test', field=field_potential, primary_var_name='potential_df')
#### Field definition

#### Equation definition
integral = Integral('i1', order=2)

laplace_base = Term.new('dw_laplace(conductivity.val, s_base, potential_base)', integral, region=overall_volume, conductivity=conductivity, potential_base=fld_potential_base, s_base=fld_s_base)
laplace_df = Term.new('dw_laplace(conductivity.val, s_df, potential_df)', integral, region=overall_volume, conductivity=conductivity, potential_df=fld_potential_df, s_df=fld_s_df)

eq_base = Equation('balance', laplace_base)
eq_df = Equation('balance', laplace_df)

equations = Equations([eq_base, eq_df])
#### Equation definition

#### Solver definition
ls = PyAMGSolver({
	'i_max': 100,
	  'eps_r': 1e-12
})

nls_status = IndexedStruct()
nls = Newton({
	'i_max'      : 1,
	'eps_a'      : 1e-4,
	'macheps'	 : 1e-10
}, lin_solver=ls, status=nls_status)
#### Solver definition

problem = Problem('temporal_interference', equations=equations)
problem.set_bcs(ebcs=Conditions([bc_base_vcc, bc_base_gnd, bc_df_vcc, bc_df_gnd]))
problem.set_solver(nls)

# Solve the problem
state = problem.solve()
output(nls_status)
