from __future__ import absolute_import

# Add the other script file path, used for the modulation envelope
import sys
import yaml

from sfepy import data_dir
import numpy as np

with open('/mnt/c/Users/Dimitris/Nextcloud/Documents/Neuroscience Bachelor Thesis/Public Repository/tacs-temporal-interference/Scripts/FEM/sim_settings.yml') as stream:
	settings = yaml.safe_load(stream)

sys.path.append(settings['SfePy']['lib_path'])
import Meshing.modulation_envelope as mod_env

filename_mesh = settings['SfePy']['real_brain']['mesh_file']

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

############# Laplace.

## Materials
conductivity_values = {
	'Skin': 0.17,
	'Skull': 0.003504,
	'CSF': 1.776,
	'GM': 0.2391,
	'WM': 0.2651,
    'Cerebellum': 0.6597,
    'Fp1': 5.96e7, # Copper
    'Fp2': 5.96e7,
    'F7': 5.96e7,
    'F3': 5.96e7,
    'Fz': 5.96e7,
    'F4': 5.96e7,
    'F8': 5.96e7,
    'T7': 5.96e7,
    'C3': 5.96e7,
    'Cz': 5.96e7,
    'C4': 5.96e7,
    'T8': 5.96e7,
    'P7': 5.96e7,
    'P3': 5.96e7,
    'Pz': 5.96e7,
    'P4': 5.96e7,
    'P8': 5.96e7,
    'O1': 5.96e7,
    'O2': 5.96e7,
}

materials = {
	'material' : 'get_conductivity',
}
## Materials

## Regions
regions = {
	'Omega': 'all',
	'Skin' : 'cells of group 0',
	'Skull' : 'cells of group 1',
	'CSF' : 'cells of group 2',
	'GM' : 'cells of group 3',
	'WM' : 'cells of group 4',
	'Cerebellum' : 'cells of group 5',
    'Fp1' : 'cells of group 10',
    'Fp2' : 'cells of group 11',
    'F7' : 'cells of group 12',
    'F3' : 'cells of group 13',
    'Fz' : 'cells of group 14',
    'F4' : 'cells of group 15',
    'F8' : 'cells of group 16',
    'T7' : 'cells of group 17',
    'C3' : 'cells of group 18',
    'Cz' : 'cells of group 19',
    'C4' : 'cells of group 20',
    'T8' : 'cells of group 21',
    'P7' : 'cells of group 22',
    'P3' : 'cells of group 23',
    'Pz' : 'cells of group 24',
    'P4' : 'cells of group 25',
    'P8' : 'cells of group 26',
    'O1' : 'cells of group 27',
    'O2' : 'cells of group 28',
    'Gamma_Base_VCC' : ('vertices of group 25', 'facet'),
	'Gamma_Base_GND' : ('vertices of group 12', 'facet'),
	'Gamma_DF_VCC' : ('vertices of group 23', 'facet'),
	'Gamma_DF_GND' : ('vertices of group 16', 'facet'),
}
## Regions

## Fields
field_1 = {
	'name' : 'voltage',
	'dtype' : 'real',
	'shape' : (1,),
	'region' : 'Omega',
	'approx_order' : 1,
}
## Fields

## Boundary Conditions
ebc_1 = {
	'name' : 'base_vcc',
	'region' : 'Gamma_Base_VCC',
	'dofs' : {'potential_base.0' : 150.0},
}
ebc_2 = {
	'name' : 'base_gnd',
	'region' : 'Gamma_Base_GND',
	'dofs' : {'potential_base.0' : -150.0},
}

ebc_3 = {
	'name' : 'df_vcc',
	'region' : 'Gamma_DF_VCC',
	'dofs' : {'potential_df.0' : 150.0},
}
ebc_4 = {
	'name' : 'df_gnd',
	'region' : 'Gamma_DF_GND',
	'dofs' : {'potential_df.0' : -150.0},
}
## Boundary Conditions

## Variables
### Base Potential
variable_1 = {
	'name' : 'potential_base',
	'kind' : 'unknown field',
	'field' : 'voltage',
	'order' : 0, # order in the global vector of unknowns
}
variable_2 = {
	'name' : 's_base',
	'kind' : 'test field',
	'field' : 'voltage',
	'dual' : 'potential_base',
}
### Base Potential

### Difference Potential
variable_3 = {
	'name' : 'potential_df',
	'kind' : 'unknown field',
	'field' : 'voltage',
	'order' : 1, # order in the global vector of unknowns
}
variable_4 = {
	'name' : 's_df',
	'kind' : 'test field',
	'field' : 'voltage',
	'dual' : 'potential_df',
}
### Difference Potential
## Variables

## Equations
integrals = {
	'i1' : 2,
}

equations = {
	'quasi_static_base' : """
		dw_laplace.i1.Omega(material.val, s_base, potential_base) = 0
	""",
	'quasi_static_df' : """
		dw_laplace.i1.Omega(material.val, s_df, potential_df) = 0
	""",
}
## Equations

## Functions
functions = {
	'get_conductivity' : (lambda ts, coors, mode=None, equations=None, term=None, problem=None, **kwargs:
				   get_conductivity(ts, coors, mode, equations, term, problem, conductivities=conductivity_values),),
}
## Functions

## Solvers
solvers = {
	'ls' : ('ls.pyamg', {
		'i_max': 100,
		'eps_r': 1e-12,
		'method': 'ruge_stuben_solver',
	}),
	'newton' : ('nls.newton', {
		'i_max'      : 1,
		'eps_a'      : 1e-4,
		'macheps'	 : 1e-10,
	}),
}
## Solvers

options ={
	'output_dir': settings['SfePy']['real_brain']['output_dir'],
	'post_process_hook': 'post_process',
}

def post_process(out, problem, state, extend=False):
	from sfepy.base.base import Struct

	e_field_base = problem.evaluate('-ev_grad.i1.Omega(potential_base)', mode='qp')
	e_field_df = problem.evaluate('-ev_grad.i1.Omega(potential_df)', mode='qp')

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
