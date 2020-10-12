from __future__ import absolute_import
from sfepy import data_dir
import numpy as np

filename_mesh = '/mnt/c/Users/Dimitris/Nextcloud/Documents/Neuroscience Bachelor Thesis/Public Repository/tacs-temporal-interference/Scripts/msh.vtk'

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
	'Skin': 0.3300,
	'Skull': 0.0042,
	'CSF': 1.776,
	'Base_VCC': 5.96e7, # Copper
	'Base_GND': 5.96e7,
	'DF_VCC': 5.96e7,
	'DF_GND': 5.96e7,
}

materials = {
	'material' : 'get_conductivity',
}
## Materials

## Regions
regions = {
	'Omega': 'all',
	'Skin' : 'cells of group 1',
	'Skull' : 'cells of group 2',
	'CSF' : 'cells of group 3',
	'Base_VCC' : 'cells of group 4',
	'Base_GND' : 'cells of group 5',
	'DF_VCC' : 'cells of group 6',
	'DF_GND' : 'cells of group 7',
	'Gamma_Base_VCC' : ('cells of group 1 *v cells of group 4', 'facet'),
	'Gamma_Base_GND' : ('cells of group 1 *v cells of group 5', 'facet'),
	'Gamma_DF_VCC' : ('cells of group 1 *v cells of group 6', 'facet'),
	'Gamma_DF_GND' : ('cells of group 1 *v cells of group 7', 'facet'),
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
	'dofs' : {'potential_base.0' : 10.0},
}
ebc_2 = {
	'name' : 'base_gnd',
	'region' : 'Gamma_Base_GND',
	'dofs' : {'potential_base.0' : 0},
}

ebc_3 = {
	'name' : 'df_vcc',
	'region' : 'Gamma_DF_VCC',
	'dofs' : {'potential_df.0' : 10.0},
}
ebc_4 = {
	'name' : 'df_gnd',
	'region' : 'Gamma_DF_GND',
	'dofs' : {'potential_df.0' : 0},
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
	'ls' : ('ls.scipy_direct', {}),
	'newton' : ('nls.newton', {
		'i_max'      : 4,
		'eps_a'      : 1e-5,
	}),
}
## Solvers
