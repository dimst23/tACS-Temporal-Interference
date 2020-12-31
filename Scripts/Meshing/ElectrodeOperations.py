import pymesh
import numpy as np

class ElectrodeOperations:
    def __init__(self, surface_mesh: pymesh.Mesh, electrode_attributes: dict):
        self.surface_mesh = surface_mesh
        self.surface_mesh.enable_connectivity() # Required by some functions below
        self.electrode_attributes = electrode_attributes
        self.electrode_array = {}

    def add_electrodes_on_skin(self):
        if not self.electrode_array:
            raise AttributeError('The electrodes shall be positioned first before added on the surface. Please call the positioning function first.')
        electrode_mesh = self.get_electrode_single_mesh()
        # Get the surface outline including the electrode
        model = pymesh.merge_meshes((self.surface_mesh, electrode_mesh))
        outer_hull = pymesh.compute_outer_hull(model)

        # Create the surface with the electrode mesh imprinted
        electrode_tan_mesh = pymesh.boolean(electrode_mesh, self.surface_mesh, 'difference')
        outer_diff = pymesh.boolean(outer_hull, electrode_tan_mesh, 'difference')
        conditioned_surface = pymesh.merge_meshes((outer_diff, electrode_tan_mesh))

        # Generate the surface with the electrode on
        face_id = np.arange(conditioned_surface.num_faces)
        conditioned_surface = pymesh.remove_duplicated_vertices(conditioned_surface)[0] # Remove any duplicate vertices

        return [pymesh.submesh(conditioned_surface, np.isin(face_id, pymesh.detect_self_intersection(conditioned_surface)[:, 0], invert=True), 0), outer_diff]  # Get rid of the duplicate faces on the tangent surface, without merging the points

    def standard_electrode_positioning(self, width=4, radius=4, elements=150):
        closest_point = pymesh.distance_to_mesh(self.surface_mesh, self.electrode_attributes['coordinates'])[1] # Get the closest point to the one provided

        i = 0
        for electrode_name in self.electrode_attributes['names']:
            p_i = self.surface_mesh.vertices[self.surface_mesh.faces[closest_point[i]]][0] # Get the surface vertex coordinates
            electrode_orientation = self.__orient_electrode(p_i) # Orient the electrode perpendicular to the surface

            # Generate the electrode cylinder and save to the output dictionary
            electrode_cylinder = pymesh.generate_cylinder(p_i - (width * electrode_orientation)/4., p_i + (width * electrode_orientation)/4., radius, radius, elements)
            self.electrode_array[electrode_name] = electrode_cylinder
            i = i + 1

    def get_electrode_array(self):
        if not self.electrode_array:
            raise AttributeError('Electrodes are not positioned yet. Please call the positioning function.')
        return self.electrode_array

    def get_electrode_single_mesh(self):
        if not self.electrode_array:
            raise AttributeError('Electrodes are not positioned yet. Please call the positioning function.')
        return pymesh.merge_meshes((e_mesh for e_mesh in self.electrode_array.values()))

    def __orient_electrode(self, init_point):
        """[summary]

        Args:
            init_point ([type]): [description]

        Returns:
            [type]: [description]
        """
        point_id = np.where(np.sum(self.surface_mesh.vertices == init_point, axis=1))[0][0] # Unique point assumed
        face = self.surface_mesh.get_vertex_adjacent_faces(point_id)[0]

        points = []
        for point in self.surface_mesh.vertices[self.surface_mesh.faces[face]]:
            if np.sum(point != init_point):
                points.append(point)
        p_1 = points[0] - init_point
        p_2 = points[1] - init_point

        normal = np.cross(p_1, p_2)
        return normal/np.linalg.norm(normal)

