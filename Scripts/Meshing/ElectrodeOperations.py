import pymesh
import numpy as np

class ElectrodeOperations:
    def __init__(self, surface_mesh: pymesh.Mesh, electrode_attributes: dict):
        self.electrode_attributes = electrode_attributes
        self.surface_mesh = surface_mesh
        self.electrode_array = {}

    def orient_electrode(self, init_point):
        """Orient the electrode along the surface. Connectivy shall be enabled in the mesh that has been given.

        Args:
            mesh ([type]): [description]
            init_point ([type]): [description:

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

    def add_electrodes(self):
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

    def standard_electrode_positioning(self, width=3, radius=4, elements=150):
        closest_point = pymesh.distance_to_mesh(self.surface_mesh, self.electrode_attributes['coordinates'])[1] # Get the closest point to the one provided

        i = 0
        for electrode_name in self.electrode_attributes['names']:
            p_i = self.surface_mesh.vertices[self.surface_mesh.faces[closest_point[i]]][0] # Get the surface vertex coordinates
            electrode_orientation = self.orient_electrode(p_i) # Orient the electrode perpendicular to the surface

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

    # def region_points(boundary_surfaces: list, electrode_mesh, shift: float):
    #     points = {}
    #     i = 1
    #     for surface in boundary_surfaces:
    #         vertices = surface.vertices
    #         max_z_point_id = np.where(vertices[:, 2] == np.amax(vertices[:, 2]))[0][0]
    #         points[str(i)] = {
    #             'point': vertices[max_z_point_id] - np.absolute(np.multiply(vertices[max_z_point_id]/np.linalg.norm(vertices[max_z_point_id]), [0, 0, shift])),
    #             #'point': (1 - shift)*vertices[max_z_point_id],
    #         }
    #         i = i + 1

    #     i = 10
    #     electrode_regions = electrode_mesh.get_attribute('face_sources')
    #     electrode_mesh.add_attribute('vertex_valance')
    #     vertex_valance = electrode_mesh.get_attribute('vertex_valance')
    #     for region in np.unique(electrode_regions):
    #         faces = electrode_mesh.faces[np.where(electrode_regions == region)[0]]
    #         vertices = electrode_mesh.vertices[np.unique(faces)]
    #         max_valance_point = np.where(vertex_valance[np.unique(faces)] == np.amax(vertex_valance[np.unique(faces)]))[0][0]
    #         #print(max_valance_point)
    #         #print(np.amax(vertex_valance[np.unique(faces)]))
    #         points[str(i)] = {
    #             'point': vertices[max_valance_point] - np.multiply(vertices[max_valance_point]/np.linalg.norm(vertices[max_valance_point]), shift),
    #             #'point': (1 - shift)*vertices[max_valance_point],
    #         }
    #         i = i + 1

    #     return points
    #     # For the boundaries find the max z and then reduce the required amount from that in order to have the point IN the desired region for the poly file to be meshed with the correct label
    #     # The electrode meshes shall be identified and an inner point to be calculated
