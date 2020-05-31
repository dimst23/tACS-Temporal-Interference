import numpy as np

# Binary save does not work yet
def gmsh_write(surfaces, domains, physical_tags, bounding_surface_tag, file_name: str, binary=False):
    c_int = np.dtype("i")
    c_size_t = np.dtype("P")
    c_double = np.dtype("d")

    with open(file_name, "wb") as m_file:
        # MeshFormat (Check)
        m_file.write(b"$MeshFormat\n")
        file_type = 1 if binary else 0
        data_size = c_size_t.itemsize
        m_file.write("4.1 {} {}\n".format(file_type, data_size).encode("utf-8"))

        if binary:
            np.array([1], dtype=c_int).tofile(m_file) # To detect endianess
            m_file.write(b"\n")
        m_file.write(b"$EndMeshFormat\n")

        # PhysicalNames (Check)
        m_file.write(b"$PhysicalNames\n")
        m_file.write("{}\n".format(len(physical_tags)).encode("utf-8"))
        for physical_tag in physical_tags:
            m_file.write('3 {} "{}"\n'.format(*physical_tag).encode("utf-8"))
        m_file.write(b"$EndPhysicalNames\n")

        # Entities (Check)
        m_file.write(b"$Entities\n")
        
        if binary:
            np.array([0, 0, len(surfaces), len(domains)], dtype=c_size_t).tofile(m_file)
            for i in range(0, len(surfaces)):
                np.array([i + 1], dtype=c_int).tofile(m_file)
                np.array([np.amin(surfaces[i].vertices[:, 0]), np.amin(surfaces[i].vertices[:, 1]), np.amin(surfaces[i].vertices[:, 2]), np.amax(surfaces[i].vertices[:, 0]), np.amax(surfaces[i].vertices[:, 1]), np.amax(surfaces[i].vertices[:, 2])], dtype=c_double).tofile(m_file)
                np.array([0, 0], dtype=c_int).tofile(m_file)
            m_file.write(b"\n")
            for i in range(0, len(domains)):
                np.array([i + 1], dtype=c_int).tofile(m_file)
                np.array([np.amin(domains[i].vertices[:, 0]), np.amin(domains[i].vertices[:, 1]), np.amin(domains[i].vertices[:, 2]), np.amax(domains[i].vertices[:, 0]), np.amax(domains[i].vertices[:, 1]), np.amax(domains[i].vertices[:, 2])], dtype=c_double).tofile(m_file)
                np.array([1], dtype=c_size_t).tofile(m_file)
                np.array([physical_tags[i][0]], dtype=c_int).tofile(m_file)
                np.array([1], dtype=c_size_t).tofile(m_file)
                np.array([bounding_surface_tag[i]], dtype=c_int).tofile(m_file)
            m_file.write(b"\n")
        else:
            m_file.write("0 0 {} {}\n".format(len(surfaces), len(domains)).encode("utf-8"))
            for i in range(0, len(surfaces)):
                m_file.write("{} {} {} {} {} {} {} {} {}\n".format(i + 1, np.amin(surfaces[i].vertices[:, 0]), np.amin(surfaces[i].vertices[:, 1]), np.amin(surfaces[i].vertices[:, 2]), np.amax(surfaces[i].vertices[:, 0]), np.amax(surfaces[i].vertices[:, 1]), np.amax(surfaces[i].vertices[:, 2]), 0, 0).encode("utf-8"))
            for i in range(0, len(domains)):
                m_file.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(i + 1, np.amin(domains[i].vertices[:, 0]), np.amin(domains[i].vertices[:, 1]), np.amin(domains[i].vertices[:, 2]), np.amax(domains[i].vertices[:, 0]), np.amax(domains[i].vertices[:, 1]), np.amax(domains[i].vertices[:, 2]), 1, physical_tags[i][0], 1, bounding_surface_tag[i]).encode("utf-8"))
        m_file.write(b"$EndEntities\n")

        # Nodes (Checked?)
        float_fmt=".16e"

        surfs = 0
        voxels = 0
        for surf in surfaces:
            surfs = surfs + surf.num_vertices
        for domain in domains:
            voxels = voxels + domain.num_vertices

        m_file.write(b"$Nodes\n")
        num_blocks = len(surfaces) + len(domains)
        min_tag = 1
        max_tag = surfs + voxels

        if binary:
            np.array([num_blocks, surfs + voxels, min_tag, max_tag], dtype=c_size_t).tofile(m_file)
            m_file.write(b"\n")
        else:
            m_file.write("{} {} {} {}\n".format(num_blocks, surfs + voxels, min_tag, max_tag).encode("utf-8"))

        ta = 1
        face_elements = []
        for i in range(0, len(surfaces)):
            # dim_entity, is_parametric
            num_elem = surfaces[i].num_vertices

            if binary:
                np.array([2, i + 1, 0], dtype=c_int).tofile(m_file)
                np.array([num_elem], dtype=c_size_t).tofile(m_file)
                #m_file.write(b"\n")
                np.arange(ta, ta + num_elem, dtype=c_size_t).tofile(m_file)
                #m_file.write(b"\n")
                surfaces[i].vertices.astype(c_double).tofile(m_file)
                face_elements.append(surfaces[i].faces.astype(c_size_t) + 1 + (ta - 1))
            else:
                m_file.write("{} {} {} {}\n".format(2, i + 1, 0, num_elem).encode("utf-8"))
                np.arange(ta, ta + num_elem).tofile(m_file, "\n", "%d")
                m_file.write(b"\n")
                np.savetxt(m_file, surfaces[i].vertices, delimiter=" ", fmt="%" + float_fmt)
                #face_elements.append(surfaces[i].faces + 1 + (ta - 1))
                face_elements.append(surfaces[i].faces.astype(c_size_t) + 1 + (ta - 1))
            ta = ta + num_elem
            m_file.write(b"\n")

        voxel_elements = []
        for i in range(0, len(domains)):
            num_elem = domains[i].num_vertices

            if binary:
                np.array([3, i + 1, 0], dtype=c_int).tofile(m_file)
                np.array([num_elem], dtype=c_size_t).tofile(m_file)
                #m_file.write(b"\n")
                np.arange(ta, ta + num_elem, dtype=c_size_t).tofile(m_file)
                #m_file.write(b"\n")
                domains[i].vertices.astype(c_double).tofile(m_file)
                voxel_elements.append(domains[i].voxels.astype(c_size_t) + 1 + (ta - 1))
            else:
                m_file.write("{} {} {} {}\n".format(3, i + 1, 0, num_elem).encode("utf-8"))
                np.arange(ta, ta + num_elem).tofile(m_file, "\n", "%d")
                m_file.write(b"\n")
                np.savetxt(m_file, domains[i].vertices, delimiter=" ", fmt="%" + float_fmt)
                #voxel_elements.append(domains[i].voxels + 1 + (ta - 1))
                voxel_elements.append(domains[i].voxels.astype(c_size_t) + 1 + (ta - 1))
            ta = ta + num_elem
            m_file.write(b"\n")
        m_file.write(b"$EndNodes\n")

        # Elements
        surfs = 0
        voxels = 0
        for surf in surfaces:
            surfs = surfs + surf.num_faces
        for domain in domains:
            voxels = voxels + domain.num_voxels

        m_file.write(b"$Elements\n")
        num_blocks = len(face_elements) + len(voxel_elements)
        min_tag = 1
        max_tag = surfs + voxels

        if binary:
            np.array([num_blocks, surfs + voxels, min_tag, max_tag], dtype=c_size_t).tofile(m_file)
            m_file.write(b"\n")
        else:
            m_file.write("{} {} {} {}\n".format(num_blocks, surfs + voxels, min_tag, max_tag).encode("utf-8"))

        ta = 1
        for i in range(0, len(face_elements)):
            num_elem = face_elements[i].shape[0]

            if binary:
                np.array([2, i + 1, 2], dtype=c_int).tofile(m_file)
                np.array([num_elem], dtype=c_size_t).tofile(m_file)
                #m_file.write(b"\n")
                #np.arange(ta, ta + num_elem, dtype=c_size_t).tofile(m_file)
                np.column_stack([np.arange(ta, ta + num_elem, dtype=c_size_t), face_elements[i]]).tofile(m_file)
            else:
                m_file.write("{} {} {} {}\n".format(2, i + 1, 2, num_elem).encode("utf-8"))
                np.savetxt(m_file, np.column_stack([np.arange(ta, ta + num_elem), face_elements[i]]), "%d", " ")
            ta = ta + num_elem
            m_file.write(b"\n")

        for i in range(0, len(voxel_elements)):
            num_elem = voxel_elements[i].shape[0]

            if binary:
                np.array([3, i + 1, 4], dtype=c_int).tofile(m_file)
                np.array([num_elem], dtype=c_size_t).tofile(m_file)
                #m_file.write(b"\n")
                #np.arange(ta, ta + num_elem, dtype=c_size_t).tofile(m_file)
                np.column_stack([np.arange(ta, ta + num_elem, dtype=c_size_t), voxel_elements[i]]).tofile(m_file)
            else:
                m_file.write("{} {} {} {}\n".format(3, i + 1, 4, num_elem).encode("utf-8"))
                np.savetxt(m_file, np.column_stack([np.arange(ta, ta + num_elem), voxel_elements[i]]), "%d", " ")
            ta = ta + num_elem
            m_file.write(b"\n")
        m_file.write(b"$EndElements\n")
