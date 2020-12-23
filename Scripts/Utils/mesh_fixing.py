import os
import gc
import multiprocessing
import meshio

base_path = r"/mnt/c/Users/Dimitris/Downloads/Binary_STL"
folders = next(os.walk(base_path))[1]

def fixing(folder):
    global base_path
    for file_n in next(os.walk(os.path.join(base_path, folder)))[2]:
        print("\nConverting to ASCII file")
        temp_file = meshio.read(os.path.join(base_path, folder, file_n))
        temp_file.write(os.path.join(base_path, folder, file_n))
        del temp_file

        print("Fixing")
        os.system(r"/home/dimitris/repos/MeshFix-V2.1/bin64/MeshFix " + os.path.join(base_path, folder, file_n) + r" " + os.path.join(base_path, folder, file_n.split('.')[0] + '_fixed.stl') + r" -j")
    return 0

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = 16)
    pool.map_async(fixing, folders)
