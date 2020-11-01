import os

base_path = r"C:\Users\Dimitris\Documents\Brains"
folders = next(os.walk(base_path))[1]

for folder in folders:
	for file in next(os.walk(os.path.join(base_path, folder)))[2]:
		os.system(r"D:\Thesis\Tests\model\MeshFix-V2.1\bin64\MeshFix.exe " + os.path.join(base_path, folder, file) + r" " + os.path.join(base_path, folder, file.split('.')[0] + '_fixed.stl') + r" -j")
