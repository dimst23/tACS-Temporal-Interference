# Transcranial Temporal Interference Stimulation

This project is part of the Bachelor thesis during the studies at the Physics Department of the Aristotle University of Thessaloniki, Greece.

## Short description

Simulations of the temporally interfering electric field distribution are conducted on the human brain using a _Simple Anthropomorphic model_ **(SAM)** and realistic brain models from the [PHM](https://itis.swiss/virtual-population/regional-human-models/phm-repository/) repository. The different material conductivities are drawn from the [IT'IS Virtual Population Tissue Properties](https://itis.swiss/virtual-population/tissue-properties/database/low-frequency-conductivity/) or from different papers, referenced accordingly.

## Further Information

In this repository there is a Wiki in which you can find detailed information about the code and the approaches taken to solve the problems.

## Structure of the Repository

* [assets/](.\tacs-temporal-interference\assets)
  * [images/](.\tacs-temporal-interference\assets\images)
* [CAD/](.\tacs-temporal-interference\CAD)
  * [SAM/](.\tacs-temporal-interference\CAD\SAM)
  * [Sphere/](.\tacs-temporal-interference\CAD\Sphere)
    * [spheres_brain.stl](.\tacs-temporal-interference\CAD\Sphere\spheres_brain.stl)
    * [spheres_csf.stl](.\tacs-temporal-interference\CAD\Sphere\spheres_csf.stl)
    * [spheres_outer.stl](.\tacs-temporal-interference\CAD\Sphere\spheres_outer.stl)
    * [spheres_skin.stl](.\tacs-temporal-interference\CAD\Sphere\spheres_skin.stl)
    * [spheres_skull.stl](.\tacs-temporal-interference\CAD\Sphere\spheres_skull.stl)
* [Jupyter Notebooks/](.\tacs-temporal-interference\Jupyter Notebooks)
  * [Test Bench/](.\tacs-temporal-interference\Jupyter Notebooks\Test Bench)
    * [modulation_envelope_tests.ipynb](.\tacs-temporal-interference\Jupyter Notebooks\Test Bench\modulation_envelope_tests.ipynb)
    * [README.md](.\tacs-temporal-interference\Jupyter Notebooks\Test Bench\README.md)
  * [fem_simulation_analysis.ipynb](.\tacs-temporal-interference\Jupyter Notebooks\fem_simulation_analysis.ipynb)
  * [modulation_envelope.ipynb](.\tacs-temporal-interference\Jupyter Notebooks\modulation_envelope.ipynb)
  * [sim_analysis.ipynb](.\tacs-temporal-interference\Jupyter Notebooks\sim_analysis.ipynb)
* [MATLAB Workspaces and Scripts/](.\tacs-temporal-interference\MATLAB Workspaces and Scripts)
  * [Workspaces/](.\tacs-temporal-interference\MATLAB Workspaces and Scripts\Workspaces)
    * [BaseFrequency4Layer_Smaller.mat](.\tacs-temporal-interference\MATLAB Workspaces and Scripts\Workspaces\BaseFrequency4Layer_Smaller.mat)
    * [DeltaFrequency4Layer_Smaller.mat](.\tacs-temporal-interference\MATLAB Workspaces and Scripts\Workspaces\DeltaFrequency4Layer_Smaller.mat)
  * [arrange_elements.m](.\tacs-temporal-interference\MATLAB Workspaces and Scripts\arrange_elements.m)
  * [grid_points.m](.\tacs-temporal-interference\MATLAB Workspaces and Scripts\grid_points.m)
* [Scripts/](.\tacs-temporal-interference\Scripts)
  * [Archive/](.\tacs-temporal-interference\Scripts\Archive)
    * [FEM.py](.\tacs-temporal-interference\Scripts\Archive\FEM.py)
    * [meshing.py](.\tacs-temporal-interference\Scripts\Archive\meshing.py)
    * [simple.py](.\tacs-temporal-interference\Scripts\Archive\simple.py)
    * [simple_meshing.py](.\tacs-temporal-interference\Scripts\Archive\simple_meshing.py)
  * [FEM/](.\tacs-temporal-interference\Scripts\FEM)
    * [Good Files/](.\tacs-temporal-interference\Scripts\FEM\Good Files)
    * [real_head.py](.\tacs-temporal-interference\Scripts\FEM\real_head.py)
    * [real_head_10-20.py](.\tacs-temporal-interference\Scripts\FEM\real_head_10-20.py)
    * [sim_settings.yml](.\tacs-temporal-interference\Scripts\FEM\sim_settings.yml)
    * [sphere.py](.\tacs-temporal-interference\Scripts\FEM\sphere.py)
  * [GMSH/](.\tacs-temporal-interference\Scripts\GMSH)
    * [spheres.geo](.\tacs-temporal-interference\Scripts\GMSH\spheres.geo)
  * [Meshing/](.\tacs-temporal-interference\Scripts\Meshing)
    * [electrode_operations.py](.\tacs-temporal-interference\Scripts\Meshing\electrode_operations.py)
    * [gmsh_write.py](.\tacs-temporal-interference\Scripts\Meshing\gmsh_write.py)
    * [mesh_operations.py](.\tacs-temporal-interference\Scripts\Meshing\mesh_operations.py)
    * [modulation_envelope.py](.\tacs-temporal-interference\Scripts\Meshing\modulation_envelope.py)
    * [phm_model_meshing.py](.\tacs-temporal-interference\Scripts\Meshing\phm_model_meshing.py)
  * [Utils/](.\tacs-temporal-interference\Scripts\Utils)
    * [mesh_fixing.py](.\tacs-temporal-interference\Scripts\Utils\mesh_fixing.py)
  * [10-20_elec.mat](.\tacs-temporal-interference\Scripts\10-20_elec.mat)
  * [main.py](.\tacs-temporal-interference\Scripts\main.py)
  * [sphere_meshing.py](.\tacs-temporal-interference\Scripts\sphere_meshing.py)
* [LICENSE](.\tacs-temporal-interference\LICENSE)
* [README.md](.\tacs-temporal-interference\README.md)

