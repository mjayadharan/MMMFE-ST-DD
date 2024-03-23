# Multiscale Mortar Mixed Finite Element method using novel   Space-Time Domain Decomposition (MMMFE-ST-DD)
Fluid flow simulator using multiscale space-time domains. 

__Note__: If you use the code for research purposes, please cite the following original publications: [A space-time multiscale mortar mixed finite element method for parabolic equations
M Jayadharan, M Kern, M Vohralík, I Yotov
SIAM Journal on Numerical Analysis 61 (2), 675-706](https://epubs.siam.org/doi/abs/10.1137/21M1447945)
and
[Multiscale mortar mixed finite element methods for the Biot system of poroelasticity M Jayadharan, I Yotov arXiv preprint arXiv:2211.02949](https://arxiv.org/abs/2211.02949)

Code developed to simulate time-dependent diffusion problem using Multiscale Mortar Mixed Finite Elements(MMMFE). Model can be easily adapted to simulate other fluid flow models based on linear PDEs. The novelty of the simulator lies in using multiple subdomains with variable time steps and mesh size for each subdomain. This give rise to a space-time DD technique allowing non-matching grids for sub-domains in both space and time dimensions. Sub-domain solves are done in parallel across different processors using MPI. Computed solutions are outputted and visualized on a global space-time grid in the .vtk and .vtu formats. Details of the spaces used and rough algorithm can be found in report.pdf and algorithm.pdf respectively. Theoretical results guaranteeing convergence and stability of the problem along with a priori error estimates have been proved and will appear in SINUM journal and can also be found [here](https://arxiv.org/abs/2110.02132). 
![github_space_time_dd](https://user-images.githubusercontent.com/35903705/86996707-51287c00-c17a-11ea-8d9c-584aa2cfc47b.png)

### Note:
- The simulator is written using deal.ii FE package which is based on C++.  
-  All instructions are with respect to a terminal in linux/mac. Please use the ubuntu sub-system if you are using windows 10. A good installation guide for the linux sub-system can be found [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
- Image/animation processing and visualization can be done using paraview. Installation guide can be found [here.](https://www.paraview.org/Wiki/ParaView:Build_And_Install)

## Author
-----------
Manu Jayadharan, Department of Mathematics at University of Pittsburgh 9/17/2019

email: [manu.jayadharan@gmail.com](mailto:manu.jayadharan@gmail.com), [manu.jayadharan@pitt.edu](mailto:manu.jayadharan@pitt.edu)  
[researchgate](https://www.researchgate.net/profile/Manu_Jayadharan)  
[linkedin](https://www.linkedin.com/in/manu-jayadharan/)


## Collaborators
------------------
- Michel Kern, INRIA(French Institute for Research in Computer Science and Automation).  
Maintaining branch: *michel_Andra_testcase* .   
[INRIA webpage](https://who.rocq.inria.fr/Michel.Kern/)

### New updates: Aug 2020
* Added mixed boundary condition functionality.  
* More options to save the output at different time steps.  
* Improved user interface using parameter file.  
* Cleaned up src/darcy_main.cc file to hide more details. 




## deal.ii 9.5.2 requirement (latest at the time)
---------------------------------------
Need deal.ii configured with mpi  to compile and run the simulations. Latest version of dealii can be found at : [https://www.dealii.org/download.html](https://www.dealii.org/download.html)

**deal.ii installation instruction:** Follow [readme](https://www.dealii.org/9.2.0/readme.html) file to install latest version of deal.ii with `-DDEAL_II_WITH_MPI=ON` .. -DCMAKE_PREFIX_PATH=path_to_mpi_lib flags to cmake.   
Note that if you have trouble finding the mpi library while building, do the following, manually pass the location of the compiler files to cmake as follows:     
```
cmake -DCMAKE_C_COMPILER="</location to/mpicc"\
              -DCMAKE_CXX_COMPILER="/location to/mpicxx"\
              -DCMAKE_Fortran_COMPILER="/location to/mpif90"\ <..rest of the arguments to cmake..>

```   
A thread on how to solve this issue can be found [here](https://groups.google.com/forum/#!newtopic/dealii/dealii/y1xS0Fe-k6w).  
If you still have trouble configuring deal.ii with mpi, please seek help at this dedicated [google group](https://groups.google.com/forum/#!forum/dealii) or contact the author.  
## Compilation instructions.
-------------------------------------------
`cmake -DDEAL_II_DIR=/path to dealii installation folder/ .` from the main directory

`make release` for faster compilations

`make debug` for more careful compilations with warnings

`mpirun -n 'j' DarcyVT` where j is the number of subdomains(processses)

**Please contact the author for further instructions.**

## Quick start guide for the simulator.
-------------------------------------
* The main file is located in `src/darcy_main.cc`, which has only the least minimum info to run the code. Most of the details are hidden and are implemented in src/darcy_vtdd.cc and header files in `inc/`.    
* Compilation needs to be done once using the instructions given above and the parameter file  can be changed subsequentyly to change the behaviour of the simulator without recompiling.  
* Most of the parameters including number of refinements, mortar_degree, max_number of gmres iterations, final_time, subdomain mesh size
ratio etc are fed to the executable file DarcyVT using parameter.txt in the main folder. Please see the section on how to modify parameter file to customize physical parameters and modify other features of the simulator.   
* Currently the simulator works only for rectangle shaped domains, more complicated domain boundaries could be dealt by labelling the boundary ids accordingly in inc/utilities. Other changes might also be required depending on the complexity of the domain boundary, please contact the author if that is the case.  
* Boundary and initial conditions along with source terms are taken from `inc/data.h` files. Function classes inside the data file can easily be modifed as required. See the section mixed boundary condition to see how to quickly assign constant Dirichlet or Neumann boundary conditions for each part of the boundary.
* If `mortar degree == 2`, then the mortar refinement is done only in every other refinement cycle, this is to maintain _H = Csqrt{h}_ and _Delta_T = Csqrt(Delta_t)_ mesh size relation between the subdomain and MORTAR mesh.  
* Mortar mesh configuration: 
  _mesh_pattern_sub_d0 3 2 5_ means the initial mesh for subdomain0 in refinement cycle 0 has 3 partitions in x ,2 partitions     in the y direction, and 5 partiions in the _time (_or _z)_ direction. Partition in the time direction is used to calculate the       Delta_t   required for backward euler time-stepping by using final_time = Delta_t*number of partitions in time irection.

Reading from parameter file
---------------------------
* Several physical parameters and other program features are loaded from the parameter.txt file, once the program is successfuly compiled.
* Try to keep the blank spaces and newlines in the file fixed, as changing it might cause to error in loading them. If you wish to change the structure of the parameter file or feed a different file, you could do so by modifying the `parameter_pull_in` subroutine inside the `inc/filecheck_utility.h` file.
* Whenever an input is assigned to a boolean variable (this is mentioned in the parameter file), use 0 for false and 1 for true. For example use is_manufact_soln(bool): 1, if you would you are using a manufactures solution, otherwise use 0.  
* By default, the simulator runs in the convergence test mode, where a manufactured solution is used to test the convergence rate of the algorithm. If you  would like to simulate a real life example, change is_manufact_soln to 0(false). Doing so will also disable the calculation and output of convergence rates.

## Mixed boundary condition
* There are options to use custom mixed bounday conditions.  
* If the boundary conditions are simple enough (constant Dirichlet or constant Neumann on each part of the boundary), the code enables easy implementation with modification of the parameter file, without the need of recompiling.  
* Use 'N' for Neumann part of the boundary and 'D' for the Dirichlet part followed by a space and the float type number, which will be used to create a constant boundary condition on the specified part. 
* For more complicated boundary condition, please use `inc/data.h` to make appropriate changes.

## Solution plots
* By default, space-time plots are saved in the _.vtu_ format (Paraview compatible) inside the `space-time-plots` folder. Load the _.pvtu_ to get the global domain view (all the sub-domains pasted together).  
* If you wish to look at the solution at each time step separately (for example to make an animation of the evolution of the solution), modify parameter file to reflect need_plot_at_each_time_step: 1. By default, it is set to zero and hence this feature is disabled.

Further improvements.
---------------------
1. The bottle neck in the simulation is where we do the projections across the interface from mortar to subdomain space-time mesh and vice-versa. This is mainly due to the inefficiency of the built in `FEFieldFunction()` from deal.ii which could be slow in finding the quadrature points around a general point for FE in a different mesh.  This could be sped up significantly by reimplementing the `project_boundary_value()` subroutine in projector.h where we could also save the inner product between basis functions from FE spaces coming from different meshes( in this case, the space-time mesh in subdomain and in the mortar) and use this in the remaining projections in the iteration. More information can be found here: [note1](https://user-images.githubusercontent.com/35903705/97474663-ba4b1f80-1922-11eb-9445-8e9e729c9a8b.jpg), 
[note2](https://user-images.githubusercontent.com/35903705/97474672-bcad7980-1922-11eb-8e7f-3ea41dab1c54.jpg).  
2. Optimization could be done in terms of storage if we save the `FEValue` and `FEFaceValues` objects during the time-stepping iterations. But currently, this is not needed because the calculations are not memory intense yet.   
3. Implementing pre-conditioner for the GMRES iterattions. Further theoretical analysis could accompany this imporovement.
