# MMMFE-ST-DD
Fluid flow simulator using multiscale space-time domains. 

Code developed to simulate time-dependent diffusion problem using Multiscale Mortar Mixed Finite Elements(MMMFE). Model can be easily adapted to simulate other fluid flow models based on linear PDEs. The novelty of the simulator lies in using multiple subdomains with variable time steps and mesh size for each subdomain. This give rise to a space-time DD technique allowing non-matching grids for sub-domains in both space and time dimensions. Sub-domain solves are done in parallel across different processors using MPI. Computed solutions are outputted and visualized on a global space-time grid in the .vtk and .vtu formats. Details of the spaces used and rough algorithm can be found in report.pdf and algorithm.pdf respectively. Theoretical results guaranteeing convergence and stability of the problem along with a priori error estimates are proved and being published.
![github_space_time_dd](https://user-images.githubusercontent.com/35903705/86996707-51287c00-c17a-11ea-8d9c-584aa2cfc47b.png)


The simulator is written using deal.ii FE package which is based on C++.

## Author
-----------
Manu Jayadharan, Department of Mathematics at University of Pittsburgh 9/17/2019

email: [manu.jayadharan@gmail.com](mailto:manu.jayadharan@gmail.com)  
[reserachgate link](https://www.researchgate.net/profile/Manu_Jayadharan)  
[linkedin profile](https://www.linkedin.com/in/manu-jayadharan/)


## Collaborators
------------------
- Michel Kern, INRIA(French Institute for Research in Computer Science and Automation).  
Maintaining branch: *michel_Andra_testcase* .   
[INRIA webpage](https://who.rocq.inria.fr/Michel.Kern/)


## deal.ii 9.1 requirement (latest at the time)
---------------------------------------
Need deal.ii configured with mpi  to compile and run the simulations. Latest version of dealii can be found at : [https://www.dealii.org/download.html](https://www.dealii.org/download.html)

**deal.ii installation instruction:** Follow readme file to install latest version of deal.ii with `-DDEAL_II_WITH_MPI=ON` flag to cmake. 


## Compilation instructions.
-------------------------------------------
`cmake -DDEAL_II_DIR=/path to dealii installation folder/ .` from the main directory

`make release` for faster compilations

`make debug` for more careful compilations with warnings

`mpirun -n 'j' DarcyVT` where j is the number of subdomains(processses)

**Please contact the author for further instructions.**

## Quick start guide for the simulator.
-------------------------------------
* Most of the parameters including number of refinements, mortar_degree, max_number of gmres iterations, final_time, subdomain mesh size
ratio etc are fed to the executable file DarcyVT using parameter.txt in the main folder. This file can simply be modified
without recompiling the program.  
* Currently parameter.txt is designed to work only for 4 subdomain DD, but could easily be modified to work with a different number of subdomains. Look at darcy_main.cc to see how the parameters are fed to the program and make necessary changes.  
* If mortar degree==2, then the mortar refinement is done only in every other refinement cycle, this is to maintain H=Csqrt(h) and Delta_T =Csqrt(Delta_t) mesh size relation between the subdomain and MORTAR mesh.  
* Mortar mesh configuration: 
  mesh_pattern_sub_d0 3 2 5 means the initial mesh for subdomain0 in refinement cycle 0 has 3 partitions in x ,2 partitions     in the y direction, and 5 partiions in the time(z) direction. Partition in the time direction is used to calculate the       Delta_t   required for backward euler time-stepping by using final_time = Delta_t * number of partitions in time(z)           direction.

Further improvements.
---------------------
1. The bottle neck in the simulation is where we do the projections across the interface from mortar to subdomain space-time mesh and vice-versa. This is mainly due to the inefficiency of the built in FEFieldFunction() from deal.ii which could be slow in finding the quadrature points around a general point for FE in a different mesh.  This could be sped up significantly by reimplementing the project_boundary_value subroutine in projector.h where we could also save the inner product between basis functions from FE spaces coming from different meshes( in this case, the space-time mesh in subdomain and in the mortar) and use this in the remaining projections in the iteration.  
2. Optimization could be done in terms of storage if we save the FEValue and FEFaceValues objects during the time-stepping iterations. But currently, this is not needed because the calculations are not memory intense yet.   
3. Implementing pre-conditioner for the GMRES iterattions. Further theoretical analysis could accompany this imporovement.
