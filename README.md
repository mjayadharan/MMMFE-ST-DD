# MMMFE-VT-Darcy
Code development for numerical simulations to simulate time-dependent Darcy flow using Multiscale Mortar Mixed Finite Elements(MMMFE) and variable time stepping (VT) for each subdomain. 

Author.
----------------------
Manu Jayadharan, Department of Mathematics at University of Pittsburgh 9/17/2019: manu.jayadharan@gmail.com


Template: BiotDD with MMMFE, given at https://github.com/mjayadharan/BiotDD.git
--------------------------------------------------------------------

deal.ii 9.1 requirement
---------------------------------------
Requirements: Need deal.ii with mpi configured to compile and run the simulations. Latest version of dealii can be found at : https://www.dealii.org/download.html .

deal.ii installation instruction: Follow readme file to install with -DDEAL_II_WITH_MPI=ON flag to cmake. 


Compilation instruction.
-------------------------------------------
cmake -DDEAL_II_DIR=/path to dealii installation folder/ .
make release (for faster compilations)
make debug ( for more careful compilations with warnings)
mpirun -n 'j' ./DarcyVT (where j is the number of subdomains(processses))

Please contact the owner for further instructions.
