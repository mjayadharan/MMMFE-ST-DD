/* ---------------------------------------------------------------------*/
/* ---------------------------------------------------------------------
 This is the main file which compiles and run the time dependent Darcy flow using variable time stepping(VT) and multiscale mortar mixed finite elements(MMMFE).
 Template: BiotDD with mortar functionality coauthored by Eldar K.
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan,  University of Pittsburgh: Fall 2019
 */

// Utilities, data, etc.
#include "../inc/darcy_vtdd.h"


int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace vt_darcy;

        MultithreadInfo::set_thread_limit(4);
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        // Mortar mesh parameters   (non-matching checkerboard)
        std::vector<std::vector<unsigned int>> mesh_m2d(3);
        mesh_m2d[0] = {2,2};
        mesh_m2d[1] = {3,3};
//        mesh_m2d[2] = {2,2};
//        mesh_m2d[3] = {2,2};
        mesh_m2d[2] = {1,1};
        double c0=1;
        double alpha=1;
        int num_cycle=2;
        double time_step_size = 0.05;
        int n_time_steps = 1; //this is just used to define the final_time in next line and to give a starting point for the actual num_time_steps, not anywhere else.
        double final_time = n_time_steps*time_step_size;
        int max_itr=500;
        double tolerence = 1.e-11;
        BiotParameters bparam (time_step_size,n_time_steps,final_time,c0,alpha);

        // Time space mortar mesh parameters   (non-matching checkerboard in space-time)
        std::vector<std::vector<unsigned int>> mesh_m3d(3);
        mesh_m3d[0] = {2,2,10*bparam.num_time_steps}; //number of cells in each direction: in the order of x,y,time. for domain 1.
        mesh_m3d[1] = {3,3,15*bparam.num_time_steps};
//        mesh_m3d[2] = {2,2,12*bparam.num_time_steps};
//        mesh_m3d[3] = {2,2,12*bparam.num_time_steps}; //number of cells in each direction: in the order of x,y,time. for domain 4.
        mesh_m3d[2] = {1,1,2*bparam.num_time_steps}; //number of cells in each direction: in the order of x,y,time. for mortar domain.


//        //DarcyDD without mortar
//        DarcyVTProblem<2> no_mortar(1,bparam,0,0);


//        no_mortar.run (num_cycle, mesh_m2d, mesh_m3d, tolerence, max_itr);

     //DarcyDD with mortar
        DarcyVTProblem<2> lin_mortar(0,bparam,1,1);
//        DarcyVTProblem<2> quad_mortar(1,bparam,1,2);
//        DarcyVTProblem<2> cubic_mortar(1,bparam,1,3);

        lin_mortar.run(num_cycle,mesh_m2d,mesh_m3d,tolerence,max_itr,4);
//        quad_mortar.run(num_cycle,mesh_m2d,mesh_m3d,tolerence,max_itr,5);
//        cubic_mortar.run(num_cycle,mesh_m2d,mesh_m3d,tolerence,max_itr,6);

    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }

    return 0;
}
