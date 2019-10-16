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
        std::vector<std::vector<unsigned int>> mesh_m2d(5);
        mesh_m2d[0] = {2,2};
        mesh_m2d[1] = {3,3};
        mesh_m2d[2] = {3,3};
        mesh_m2d[3] = {2,2};
        mesh_m2d[4] = {1,1};
        double c0=1;
        double alpha=1;
        int num_cycle=2;
        int max_itr=500;
        double tolerence = 1.e-11;
        BiotParameters bparam (0.001,2,c0,alpha);

//        //DarcyDD without mortar
//        DarcyVTProblem<2> no_mortar(1,bparam,0,0);


//        no_mortar.run (num_cycle, mesh_m2d, tolerence, max_itr);

     //DarcyDD with mortar
        DarcyVTProblem<2> lin_mortar(0,bparam,1,1);
        DarcyVTProblem<2> quad_mortar(1,bparam,1,2);
//        DarcyVTProblem<2> cubic_mortar(1,bparam,1,3);

//        lin_mortar.run(num_cycle,mesh_m2d,tolerence,max_itr);
        quad_mortar.run(num_cycle,mesh_m2d,tolerence,max_itr);
//        cubic_mortar.run(num_cycle,mesh_m2d,tolerence,max_itr);

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
