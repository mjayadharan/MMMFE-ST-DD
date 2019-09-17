/* ---------------------------------------------------------------------*/
/* ---------------------------------------------------------------------
 This is part of a program that  implements DD for 3 Different Schemes for Biot: Monolithic, Dranined SPlit and Fixed Stress. This file is specific to Example 1 in paper on DD for BIot schemes.
 *update: The code is modified to include nonmatching subdomain grid using mortar spaces and multiscale basis.
 * ---------------------------------------------------------------------
 *
 * Authors: Manu Jayadharan, Eldar Khattatov, University of Pittsburgh:2018- 2019
 */

// Utilities, data, etc.
#include "../inc/biot_mfedd.h"

// Main function is simple here
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace dd_biot;

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
        int num_cycle=4;
        int max_itr=500;
        double tolerence = 1.e-12;
        BiotParameters bparam (0.001,1,c0,alpha);

//        //BiotDD without mortar
//        MixedBiotProblemDD<2> drained_split(1, bparam,0,0,1);
//        MixedBiotProblemDD<2> fixed_stress(1,bparam,0,0,2);
//        MixedBiotProblemDD<2> monolithic(1,bparam,0,0,0);

//        drained_split.run (num_cycle, mesh_m2d, tolerence, max_itr);
//        fixed_stress.run(num_cycle, mesh_m2d, tolerence, max_itr);
//        monolithic.run (num_cycle, mesh_m2d, tolerence, max_itr);

     //BiotDD with mortar
        MixedBiotProblemDD<2> lin_mortar(1,bparam,1,1,0);
        MixedBiotProblemDD<2> quad_mortar(1,bparam,1,2,0);
        MixedBiotProblemDD<2> cubic_mortar(1,bparam,1,3,0);

        lin_mortar.run(num_cycle,mesh_m2d,tolerence,max_itr);
//        quad_mortar.run(num_cycle,mesh_m2d,tolerence,max_itr);
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
