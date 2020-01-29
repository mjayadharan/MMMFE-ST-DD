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
#include <fstream>
#include <string>
#include <cassert>


int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace vt_darcy;

        MultithreadInfo::set_thread_limit(4);
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        //declaring parameter variables .
        double c_0, alpha, final_time, tolerence;

        int num_refinement, max_iteration;

        //declaring mesh refinement structure for space-time mortar
        std::vector<int> zeros_vector(3,0);
        std::vector<std::vector<int>> mesh_m3d(3,zeros_vector);

        std::string dummy_string; //for getting rid of string in the parameter.dat
        {//Reading parameters from parameter.dat file
            MPI_Comm mpi_communicator_1(MPI_COMM_WORLD);
            MPI_Status mpi_status_1;
            int mpi_send_bool(0), mpi__rec_bool(0);
        	const unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator_1);
        	const unsigned int n_processes = Utilities::MPI::n_mpi_processes(mpi_communicator_1);
        	if(this_mpi!=0)
        	{
        		  MPI_Recv(&mpi__rec_bool,  1, MPI_INT, this_mpi-1, this_mpi-1, mpi_communicator_1, &mpi_status_1);

        	}

				std::ifstream parameter_file ("parameter.txt");
				assert(parameter_file.is_open());
				parameter_file>>dummy_string>>c_0;
				parameter_file>>dummy_string>>alpha;
				parameter_file>>dummy_string>>num_refinement;
				parameter_file>>dummy_string>>final_time;
				parameter_file>>dummy_string>>tolerence;
				parameter_file>>dummy_string>>max_iteration;
				parameter_file>>dummy_string>>mesh_m3d[0][0]>>mesh_m3d[0][1]>>mesh_m3d[0][2];
				parameter_file>>dummy_string>>mesh_m3d[1][0]>>mesh_m3d[1][1]>>mesh_m3d[1][2];
				parameter_file>>dummy_string>>mesh_m3d[2][0]>>mesh_m3d[2][1]>>mesh_m3d[2][2];
//  	      	parameter_file>>dummy_string>>mesh_m3d[3][0]>>mesh_m3d[3][1]>>mesh_m3d[3][2];
//	        	parameter_file>>dummy_string>>mesh_m3d[4][0]>>mesh_m3d[4][1]>>mesh_m3d[4][2];

        	parameter_file.close();

        	if(this_mpi!=n_processes-1)
        	        	{
        	        		  MPI_Send(&mpi_send_bool,  1, MPI_INT, this_mpi+1, this_mpi, mpi_communicator_1);

        	        	}

        }//end of reading parameter.dat file.

//        // Mortar mesh parameters   (non-matching checkerboard)
//        std::vector<std::vector<unsigned int>> mesh_m2d(3);
//        mesh_m2d[0] = {2,2};
//        mesh_m2d[1] = {3,3};
////        mesh_m2d[2] = {2,2};
////        mesh_m2d[3] = {2,2};
//        mesh_m2d[2] = {1,1};
//        double c0=1;
//        double alpha=1;
//        int num_refinement=2;
//        double time_step_size = 0.05;
//        int n_time_steps = 1; //this is just used to define the final_time in next line and to give a starting point for the actual num_time_steps, not anywhere else.
//        double final_time = n_time_steps*time_step_size;
//        int max_itr=500;
//        double tolerence = 1.e-11;

        BiotParameters bparam (1.0,1,final_time,c_0,alpha);

//        // Time space mortar mesh parameters   (non-matching checkerboard in space-time)
//        std::vector<std::vector<unsigned int>> mesh_m3d(3);
//        mesh_m3d[0] = {2,2,10*bparam.num_time_steps}; //number of cells in each direction: in the order of x,y,time. for domain 1.
//        mesh_m3d[1] = {3,3,15*bparam.num_time_steps};
////        mesh_m3d[2] = {2,2,12*bparam.num_time_steps};
////        mesh_m3d[3] = {2,2,12*bparam.num_time_steps}; //number of cells in each direction: in the order of x,y,time. for domain 4.
//        mesh_m3d[2] = {1,1,2*bparam.num_time_steps}; //number of cells in each direction: in the order of x,y,time. for mortar domain.


//        //DarcyDD without mortar
//        DarcyVTProblem<2> no_mortar(1,bparam,0,0);


//        no_mortar.run (num_refinement, mesh_m2d, mesh_m3d, tolerence, max_itr);

     //DarcyDD with mortar
        DarcyVTProblem<2> constant_mortar(0,bparam,1,0);
//        DarcyVTProblem<2> lin_mortar(0,bparam,1,1);
//        DarcyVTProblem<2> quad_mortar(1,bparam,1,2);
//        DarcyVTProblem<2> cubic_mortar(1,bparam,1,3);

        constant_mortar.run(num_refinement,mesh_m3d,tolerence,max_iteration,4);
//        lin_mortar.run(num_refinement,mesh_m3d,tolerence,max_iteration,4);
//        quad_mortar.run(num_refinement,mesh_m2d,mesh_m3d,tolerence,max_itr,5);
//        cubic_mortar.run(num_refinement,mesh_m2d,mesh_m3d,tolerence,max_itr,6);

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
