/* ---------------------------------------------------------------------*/
/* ---------------------------------------------------------------------
 This is the main file which compiles and run the time dependent Darcy flow using variable time stepping(VT) and
 multiscale mortar mixed finite elements(MMMFE). All the details are hidden in the source file src/darcy_vtdd.cc and other
 header files in inc/. After making and compiling the files, a custom parameter file("parameter.txt") can be used to load parameters
 and desired characteristics of the solver.
 Template: BiotDD with mortar functionality coauthored by Eldar K.
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan,  University of Pittsburgh: 2019-2020
 *
 */

// Utilities, data, etc.
#include "../inc/darcy_vtdd.h"
#include "../inc/filecheck_utility.h"

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
        int space_degree, mortar_degree, num_refinement, max_iteration;

        //declaring mesh refinement structure for space-time mortar
        std::vector<int> zeros_vector(3,0);
        std::vector<std::vector<int>> mesh_m3d;

        //boundary condition vector. 'D':Dirichlet, 'N': Neumann
        std::vector<char> bc_con(4,'D');
        std::vector<double> nm_bc_con_funcs(4,0.0);

        bool is_manufact_solution, need_each_time_step_plot;
        std::string dummy_string; //for getting rid of string in the parameter.dat

        /*
         * Block for pulling in parameters and other desired program features from a parameter file
         */
        {
            MPI_Comm mpi_communicator_1(MPI_COMM_WORLD);
            MPI_Status mpi_status_1;
            int mpi_send_bool(0), mpi__rec_bool(0);
        	const unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator_1);
        	const unsigned int n_processes = Utilities::MPI::n_mpi_processes(mpi_communicator_1);

        	mesh_m3d.resize(n_processes+1, zeros_vector);


        	if(this_mpi!=0)
        	{
        		  MPI_Recv(&mpi__rec_bool,  1, MPI_INT, this_mpi-1, this_mpi-1, mpi_communicator_1, &mpi_status_1);

        	}
        	// Pulling in the parameter and other requirements from input("parameter.txt") file
        	parameter_pull_in (c_0, alpha, space_degree, mortar_degree, num_refinement,
        			final_time, tolerence, max_iteration, need_each_time_step_plot,
					bc_con, nm_bc_con_funcs, is_manufact_solution, mesh_m3d, n_processes, "parameter.txt");

        	if(this_mpi!=n_processes-1)
        	        	{
        	        		  MPI_Send(&mpi_send_bool,  1, MPI_INT, this_mpi+1, this_mpi, mpi_communicator_1);

        	        	}

        }//end of reading parameter.dat file.




        BiotParameters bparam (1.0,1,final_time,c_0,alpha);
        //Instantiating the class
        DarcyVTProblem<2> problem_2d(space_degree, bparam, 1, mortar_degree, bc_con,
        		nm_bc_con_funcs, is_manufact_solution, need_each_time_step_plot);

        //Solving the problem
        problem_2d.run(num_refinement, mesh_m3d,
        		tolerence, max_iteration, mortar_degree+1);



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

