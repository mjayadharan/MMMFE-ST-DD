/* ---------------------------------------------------------------------*/
/* ---------------------------------------------------------------------
 This is the main file which compiles and run the time dependent Darcy flow using variable time stepping(VT) and multiscale mortar mixed finite elements(MMMFE).
 Template: BiotDD with mortar functionality coauthored by Eldar K.
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan,  University of Pittsburgh: 2019-2020
 */

// Utilities, data, etc.
#include "../inc/darcy_vtdd.h"
#include <fstream>
#include <string>
#include <cassert>

//To check parameter entry compatibilities.
template<typename T>
	bool is_inside(std::vector<T> vect, T int_el ){
			/*
			 * Manu_j
			 * Simple function to check whether an element of type T is in a vector of type T.
			 * Will be useful in setting mixed bc.
			 */
		bool int_el_found = std::find(vect.begin(), vect.end(), int_el) != vect.end();
		return int_el_found;
	}

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
//        std::vector<std::vector<int>> mesh_m3d(5,zeros_vector);

        //boundary condition vector. 'D':Dirichlet, 'N': Neumann
        std::vector<char> bc_con(4,'D');
        std::vector<char>possible_bc = {'D','N'};

        std::string dummy_string; //for getting rid of string in the parameter.dat
        {//Reading parameters from parameter.dat file
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

				std::ifstream parameter_file ("parameter.txt");
				assert(parameter_file.is_open());
				parameter_file>>dummy_string>>c_0;
				parameter_file>>dummy_string>>alpha;
				parameter_file>>dummy_string>>space_degree;
				parameter_file>>dummy_string>>mortar_degree;
				parameter_file>>dummy_string>>num_refinement;
				parameter_file>>dummy_string>>final_time;
				parameter_file>>dummy_string>>tolerence;
				parameter_file>>dummy_string>>max_iteration;
				parameter_file>>dummy_string>>bc_con[0];
				parameter_file>>dummy_string>>bc_con[1];
				parameter_file>>dummy_string>>bc_con[2];
				parameter_file>>dummy_string>>bc_con[3];
				for(unsigned int sub_id=0; sub_id<n_processes+1; sub_id++)
					parameter_file>>dummy_string>>mesh_m3d[sub_id][0]>>mesh_m3d[sub_id][1]>>mesh_m3d[sub_id][2];

				parameter_file.close();

        	if(this_mpi!=n_processes-1)
        	        	{
        	        		  MPI_Send(&mpi_send_bool,  1, MPI_INT, this_mpi+1, this_mpi, mpi_communicator_1);

        	        	}

        }//end of reading parameter.dat file.

        for (auto bc_type:bc_con){
        	assert(is_inside<char>(possible_bc, bc_type) && "\n\nincompatible boundary condition read "
        			"from parameter file. Please provide either D or N dependeing on whether "
        			"Dirichlet or Neumann boundary condition is desired\n");
        }

        BiotParameters bparam (1.0,1,final_time,c_0,alpha);

        //Solving the problem.
        DarcyVTProblem<2> problem_2d(space_degree,bparam,1,mortar_degree, bc_con);
        problem_2d.run(num_refinement,mesh_m3d,tolerence,max_iteration,mortar_degree+1);



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
