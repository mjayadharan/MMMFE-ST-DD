/*
 * filecheck_utility.h
 *
 *  Created on: Aug 10, 2020
 *      Author: Manu Jayadharan, University of Pittsburgh
 *
 *  Purpose: Contains subroutines to load and maintain parameters properly from external files.
 *  By default, parameters and desired features are loaded from parameter.txt which can be changed
 *  in the parameter_pull_in subroutine.
 */

#ifndef INC_FILECHECK_UTILITY_H_
#define INC_FILECHECK_UTILITY_H_

#include <deal.II/base/exceptions.h>

#include <string>

using namespace dealii;

//To check parameter entry compatibilities.
template<typename T>
	bool is_inside(std::vector<T> vect, T int_el )
{
			/*
			 * Manu_j
			 * Simple function to check whether an element of type T is in a vector of type T.
			 * Will be useful in setting mixed bc.
			 */
		bool int_el_found = std::find(vect.begin(), vect.end(), int_el) != vect.end();
		return int_el_found;
}

void parameter_pull_in (double &c_0, double &alpha, double &coe_a, int &space_degree, int &mortar_degree, int &num_refinement,
        			double &final_time, double &tolerence, int &max_iteration, bool &need_each_time_step_plot,
					std::vector<char> &bc_con, std::vector<double> &nm_bc_con_funcs, bool &is_manufact_solution,
					std::vector<std::vector<int>> &mesh_m3d, std::vector<std::vector<int>> &mesh_m3d_mortar,
					unsigned int n_processes, std::string file_name="parameter.txt")
{
	std::string dummy_string; //for getting rid of extra strings in the parameter file
	std::ifstream parameter_file (file_name);
	assert(parameter_file.is_open());


	for (int skip_line=0; skip_line<4; ++skip_line)
		std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>c_0;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>alpha;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>coe_a;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>space_degree;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>mortar_degree;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>num_refinement;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>final_time;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>tolerence;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>max_iteration;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>need_each_time_step_plot;
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>bc_con[0]>>nm_bc_con_funcs[0];
	parameter_file>>dummy_string>>bc_con[1]>>nm_bc_con_funcs[1];
	parameter_file>>dummy_string>>bc_con[2]>>nm_bc_con_funcs[2];
	parameter_file>>dummy_string>>bc_con[3]>>nm_bc_con_funcs[3];
	std::getline(parameter_file, dummy_string);

	parameter_file>>dummy_string>>is_manufact_solution;
	std::getline(parameter_file, dummy_string);

	for(unsigned int sub_id=0; sub_id<n_processes+1; sub_id++)
		parameter_file>>dummy_string>>mesh_m3d[sub_id][0]>>mesh_m3d[sub_id][1]>>mesh_m3d[sub_id][2];
	std::getline(parameter_file, dummy_string);

	for(unsigned int sub_id=0; sub_id<n_processes+1; sub_id++)
		parameter_file>>dummy_string>>mesh_m3d_mortar[sub_id][0]>>mesh_m3d_mortar[sub_id][1]>>mesh_m3d_mortar[sub_id][2];

	parameter_file.close();

	//Making sure that the data is compatible
    std::vector<char>possible_bc = {'D','N'};
    for (auto bc_type:bc_con){
	AssertThrow(is_inside<char>(possible_bc, bc_type), ExcMessage( "\n\nincompatible boundary condition read "
    			"from parameter file. Please provide either D or N dependeing on whether "
			"Dirichlet or Neumann boundary condition is desired\n"));
    }
}

#endif /* INC_FILECHECK_UTILITY_H_ */
