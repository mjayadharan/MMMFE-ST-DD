/* ---------------------------------------------------------------------*/
/* ---------------------------------------------------------------------
 This is part of a program that  implements DD for 3 Different Schemes for Biot: Monolithic, Dranined SPlit and Fixed Stress. This file is specific to Example 1 in paper on DD for BIot schemes.
 *update: The code is modified to include nonmatching subdomain grid using mortar spaces and multiscale basis.
 * ---------------------------------------------------------------------
 *
 * Authors: Manu Jayadharan, Eldar Khattatov, University of Pittsburgh:2018- 2019
 */

// Internals,.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
// Extra for MPI and mortars
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/timer.h>
// C++
#include <fstream>
#include <iostream>
#include <random>
// Utilities, data, etc.
#include "../inc/biot_mfedd.h"
#include "../inc/utilities.h"
#include "../inc/data.h"


namespace dd_biot
{
    using namespace dealii;

    // MixedElasticityDD class constructor
    template <int dim>
    MixedBiotProblemDD<dim>::MixedBiotProblemDD (const unsigned int degree,
                                                 const BiotParameters &bprm,
                                                 const unsigned int mortar_flag,
                                                 const unsigned int mortar_degree,
												 unsigned int split_flag)
            :
            mpi_communicator (MPI_COMM_WORLD),
            P_coarse2fine (false),
            P_fine2coarse (false),
            n_domains(dim,0),
            prm(bprm),
            degree (degree),
            mortar_degree(mortar_degree),
            mortar_flag (mortar_flag),
            cg_iteration(0),
			gmres_iteration(0),
            max_cg_iteration(0),
			max_cg_iteration_darcy(0),
            qdegree(11),
			split_flag(split_flag),
            fe (FE_BDM<dim>(degree), dim,
                FE_DGQ<dim>(degree-1), dim,
                FE_DGQ<dim>(degree-1), 0.5*dim*(dim-1),
                FE_BDM<dim>(degree), 1,
                FE_DGQ<dim>(degree-1), 1),
			fe_elast (FE_BDM<dim>(degree), dim,
				FE_DGQ<dim>(degree-1), dim,
				FE_DGQ<dim>(degree-1), 0.5*dim*(dim-1)),
			fe_darcy (FE_BDM<dim>(degree), 1,
				FE_DGQ<dim>(degree-1), 1),
            dof_handler (triangulation),
			dof_handler_elast(triangulation),
			dof_handler_darcy(triangulation),
            fe_mortar (FE_RaviartThomas<dim>(mortar_degree), dim,
                       FE_Nothing<dim>(), dim,
                       FE_Nothing<dim>(), 0.5*dim*(dim-1),
                       FE_RaviartThomas<dim>(mortar_degree), 1,
                       FE_Nothing<dim>(), 1),
            dof_handler_mortar (triangulation_mortar),
            pcout (std::cout,
                   (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
            computing_timer (mpi_communicator,
                             pcout,
                             TimerOutput::summary,
                             TimerOutput::wall_times),
			grid_diameter(0)
    {}


    template <int dim>
    void MixedBiotProblemDD<dim>::set_current_errors_to_zero()
    {
      std::fill(err.l2_l2_errors.begin(), err.l2_l2_errors.end(), 0.0);
      std::fill(err.l2_l2_norms.begin(), err.l2_l2_norms.end(), 0.0);

      std::fill(err.linf_l2_errors.begin(), err.linf_l2_errors.end(), 0.0);
      std::fill(err.linf_l2_norms.begin(), err.linf_l2_norms.end(), 0.0);

      std::fill(err.velocity_stress_l2_div_errors.begin(), err.velocity_stress_l2_div_errors.end(), 0.0);
      std::fill(err.velocity_stress_l2_div_norms.begin(), err.velocity_stress_l2_div_norms.end(), 0.0);

      std::fill(err.velocity_stress_linf_div_errors.begin(), err.velocity_stress_linf_div_errors.end(), 0.0);
      std::fill(err.velocity_stress_linf_div_norms.begin(), err.velocity_stress_linf_div_norms.end(), 0.0);

      std::fill(err.pressure_disp_l2_midcell_errors.begin(), err.pressure_disp_l2_midcell_errors.end(), 0.0);
      std::fill(err.pressure_disp_l2_midcell_norms.begin(), err.pressure_disp_l2_midcell_norms.end(), 0.0);

      std::fill(err.pressure_disp_linf_midcell_errors.begin(), err.pressure_disp_linf_midcell_errors.end(), 0.0);
      std::fill(err.pressure_disp_linf_midcell_norms.begin(), err.pressure_disp_linf_midcell_norms.end(), 0.0);
    }

    // MixedBiotProblemDD::make_grid_and_dofs
    template <int dim>
    void MixedBiotProblemDD<dim>::make_grid_and_dofs ()
    {	    	pcout<<"\n split_flag value is "<<split_flag<<"\n";

        TimerOutput::Scope t(computing_timer, "Make grid and DoFs");
        if(split_flag==0)
        	system_matrix.clear();
        else if(split_flag!=0)
        {
        	system_matrix_elast.clear();
        	system_matrix_darcy.clear();
        }

        //double lower_left, upper_right;
        //const unsigned int n_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
        const unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator);

        // Find neighbors
        neighbors.resize(GeometryInfo<dim>::faces_per_cell, 0);
        find_neighbors(dim, this_mpi, n_domains, neighbors);

        // Make interface data structures
        faces_on_interface.resize(GeometryInfo<dim>::faces_per_cell,0);
        faces_on_interface_mortar.resize(GeometryInfo<dim>::faces_per_cell,0);

        // Label interface faces and count how many of them there are per interface
        mark_interface_faces(triangulation, neighbors, p1, p2, faces_on_interface);
        if (mortar_flag)
            mark_interface_faces(triangulation_mortar, neighbors, p1, p2, faces_on_interface_mortar);

        dof_handler.distribute_dofs(fe);
        DoFRenumbering::component_wise (dof_handler);


        if(split_flag!=0){
        	dof_handler_elast.distribute_dofs(fe_elast);
        	DoFRenumbering::component_wise (dof_handler_elast);
        	dof_handler_darcy.distribute_dofs(fe_darcy);
        	DoFRenumbering::component_wise (dof_handler_darcy);
        }


        if (mortar_flag)
        {
            dof_handler_mortar.distribute_dofs(fe_mortar);
            DoFRenumbering::component_wise (dof_handler_mortar);
        }

        std::vector<types::global_dof_index> dofs_per_component (dim*dim + dim + 0.5*dim*(dim-1) + dim + 1);
        DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
        unsigned int n_s=0, n_u=0, n_g=0, n_z = 0, n_p = 0;

        for (unsigned int i=0; i<dim; ++i)
        {
            n_s += dofs_per_component[i*dim];
            n_u += dofs_per_component[dim*dim + i];

            // Rotation is scalar in 2d and vector in 3d, so this:
            if (dim == 2)
                n_g = dofs_per_component[dim*dim + dim];
            else if (dim == 3)
                n_g += dofs_per_component[dim*dim + dim + i];
        }

        n_z = dofs_per_component[dim*dim + dim + 0.5*dim*(dim-1)];
        n_p = dofs_per_component[dim*dim + dim + 0.5*dim*(dim-1) + dim];

        n_stress = n_s;
        n_disp= n_u;
        n_rot=n_g;
        n_flux = n_z;
        n_pressure= n_p;
        n_Elast= n_s+n_u+n_g;

        if(split_flag==0) //monolithic coupled scheme
        {

			BlockDynamicSparsityPattern dsp(5, 5);
			dsp.block(0, 0).reinit (n_s, n_s);
			dsp.block(0, 1).reinit (n_s, n_u);
			dsp.block(0, 2).reinit (n_s, n_g);
			dsp.block(0, 3).reinit (n_s, n_z);
			dsp.block(0, 4).reinit (n_s, n_p);

			dsp.block(1, 0).reinit (n_u, n_s);
			dsp.block(1, 1).reinit (n_u, n_u);
			dsp.block(1, 2).reinit (n_u, n_g);
			dsp.block(1, 3).reinit (n_u, n_z);
			dsp.block(1, 4).reinit (n_u, n_p);

			dsp.block(2, 0).reinit (n_g, n_s);
			dsp.block(2, 1).reinit (n_g, n_u);
			dsp.block(2, 2).reinit (n_g, n_g);
			dsp.block(2, 3).reinit (n_g, n_z);
			dsp.block(2, 4).reinit (n_g, n_p);

			dsp.block(3, 0).reinit (n_z, n_s);
			dsp.block(3, 1).reinit (n_z, n_u);
			dsp.block(3, 2).reinit (n_z, n_g);
			dsp.block(3, 3).reinit (n_z, n_z);
			dsp.block(3, 4).reinit (n_z, n_p);

			dsp.block(4, 0).reinit (n_p, n_s);
			dsp.block(4, 1).reinit (n_p, n_u);
			dsp.block(4, 2).reinit (n_p, n_g);
			dsp.block(4, 3).reinit (n_p, n_z);
			dsp.block(4, 4).reinit (n_p, n_p);

			dsp.collect_sizes ();
			DoFTools::make_sparsity_pattern (dof_handler, dsp);

			// Initialize system matrix
			sparsity_pattern.copy_from(dsp);
			system_matrix.reinit (sparsity_pattern);

			// Reinit solution and RHS vectors
			solution_bar.reinit (5);
			solution_bar.block(0).reinit (n_s);
			solution_bar.block(1).reinit (n_u);
			solution_bar.block(2).reinit (n_g);
			solution_bar.block(3).reinit (n_z);
			solution_bar.block(4).reinit (n_p);
			solution_bar.collect_sizes ();
			solution_bar = 0;

			// Reinit solution and RHS vectors
			solution_star.reinit (5);
			solution_star.block(0).reinit (n_s);
			solution_star.block(1).reinit (n_u);
			solution_star.block(2).reinit (n_g);
			solution_star.block(3).reinit (n_z);
			solution_star.block(4).reinit (n_p);
			solution_star.collect_sizes ();
			solution_star = 0;

			system_rhs_bar.reinit (5);
			system_rhs_bar.block(0).reinit (n_s);
			system_rhs_bar.block(1).reinit (n_u);
			system_rhs_bar.block(2).reinit (n_g);
			system_rhs_bar.block(3).reinit (n_z);
			system_rhs_bar.block(4).reinit (n_p);
			system_rhs_bar.collect_sizes ();
			system_rhs_bar = 0;

			system_rhs_star.reinit (5);
			system_rhs_star.block(0).reinit (n_s);
			system_rhs_star.block(1).reinit (n_u);
			system_rhs_star.block(2).reinit (n_g);
			system_rhs_star.block(3).reinit (n_z);
			system_rhs_star.block(4).reinit (n_p);
			system_rhs_star.collect_sizes ();
			system_rhs_star = 0;

			//adding vectors required for storing mortar solutions.
			if (mortar_flag)
			        {
			            std::vector<types::global_dof_index> dofs_per_component_mortar (dim*dim + dim + 0.5*dim*(dim-1) + dim + 1);
			            DoFTools::count_dofs_per_component (dof_handler_mortar, dofs_per_component_mortar);
			            unsigned int n_s_mortar=0, n_u_mortar=0, n_g_mortar=0, n_z_mortar=0, n_p_mortar=0;

			            for (unsigned int i=0; i<dim; ++i)
			            {
			                n_s_mortar += dofs_per_component_mortar[i*dim];
			                n_u_mortar += dofs_per_component_mortar[dim*dim + i];

			                // Rotation is scalar in 2d and vector in 3d, so this:
			                if (dim == 2)
			                    n_g_mortar = dofs_per_component_mortar[dim*dim + dim];
			                else if (dim == 3)
			                    n_g_mortar += dofs_per_component_mortar[dim*dim + dim + i];
			            }

			            n_z_mortar = dofs_per_component_mortar[dim*dim + dim + 0.5*dim*(dim-1)];
			            n_p_mortar = dofs_per_component_mortar[dim*dim + dim + 0.5*dim*(dim-1) + dim];

			            n_stress = n_s_mortar;
			            n_disp = n_u_mortar;
			            n_rot = n_g_mortar;
			            n_flux = n_z_mortar;
			            n_pressure = n_p_mortar;

			            solution_bar_mortar.reinit(5);
			            solution_bar_mortar.block(0).reinit (n_s_mortar);
			            solution_bar_mortar.block(1).reinit (n_u_mortar);
			            solution_bar_mortar.block(2).reinit (n_g_mortar);
			            solution_bar_mortar.block(3).reinit (n_z_mortar);
			            solution_bar_mortar.block(4).reinit (n_p_mortar);
			            solution_bar_mortar.collect_sizes ();

			            solution_star_mortar.reinit(5);
			            solution_star_mortar.block(0).reinit (n_s_mortar);
			            solution_star_mortar.block(1).reinit (n_u_mortar);
			            solution_star_mortar.block(2).reinit (n_g_mortar);
			            solution_star_mortar.block(3).reinit (n_z_mortar);
			            solution_star_mortar.block(4).reinit (n_p_mortar);
			            solution_star_mortar.collect_sizes ();
			        }
        }
        else if(split_flag!=0){
        	{ //Elasticity part
				BlockDynamicSparsityPattern dsp(3, 3);
				dsp.block(0, 0).reinit(n_s, n_s);
				dsp.block(0, 1).reinit(n_s, n_u);
				dsp.block(0, 2).reinit(n_s, n_g);
				dsp.block(1, 0).reinit(n_u, n_s);
				dsp.block(1, 1).reinit(n_u, n_u);
				dsp.block(1, 2).reinit(n_u, n_g);
				dsp.block(2, 0).reinit(n_g, n_s);
				dsp.block(2, 1).reinit(n_g, n_u);
				dsp.block(2, 2).reinit(n_g, n_g);
				dsp.collect_sizes();
				DoFTools::make_sparsity_pattern(dof_handler_elast, dsp);

				// Initialize system matrix
				sparsity_pattern_elast.copy_from(dsp);
				system_matrix_elast.reinit(sparsity_pattern_elast);

				// Reinit solution and RHS vectors
				solution_bar_elast.reinit(3);
				solution_bar_elast.block(0).reinit(n_s);
				solution_bar_elast.block(1).reinit(n_u);
				solution_bar_elast.block(2).reinit(n_g);
				solution_bar_elast.collect_sizes();
				solution_bar_elast = 0;

				// Reinit solution and RHS vectors
				solution_star_elast.reinit(3);
				solution_star_elast.block(0).reinit(n_s);
				solution_star_elast.block(1).reinit(n_u);
				solution_star_elast.block(2).reinit(n_g);
				solution_star_elast.collect_sizes();
				solution_star_elast = 0;

				system_rhs_bar_elast.reinit(3);
				system_rhs_bar_elast.block(0).reinit(n_s);
				system_rhs_bar_elast.block(1).reinit(n_u);
				system_rhs_bar_elast.block(2).reinit(n_g);
				system_rhs_bar_elast.collect_sizes();
				system_rhs_bar_elast = 0;

				system_rhs_star_elast.reinit(3);
				system_rhs_star_elast.block(0).reinit(n_s);
				system_rhs_star_elast.block(1).reinit(n_u);
				system_rhs_star_elast.block(2).reinit(n_g);
				system_rhs_star_elast.collect_sizes();
				system_rhs_star_elast = 0;
        	}//end of Elasticity part
//        	pcout<<"\n reached here"<<"\n";
        	{//Darcy part
        	    BlockDynamicSparsityPattern dsp(2, 2);
        	    dsp.block(0, 0).reinit (n_z, n_z);
        	    dsp.block(1, 0).reinit (n_p, n_z);
        	    dsp.block(0, 1).reinit (n_z, n_p);
        	    dsp.block(1, 1).reinit (n_p, n_p);
        	    dsp.collect_sizes ();
        	    DoFTools::make_sparsity_pattern (dof_handler_darcy, dsp);

        	    // Initialize system matrix
        	    sparsity_pattern_darcy.copy_from(dsp);
        	    system_matrix_darcy.reinit (sparsity_pattern_darcy);

        	    // Reinit solution and RHS vectors
        	    solution_bar_darcy.reinit (2);
        	    solution_bar_darcy.block(0).reinit (n_z);
        	    solution_bar_darcy.block(1).reinit (n_p);
        	    solution_bar_darcy.collect_sizes ();
        	    solution_bar_darcy=0;

        	    solution_star_darcy.reinit (2);
        	    solution_star_darcy.block(0).reinit (n_z);
        	    solution_star_darcy.block(1).reinit (n_p);
        	    solution_star_darcy.collect_sizes ();
        	    solution_star_darcy=0;

        	    system_rhs_bar_darcy.reinit (2);
        	    system_rhs_bar_darcy.block(0).reinit (n_z);
        	    system_rhs_bar_darcy.block(1).reinit (n_p);
        	    system_rhs_bar_darcy.collect_sizes ();
        	    system_rhs_bar_darcy=0;

        	    system_rhs_star_darcy.reinit (2);
        	    system_rhs_star_darcy.block(0).reinit (n_z);
        	    system_rhs_star_darcy.block(1).reinit (n_p);
        	    system_rhs_star_darcy.collect_sizes ();
        		system_rhs_star_darcy=0;

        	}//end of Darcy Part

        }


        solution.reinit (5);
        solution.block(0).reinit (n_s);
        solution.block(1).reinit (n_u);
        solution.block(2).reinit (n_g);
        solution.block(3).reinit (n_z);
        solution.block(4).reinit (n_p);
        solution.collect_sizes ();
        solution = 0;
        old_solution.reinit(solution);
        if(split_flag!=0){
        	intermediate_solution.reinit(solution);
        	if(split_flag==2){
        		older_solution.reinit(solution);
//        		intermediate_solution_old(solution);
        	}

        }



        pcout << "N stress dofs: " << n_stress << std::endl;
        pcout << "N flux dofs: " << n_flux << std::endl;
    }


    // MixedBiotProblemDD - assemble_system
    template <int dim>
    void MixedBiotProblemDD<dim>::assemble_system ()
    {
        TimerOutput::Scope t(computing_timer, "Assemble system");
        system_matrix = 0;
        //system_rhs_bar = 0;

        QGauss<dim>   quadrature_formula(degree+3);
        //QGauss<dim-1> face_quadrature_formula(qdegree);

        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    | update_gradients |
                                 update_quadrature_points  | update_JxW_values);

        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        //const unsigned int   n_face_q_points = face_quadrature_formula.size();

        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        const LameParameters<dim> lame_function;
        const KInverse<dim> k_inverse;

        std::vector<Vector<double>> lame_parameters_values(n_q_points, Vector<double>(2));
        std::vector<Tensor<2,dim>>  k_inverse_values(n_q_points);

        // Rotation variable is either a scalar(2d) or a vector(3d)
        const unsigned int rotation_dim = 0.5*dim*(dim-1);
        // Stress DoFs vectors
        std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
        std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());
        // Displacement DoFs
        const FEValuesExtractors::Vector displacement (dim*dim);
        // Velocity and Pressure DoFs
        const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));
        const FEValuesExtractors::Scalar pressure (dim*dim + dim + 0.5*dim*(dim-1) + dim);

        for (unsigned int i=0; i<dim; ++i)
        {
            const FEValuesExtractors::Vector tmp_stress(i*dim);
            stresses[i].first_vector_component = tmp_stress.first_vector_component;
            if (dim == 2 && i == 0)
            {
                const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim);
                rotations[i].component = tmp_rotation.component;
            } else if (dim == 3) {
                const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + i);
                rotations[i].component = tmp_rotation.component;
            }
        }

        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            local_matrix = 0;

            lame_function.vector_value_list (fe_values.get_quadrature_points(), lame_parameters_values);
            k_inverse.value_list (fe_values.get_quadrature_points(), k_inverse_values);

            // Velocity and pressure
            std::vector<Tensor<1,dim>>                phi_u(dofs_per_cell);
            std::vector <double>                      div_phi_u(dofs_per_cell);
            std::vector <double>                      phi_p(dofs_per_cell);

            // Stress, displacement and rotation
            std::vector<std::vector<Tensor<1,dim>>> phi_s(dofs_per_cell, std::vector<Tensor<1,dim> > (dim));
            std::vector<Tensor<1,dim>> div_phi_s(dofs_per_cell);
            std::vector<Tensor<1,dim>> phi_d(dofs_per_cell);
            std::vector<Tensor<1,rotation_dim>> phi_r(dofs_per_cell);

            Tensor<2,dim> sigma, asigma, apId;
            Tensor<1,rotation_dim> asym_i, asym_j;

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                  // Evaluate test functions
                  phi_u[k] = fe_values[velocity].value (k, q);
                  phi_p[k] = fe_values[pressure].value (k, q);

//                  for (auto el : phi_p)
//                    std::cout << "Pressure: " << el << " ";
//                  std::cout << std::endl;

                  div_phi_u[k] = fe_values[velocity].divergence (k, q);

                  for (unsigned int s_i=0; s_i<dim; ++s_i)
                  {
                    phi_s[k][s_i] = fe_values[stresses[s_i]].value (k, q);
                    div_phi_s[k][s_i] = fe_values[stresses[s_i]].divergence (k, q);
                  }
                  phi_d[k] = fe_values[displacement].value (k, q);

                  for (unsigned int r_i=0; r_i<rotation_dim; ++r_i)
                    phi_r[k][r_i] = fe_values[rotations[r_i]].value (k, q);
                }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    const double mu = lame_parameters_values[q][1];
                    const double lambda = lame_parameters_values[q][0];

                    compliance_tensor(phi_s[i], mu, lambda, asigma);
                    compliance_tensor_pressure(phi_p[i], mu, lambda, apId);
                    make_asymmetry_tensor(phi_s[i], asym_i);



                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        make_tensor(phi_s[j], sigma);
                        make_asymmetry_tensor(phi_s[j], asym_j);

                        local_matrix(i, j) += (phi_u[i] * k_inverse_values[q] * phi_u[j] - phi_p[j] * div_phi_u[i]                                     // Darcy law
                                               + prm.time_step*div_phi_u[j] * phi_p[i] + prm.c_0*phi_p[i]*phi_p[j] + prm.alpha*trace(asigma)*phi_p[j]  // Momentum
                                               + prm.alpha * prm.alpha * trace(apId)*phi_p[j]
                                               + scalar_product(asigma, sigma) + prm.alpha*scalar_product(apId, sigma)                                 // Mixed elasticity eq-ns
                                               + scalar_product(phi_d[i], div_phi_s[j])  + scalar_product(phi_d[j], div_phi_s[i])
                                               + scalar_product<1, rotation_dim>(phi_r[i], asym_j)
                                               + scalar_product<1, rotation_dim>(phi_r[j], asym_i) )
                                              * fe_values.JxW(q);
                    }
                }
            }

            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    system_matrix.add (local_dof_indices[i],
                                       local_dof_indices[j],
                                       local_matrix(i,j));
        }

//      std::ofstream mat("mat.txt");
//      system_matrix.print_formatted(mat,3,1,0,"0");
//      mat.close();
    }

    // MixedBiotProblemDD - assemble_system-corresponding to Elasticity part
       template <int dim>
       void MixedBiotProblemDD<dim>::assemble_system_elast()
       {
           TimerOutput::Scope t(computing_timer, "Assemble Elasticity system");
           system_matrix_elast = 0;
           //system_rhs_bar = 0;

           QGauss<dim>   quadrature_formula(degree+3);
           //QGauss<dim-1> face_quadrature_formula(qdegree);

           FEValues<dim> fe_values (fe, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);

           const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
           const unsigned int   n_q_points      = quadrature_formula.size();
           //const unsigned int   n_face_q_points = face_quadrature_formula.size();

           FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

           std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

           const LameParameters<dim> lame_function;
//           const KInverse<dim> k_inverse;

           std::vector<Vector<double>> lame_parameters_values(n_q_points, Vector<double>(2));
//           std::vector<Tensor<2,dim>>  k_inverse_values(n_q_points);

           // Rotation variable is either a scalar(2d) or a vector(3d)
           const unsigned int rotation_dim = 0.5*dim*(dim-1);
           // Stress DoFs vectors
           std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
           std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());
           // Displacement DoFs
           const FEValuesExtractors::Vector displacement (dim*dim);

           for (unsigned int i=0; i<dim; ++i)
           {
               const FEValuesExtractors::Vector tmp_stress(i*dim);
               stresses[i].first_vector_component = tmp_stress.first_vector_component;
               if (dim == 2 && i == 0)
               {
                   const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim);
                   rotations[i].component = tmp_rotation.component;
               } else if (dim == 3) {
                   const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + i);
                   rotations[i].component = tmp_rotation.component;
               }
           }

           typename DoFHandler<dim>::active_cell_iterator
                   cell = dof_handler.begin_active(),
                   endc = dof_handler.end();
           for (; cell!=endc; ++cell)
           {

               fe_values.reinit (cell);
               local_matrix = 0;
			   cell->get_dof_indices (local_dof_indices);


               lame_function.vector_value_list (fe_values.get_quadrature_points(), lame_parameters_values);

               // Stress, displacement and rotation
               std::vector<std::vector<Tensor<1,dim>>> phi_s(dofs_per_cell, std::vector<Tensor<1,dim> > (dim));
               std::vector<Tensor<1,dim>> div_phi_s(dofs_per_cell);
               std::vector<Tensor<1,dim>> phi_d(dofs_per_cell);
               std::vector<Tensor<1,rotation_dim>> phi_r(dofs_per_cell);

               Tensor<2,dim> sigma, asigma;
//			   Tensor<2,dim>apId;
               Tensor<1,rotation_dim> asym_i, asym_j;

               for (unsigned int q=0; q<n_q_points; ++q)
               {
                   for (unsigned int k=0; k<dofs_per_cell; ++k)
                	   if(local_dof_indices[k]<n_Elast)
                	   {

						 for (unsigned int s_i=0; s_i<dim; ++s_i)
						 {
						   phi_s[k][s_i] = fe_values[stresses[s_i]].value (k, q);
						   div_phi_s[k][s_i] = fe_values[stresses[s_i]].divergence (k, q);
						 }
						 phi_d[k] = fe_values[displacement].value (k, q);

						 for (unsigned int r_i=0; r_i<rotation_dim; ++r_i)
						   phi_r[k][r_i] = fe_values[rotations[r_i]].value (k, q);
					   }

                   for (unsigned int i=0; i<dofs_per_cell; ++i)
                	   if(local_dof_indices[i]<n_Elast)
                	   {
						   const double mu = lame_parameters_values[q][1];
						   const double lambda = lame_parameters_values[q][0];

						   compliance_tensor(phi_s[i], mu, lambda, asigma);
						   make_asymmetry_tensor(phi_s[i], asym_i);

						   for (unsigned int j=0; j<dofs_per_cell; ++j)
							   if(local_dof_indices[j]<n_Elast) //making sure we are looping only over elasticity DOFs
							   {

								   make_tensor(phi_s[j], sigma);
								   make_asymmetry_tensor(phi_s[j], asym_j);

								   //std::cout << "Rotation: " << phi_r[i] << ", asym: " << asym_j << "\n";
								   local_matrix(i, j) += (  scalar_product(asigma, sigma) +                                // Mixed elasticity eq-ns
														  + scalar_product(phi_d[i], div_phi_s[j])  + scalar_product(phi_d[j], div_phi_s[i])
														  + scalar_product<1, rotation_dim>(phi_r[i], asym_j)
														  + scalar_product<1, rotation_dim>(phi_r[j], asym_i) )
														 * fe_values.JxW(q);
								   //pressure part to be added to the rhs:  prm.alpha*scalar_product(apId, sigma)
						   }
                	   }
               }


               for (unsigned int i=0; i<dofs_per_cell; ++i)
            	   if(local_dof_indices[i]<n_Elast)
					   for (unsigned int j=0; j<dofs_per_cell; ++j)
						   if(local_dof_indices[j]<n_Elast)
							   system_matrix_elast.add (local_dof_indices[i],
												  local_dof_indices[j],
												  local_matrix(i,j));
           }

       }


       // MixedBiotProblemDD - assemble_system corresponding to the Darcy part
        template <int dim>
        void MixedBiotProblemDD<dim>::assemble_system_darcy ()
        {
            TimerOutput::Scope t(computing_timer, "Assemble Darcy system");
            system_matrix_darcy = 0;
            //system_rhs_bar = 0;

            QGauss<dim>   quadrature_formula(degree+3);
            //QGauss<dim-1> face_quadrature_formula(qdegree);

            FEValues<dim> fe_values (fe, quadrature_formula,
                                     update_values    | update_gradients |
                                     update_quadrature_points  | update_JxW_values);

            const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
            const unsigned int   n_q_points      = quadrature_formula.size();
            //const unsigned int   n_face_q_points = face_quadrature_formula.size();

            FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

            std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

            const LameParameters<dim> lame_function;
            const KInverse<dim> k_inverse;

            std::vector<Vector<double>> lame_parameters_values(n_q_points, Vector<double>(2));
            std::vector<Tensor<2,dim>>  k_inverse_values(n_q_points);

            // Rotation variable is either a scalar(2d) or a vector(3d)
            const unsigned int rotation_dim = 0.5*dim*(dim-1);
            // Velocity and Pressure DoFs
            const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));
            const FEValuesExtractors::Scalar pressure (dim*dim + dim + 0.5*dim*(dim-1) + dim);


            typename DoFHandler<dim>::active_cell_iterator
                    cell = dof_handler.begin_active(),
                    endc = dof_handler.end();
            for (; cell!=endc; ++cell)
            {
                fe_values.reinit (cell);
                local_matrix = 0;
                cell->get_dof_indices (local_dof_indices);

                lame_function.vector_value_list (fe_values.get_quadrature_points(), lame_parameters_values);
                k_inverse.value_list (fe_values.get_quadrature_points(), k_inverse_values);

                // Velocity and pressure
                std::vector<Tensor<1,dim>>                phi_u(dofs_per_cell);
                std::vector <double>                      div_phi_u(dofs_per_cell);
                std::vector <double>                      phi_p(dofs_per_cell);


                Tensor<2,dim> sigma, asigma, apId;
                Tensor<1,rotation_dim> asym_i, asym_j;

                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {
                      // Evaluate test functions
                      phi_u[k] = fe_values[velocity].value (k, q);
                      phi_p[k] = fe_values[pressure].value (k, q);
                      div_phi_u[k] = fe_values[velocity].divergence (k, q);
                    }

                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    	if(local_dof_indices[i]>=n_Elast)
						{
							const double mu = lame_parameters_values[q][1];
							const double lambda = lame_parameters_values[q][0];
							compliance_tensor_pressure(phi_p[i], mu, lambda, apId);
							for (unsigned int j=0; j<dofs_per_cell; ++j)
								if(local_dof_indices[j]>=n_Elast)
								{

									//std::cout << "Rotation: " << phi_r[i] << ", asym: " << asym_j << "\n";
									local_matrix(i, j) += (  phi_u[i] * k_inverse_values[q] * phi_u[j] - phi_p[j] * div_phi_u[i]                                     // Darcy law
														   + prm.time_step*div_phi_u[j] * phi_p[i]
														   + prm.c_0*phi_p[i]*phi_p[j]   // Momentum
														   + prm.alpha * prm.alpha * trace(apId)*phi_p[j]     )
														  * fe_values.JxW(q);

									//to be added to rhs_bar_darcy += prm.alpha*trace(asigma)*phi_p[j]
								}
						}
                }


                for (unsigned int i=0; i<dofs_per_cell; ++i)
                	if(local_dof_indices[i]>=n_Elast)
						for (unsigned int j=0; j<dofs_per_cell; ++j)
							if(local_dof_indices[j]>=n_Elast)
								system_matrix_darcy.add (local_dof_indices[i]-n_Elast,
												   local_dof_indices[j]-n_Elast,
												   local_matrix(i,j));
				}


        }
    // MixedBiotProblemDD - initialize the interface data structure for coupled monolithic sheme
    template <int dim>
    void MixedBiotProblemDD<dim>::get_interface_dofs ()
    {
        TimerOutput::Scope t(computing_timer, "Get interface DoFs");
        interface_dofs.resize(GeometryInfo<dim>::faces_per_cell, std::vector<types::global_dof_index> ());

        std::vector<types::global_dof_index> local_face_dof_indices;

        typename DoFHandler<dim>::active_cell_iterator cell, endc;

        if (mortar_flag == 0)
        {
            cell = dof_handler.begin_active(), endc = dof_handler.end();
            local_face_dof_indices.resize(fe.dofs_per_face);
        }
        else
        {
            cell = dof_handler_mortar.begin_active(),
                    endc = dof_handler_mortar.end();
            local_face_dof_indices.resize(fe_mortar.dofs_per_face);
        }
//        double local_counter=0;

        for (;cell!=endc;++cell)
        {
            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() != 0)
                {
                    cell->face(face_n)->get_dof_indices (local_face_dof_indices, 0);

                    for (auto el : local_face_dof_indices){
                        if (el < n_stress){
                            interface_dofs[cell->face(face_n)->boundary_id()-1].push_back(el);
//                            local_counter++;
                        }
                        else if(el>=n_stress+n_disp+n_rot && el<n_stress+n_disp+n_rot+n_flux){
                        	interface_dofs[cell->face(face_n)->boundary_id()-1].push_back(el);
//                        	local_counter++;
                        }

                    }
                }
        }
//        pcout<<"\n size of interface dofs: "<<local_counter<<"\n";
    }

    // MixedBiotProblemDD - initialize the interface data structure for elasticity part(split scheme)
    template <int dim>
    void MixedBiotProblemDD<dim>::get_interface_dofs_elast ()
    {
        TimerOutput::Scope t(computing_timer, "Get interface DoFs");
        interface_dofs_elast.resize(GeometryInfo<dim>::faces_per_cell, std::vector<types::global_dof_index> ());

        std::vector<types::global_dof_index> local_face_dof_indices;

        typename DoFHandler<dim>::active_cell_iterator cell, endc;

        if (mortar_flag == 0)
        {
            cell = dof_handler.begin_active(), endc = dof_handler.end();
            local_face_dof_indices.resize(fe.dofs_per_face);
        }
        else
        {
            cell = dof_handler_mortar.begin_active(),
                    endc = dof_handler_mortar.end();
            local_face_dof_indices.resize(fe_mortar.dofs_per_face);
        }

        for (;cell!=endc;++cell)
        {
            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() != 0)
                {
                    cell->face(face_n)->get_dof_indices (local_face_dof_indices, 0);

                    for (auto el : local_face_dof_indices){
                        if (el < n_stress)
                            interface_dofs_elast[cell->face(face_n)->boundary_id()-1].push_back(el);



                    }
                }
        }

    }


    // MixedBiotProblemDD - initialize the interface data structure for darcy part(split schemes)
    template <int dim>
    void MixedBiotProblemDD<dim>::get_interface_dofs_darcy ()
    {
        TimerOutput::Scope t(computing_timer, "Get interface DoFs");
        interface_dofs_darcy.resize(GeometryInfo<dim>::faces_per_cell, std::vector<types::global_dof_index> ());

        std::vector<types::global_dof_index> local_face_dof_indices;

        typename DoFHandler<dim>::active_cell_iterator cell, endc;

        if (mortar_flag == 0)
        {
            cell = dof_handler.begin_active(), endc = dof_handler.end();
            local_face_dof_indices.resize(fe.dofs_per_face);
        }
        else
        {
            cell = dof_handler_mortar.begin_active(),
                    endc = dof_handler_mortar.end();
            local_face_dof_indices.resize(fe_mortar.dofs_per_face);
        }
//        double local_counter=0;

        for (;cell!=endc;++cell)
        {
            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() != 0)
                {
                    cell->face(face_n)->get_dof_indices (local_face_dof_indices, 0);

                    for (auto el : local_face_dof_indices){
                        if (el>=n_Elast)
                            interface_dofs_darcy[cell->face(face_n)->boundary_id()-1].push_back(el);

                    }
                }
        }
    }




  // MixedBiotProblemDD - assemble RHS of star problems
  template <int dim>
  void MixedBiotProblemDD<dim>::assemble_rhs_bar ()
  {
      TimerOutput::Scope t(computing_timer, "Assemble RHS bar");
      system_rhs_bar = 0;

      QGauss<dim>   quadrature_formula(degree+3);
      QGauss<dim-1> face_quadrature_formula(qdegree);

      FEValues<dim> fe_values (fe, quadrature_formula,
                               update_values    | update_gradients |
                               update_quadrature_points  | update_JxW_values);
      FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                        update_values    | update_normal_vectors |
                                        update_quadrature_points  | update_JxW_values);

      const unsigned int dofs_per_cell   = fe.dofs_per_cell;
      const unsigned int n_q_points      = fe_values.get_quadrature().size();
      const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

      Vector<double>       local_rhs (dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      DisplacementBoundaryValues<dim> displacement_boundary_values;
      PressureBoundaryValues<dim>     pressure_boundary_values;
      displacement_boundary_values.set_time(prm.time);
      pressure_boundary_values.set_time(prm.time);
      std::vector<Vector<double>> boundary_values_elast (n_face_q_points, Vector<double>(dim));
      std::vector<double>         boundary_values_flow (n_face_q_points);

      RightHandSideElasticity<dim>    right_hand_side_elasticity;
      RightHandSidePressure<dim>      right_hand_side_pressure(prm.c_0,prm.alpha);
      right_hand_side_elasticity.set_time(prm.time);
      right_hand_side_pressure.set_time(prm.time);
      std::vector<Vector<double>> rhs_values_elast (n_q_points, Vector<double>(dim));
      std::vector<double>         rhs_values_flow (n_q_points);

      const LameParameters<dim> lame_function;
      std::vector<Vector<double>> lame_parameters_values(n_q_points, Vector<double>(2));

      typename DoFHandler<dim>::active_cell_iterator
              cell = dof_handler.begin_active(),
              endc = dof_handler.end();
      for (; cell!=endc; ++cell)
      {
          local_rhs = 0;
          fe_values.reinit (cell);

          const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));
          right_hand_side_elasticity.vector_value_list (fe_values.get_quadrature_points(),
                                                        rhs_values_elast);
          right_hand_side_pressure.value_list(fe_values.get_quadrature_points(), rhs_values_flow);
          lame_function.vector_value_list (fe_values.get_quadrature_points(), lame_parameters_values);

          // Stress DoFs vectors
          std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
          std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());

          // Displacement DoFs
          const FEValuesExtractors::Vector displacement (dim*dim);
          // Velocity and Pressure DoFs
          const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));
          const FEValuesExtractors::Scalar pressure (dim*dim + dim + 0.5*dim*(dim-1) + dim);

          for (unsigned int i=0; i<dim; ++i)
          {
              const FEValuesExtractors::Vector tmp_stress(i*dim);
              stresses[i].first_vector_component = tmp_stress.first_vector_component;
              if (dim == 2 && i == 0)
              {
                  const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim);
                  rotations[i].component = tmp_rotation.component;
              } else if (dim == 3) {
                  const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + i);
                  rotations[i].component = tmp_rotation.component;
              }
          }

          std::vector <double>                      phi_p(dofs_per_cell);
          std::vector<Tensor<1,dim> >               phi_d(dofs_per_cell);

          std::vector<double> old_pressure_values(n_q_points);
          std::vector<std::vector<Tensor<1, dim>>> old_stress(dim, std::vector<Tensor<1,dim>> (n_q_points));

          fe_values[pressure].get_function_values (old_solution, old_pressure_values);
          for (unsigned int s_i=0; s_i<dim; ++s_i)
              fe_values[stresses[s_i]].get_function_values(old_solution, old_stress[s_i]);

          // Transpose, can we avoid this?
          std::vector<std::vector<Tensor<1, dim>>> old_stress_values(n_q_points, std::vector<Tensor<1,dim>> (dim));
          for (unsigned int s_i=0; s_i<dim; ++s_i)
              for (unsigned int q=0; q<n_q_points; ++q)
                  old_stress_values[q][s_i] = old_stress[s_i][q];
          /////////////////////////////////////////////////////////

          for (unsigned int q=0; q<n_q_points; ++q)
          {
              const double mu = lame_parameters_values[q][1];
              const double lambda = lame_parameters_values[q][0];

              Tensor<2,dim> asigma, apId;
              compliance_tensor<dim>(old_stress_values[q], mu, lambda, asigma);
              compliance_tensor_pressure<dim>(old_pressure_values[q], mu, lambda, apId);

              for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                  // Evaluate test functions
                  phi_p[k] = fe_values[pressure].value (k, q);
                  phi_d[k] = fe_values[displacement].value (k, q);

              }

              for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                  local_rhs(i) += (prm.time_step*phi_p[i] * rhs_values_flow[q]
                                            + prm.c_0*old_pressure_values[q] * phi_p[i]
                                            + prm.alpha * prm.alpha * trace(apId) * phi_p[i]
                                            + prm.alpha * trace(asigma) * phi_p[i] )
                                           * fe_values.JxW(q);

                for (unsigned d_i=0; d_i<dim; ++d_i)
                      local_rhs(i) += -(phi_d[i][d_i] * rhs_values_elast[q][d_i] * fe_values.JxW(q));
              }
          }


          Tensor<2,dim> sigma;
          Tensor<1,dim> sigma_n;
          for (unsigned int face_no=0;
               face_no<GeometryInfo<dim>::faces_per_cell;
               ++face_no)
              if (cell->at_boundary(face_no) && cell->face(face_no)->boundary_id() == 0) // pressure part of the boundary
              {
                  fe_face_values.reinit (cell, face_no);

                  displacement_boundary_values.vector_value_list (fe_face_values.get_quadrature_points(),
                                                                  boundary_values_elast);
                  pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), boundary_values_flow);

                  for (unsigned int q=0; q<n_face_q_points; ++q)
                      for (unsigned int i=0; i<dofs_per_cell; ++i)
                      {
                          local_rhs(i) += -(fe_face_values[velocity].value (i, q) *
                                                     fe_face_values.normal_vector(q) *
                                                     boundary_values_flow[q] *
                                                     fe_face_values.JxW(q));

                          for (unsigned int d_i=0; d_i<dim; ++d_i)
                              sigma[d_i] = fe_face_values[stresses[d_i]].value (i, q);

                          sigma_n = sigma * fe_face_values.normal_vector(q);
                          for (unsigned int d_i=0; d_i<dim; ++d_i)
                              local_rhs(i) += ((sigma_n[d_i] * boundary_values_elast[q][d_i])
                                                        * fe_face_values.JxW(q));
                      }
              }

//          local_rhs.print(std::cout);

          cell->get_dof_indices (local_dof_indices);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
              system_rhs_bar(local_dof_indices[i]) += local_rhs(i);
      }
  }

  // MixedBiotProblemDD - assemble RHS of star problems corresponding to Elasticity part
    template <int dim>
    void MixedBiotProblemDD<dim>::assemble_rhs_bar_elast ()
    {
        TimerOutput::Scope t(computing_timer, "Assemble RHS bar elast");
        system_rhs_bar_elast = 0;

        QGauss<dim>   quadrature_formula(degree+3);
        QGauss<dim-1> face_quadrature_formula(qdegree);

        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    | update_gradients |
                                 update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);

        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        DisplacementBoundaryValues<dim> displacement_boundary_values;
//        PressureBoundaryValues<dim>     pressure_boundary_values;
        displacement_boundary_values.set_time(prm.time);
//        pressure_boundary_values.set_time(prm.time);
        std::vector<Vector<double>> boundary_values_elast (n_face_q_points, Vector<double>(dim));
//        std::vector<double>         boundary_values_flow (n_face_q_points);

        RightHandSideElasticity<dim>    right_hand_side_elasticity;
//        RightHandSidePressure<dim>      right_hand_side_pressure(prm.c_0,prm.alpha);
        right_hand_side_elasticity.set_time(prm.time);
//        right_hand_side_pressure.set_time(prm.time);
        std::vector<Vector<double>> rhs_values_elast (n_q_points, Vector<double>(dim));
//        std::vector<double>         rhs_values_flow (n_q_points);

        const LameParameters<dim> lame_function;
        std::vector<Vector<double>> lame_parameters_values(n_q_points, Vector<double>(2));

        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            local_rhs = 0;
            fe_values.reinit (cell);
            cell->get_dof_indices (local_dof_indices);

            const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));
            right_hand_side_elasticity.vector_value_list (fe_values.get_quadrature_points(),
                                                          rhs_values_elast);
//            right_hand_side_pressure.value_list(fe_values.get_quadrature_points(), rhs_values_flow);
            lame_function.vector_value_list (fe_values.get_quadrature_points(), lame_parameters_values);

            // Stress DoFs vectors
            std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
            std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());

            // Displacement DoFs
            const FEValuesExtractors::Vector displacement (dim*dim);
            // Velocity and Pressure DoFs
//            const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));
            const FEValuesExtractors::Scalar pressure (dim*dim + dim + 0.5*dim*(dim-1) + dim);

            for (unsigned int i=0; i<dim; ++i)
            {
                const FEValuesExtractors::Vector tmp_stress(i*dim);
                stresses[i].first_vector_component = tmp_stress.first_vector_component;
                if (dim == 2 && i == 0)
                {
                    const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim);
                    rotations[i].component = tmp_rotation.component;
                } else if (dim == 3) {
                    const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + i);
                    rotations[i].component = tmp_rotation.component;
                }
            }

//            std::vector <double>                      phi_p(dofs_per_cell);
            std::vector<Tensor<1,dim> >               phi_d(dofs_per_cell);
            std::vector<std::vector<Tensor<1,dim>>> phi_s(dofs_per_cell, std::vector<Tensor<1,dim> > (dim));

//            std::vector<double> old_pressure_values(n_q_points);
            std::vector<double> intermediate_pressure_values(n_q_points); //pressure coming from the split

//            std::vector<std::vector<Tensor<1, dim>>> old_stress(dim, std::vector<Tensor<1,dim>> (n_q_points));

            fe_values[pressure].get_function_values (intermediate_solution, intermediate_pressure_values);
            Tensor<2,dim> sigma;
            /////////////////////////////////////////////////////////

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                const double mu = lame_parameters_values[q][1];
                const double lambda = lame_parameters_values[q][0];

                Tensor<2,dim> asigma, apId;
//                compliance_tensor<dim>(old_stress_values[q], mu, lambda, asigma);
                compliance_tensor_pressure<dim>(intermediate_pressure_values[q], mu, lambda, apId);

                for (unsigned int k=0; k<dofs_per_cell; ++k)
                	if(local_dof_indices[k]<n_Elast){
                    // Evaluate test functions
//                    phi_p[k] = fe_values[pressure].value (k, q);
                    phi_d[k] = fe_values[displacement].value (k, q);
                    for (unsigned int s_i=0; s_i<dim; ++s_i)
                    	phi_s[k][s_i] = fe_values[stresses[s_i]].value (k, q);

                }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                	if(local_dof_indices[i]<n_Elast){
					   make_tensor(phi_s[i], sigma);
                    local_rhs(i) += -(prm.alpha*scalar_product(apId, sigma) ) //we add pressure from previous step
                                             * fe_values.JxW(q);

                  for (unsigned d_i=0; d_i<dim; ++d_i)
                        local_rhs(i) += -(phi_d[i][d_i] * rhs_values_elast[q][d_i] * fe_values.JxW(q));
                }
            }


            Tensor<2,dim> sigma_face;
            Tensor<1,dim> sigma_n;
            for (unsigned int face_no=0;
                 face_no<GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
                if (cell->at_boundary(face_no) && cell->face(face_no)->boundary_id() == 0) // pressure part of the boundary
                {
                    fe_face_values.reinit (cell, face_no);

                    displacement_boundary_values.vector_value_list (fe_face_values.get_quadrature_points(),
                                                                    boundary_values_elast);
//                    pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), boundary_values_flow);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        	if(local_dof_indices[i]<n_Elast){


                            for (unsigned int d_i=0; d_i<dim; ++d_i)
                                sigma_face[d_i] = fe_face_values[stresses[d_i]].value (i, q);

                            sigma_n = sigma_face * fe_face_values.normal_vector(q);
                            for (unsigned int d_i=0; d_i<dim; ++d_i)
                                local_rhs(i) += ((sigma_n[d_i] * boundary_values_elast[q][d_i])
                                                          * fe_face_values.JxW(q));
                        }
                }



            for (unsigned int i=0; i<dofs_per_cell; ++i)
            	if(local_dof_indices[i]<n_Elast)
            		system_rhs_bar_elast(local_dof_indices[i]) += local_rhs(i);
        }
    }


    // MixedBiotProblemDD - assemble RHS of star problems corresponding to flow part
    template <int dim>
    void MixedBiotProblemDD<dim>::assemble_rhs_bar_darcy ()
    {
        TimerOutput::Scope t(computing_timer, "Assemble RHS bar darcy");
        system_rhs_bar_darcy = 0;

        QGauss<dim>   quadrature_formula(degree+3);
        QGauss<dim-1> face_quadrature_formula(qdegree);

        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    | update_gradients |
                                 update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);

        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

//        DisplacementBoundaryValues<dim> displacement_boundary_values;
        PressureBoundaryValues<dim>     pressure_boundary_values;
//        displacement_boundary_values.set_time(prm.time);
        pressure_boundary_values.set_time(prm.time);
//        std::vector<Vector<double>> boundary_values_elast (n_face_q_points, Vector<double>(dim));
        std::vector<double>         boundary_values_flow (n_face_q_points);

//        RightHandSideElasticity<dim>    right_hand_side_elasticity;
        RightHandSidePressure<dim>      right_hand_side_pressure(prm.c_0,prm.alpha);
//        right_hand_side_elasticity.set_time(prm.time);
        right_hand_side_pressure.set_time(prm.time);
//        std::vector<Vector<double>> rhs_values_elast (n_q_points, Vector<double>(dim));
        std::vector<double>         rhs_values_flow (n_q_points);

        const LameParameters<dim> lame_function;
        std::vector<Vector<double>> lame_parameters_values(n_q_points, Vector<double>(2));

        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            local_rhs = 0;
            fe_values.reinit (cell);
            cell->get_dof_indices (local_dof_indices);

//            const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));
//            right_hand_side_elasticity.vector_value_list (fe_values.get_quadrature_points(),
//                                                          rhs_values_elast);
            right_hand_side_pressure.value_list(fe_values.get_quadrature_points(), rhs_values_flow);
            lame_function.vector_value_list (fe_values.get_quadrature_points(), lame_parameters_values);

            // Stress DoFs vectors
            std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
//            std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());

            // Displacement DoFs
//            const FEValuesExtractors::Vector displacement (dim*dim);
            // Velocity and Pressure DoFs
            const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));
            const FEValuesExtractors::Scalar pressure (dim*dim + dim + 0.5*dim*(dim-1) + dim);

            for (unsigned int i=0; i<dim; ++i)
            {
                const FEValuesExtractors::Vector tmp_stress(i*dim);
                stresses[i].first_vector_component = tmp_stress.first_vector_component;
//                if (dim == 2 && i == 0)
//                {
//                    const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim);
//                    rotations[i].component = tmp_rotation.component;
//                } else if (dim == 3) {
//                    const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + i);
//                    rotations[i].component = tmp_rotation.component;
//                }
            }
            // Stress and pressure
            std::vector<std::vector<Tensor<1,dim>>> phi_s(dofs_per_cell, std::vector<Tensor<1,dim> > (dim));
            std::vector <double>                      phi_p(dofs_per_cell);
//            std::vector<Tensor<1,dim> >               phi_d(dofs_per_cell);

            std::vector<double> old_pressure_values(n_q_points);
            std::vector<double> intermediate_pressure_values(n_q_points);
            std::vector<std::vector<Tensor<1, dim>>> old_stress(dim, std::vector<Tensor<1,dim>> (n_q_points));
            std::vector<std::vector<Tensor<1, dim>>> intermediate_stress(dim, std::vector<Tensor<1,dim>> (n_q_points));

            fe_values[pressure].get_function_values (old_solution, old_pressure_values);
            for (unsigned int s_i=0; s_i<dim; ++s_i){
            	if(split_flag==1)
            		fe_values[stresses[s_i]].get_function_values(old_solution, old_stress[s_i]);
            	else if(split_flag==2)
            		fe_values[stresses[s_i]].get_function_values(older_solution, old_stress[s_i]);

                fe_values[stresses[s_i]].get_function_values(intermediate_solution, intermediate_stress[s_i]);

            }
            // Transpose, can we avoid this?
            std::vector<std::vector<Tensor<1, dim>>> old_stress_values(n_q_points, std::vector<Tensor<1,dim>> (dim));
            std::vector<std::vector<Tensor<1, dim>>> intermediate_stress_values(n_q_points, std::vector<Tensor<1,dim>> (dim));
            for (unsigned int s_i=0; s_i<dim; ++s_i)
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    old_stress_values[q][s_i] = old_stress[s_i][q];
            		intermediate_stress_values[q][s_i] = intermediate_stress[s_i][q];
                }
            /////////////////////////////////////////////////////////

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                const double mu = lame_parameters_values[q][1];
                const double lambda = lame_parameters_values[q][0];

                Tensor<2,dim> asigma, apId;
                Tensor<2,dim> asigma2; //related to splitting
                compliance_tensor<dim>(intermediate_stress_values[q], mu, lambda, asigma2);
                compliance_tensor<dim>(old_stress_values[q], mu, lambda, asigma);
                compliance_tensor_pressure<dim>(old_pressure_values[q], mu, lambda, apId);

                for (unsigned int k=0; k<dofs_per_cell; ++k)
                	if(local_dof_indices[k]>=n_Elast){
                    // Evaluate test functions
                    phi_p[k] = fe_values[pressure].value (k, q);
//                    phi_d[k] = fe_values[displacement].value (k, q);

                }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                	if(local_dof_indices[i]>=n_Elast){
                    local_rhs(i) += (prm.time_step*phi_p[i] * rhs_values_flow[q]
                                              + prm.c_0*old_pressure_values[q] * phi_p[i]
                                              + prm.alpha * prm.alpha * trace(apId) * phi_p[i]
                                              + prm.alpha * trace(asigma) * phi_p[i]
										      - prm.alpha * trace(asigma2) * phi_p[i]									  )
                                             * fe_values.JxW(q);

                }
            }


            Tensor<2,dim> sigma;
            Tensor<1,dim> sigma_n;
            for (unsigned int face_no=0;
                 face_no<GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
                if (cell->at_boundary(face_no) && cell->face(face_no)->boundary_id() == 0) // pressure part of the boundary
                {
                    fe_face_values.reinit (cell, face_no);


                    pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), boundary_values_flow);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        	if(local_dof_indices[i]>=n_Elast){
                            local_rhs(i) += -(fe_face_values[velocity].value (i, q) *
                                                       fe_face_values.normal_vector(q) *
                                                       boundary_values_flow[q] *
                                                       fe_face_values.JxW(q));


                        }
                }



            for (unsigned int i=0; i<dofs_per_cell; ++i)
            	if(local_dof_indices[i]>=n_Elast)
            		system_rhs_bar_darcy(local_dof_indices[i]-n_Elast) += local_rhs(i);
        }
    }

    // MixedBiotProblemDD - assemble RHS of star problems
    template <int dim>
    void MixedBiotProblemDD<dim>::assemble_rhs_star (FEFaceValues<dim> &fe_face_values)
    {
        TimerOutput::Scope t(computing_timer, "Assemble RHS star");
        system_rhs_star = 0;

        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());

        for (unsigned int d=0; d<dim; ++d)
        {
            const FEValuesExtractors::Vector tmp_stress(d*dim);
            stresses[d].first_vector_component = tmp_stress.first_vector_component;

        }

        const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));

        std::vector<std::vector<Tensor<1, dim>>> interface_values(dim, std::vector<Tensor<1, dim>> (n_face_q_points));
        std::vector<Tensor<1, dim>> interface_values_flux(n_face_q_points);

        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
        for (;cell!=endc;++cell)
        {
            local_rhs = 0;

            Tensor<2,dim> sigma;
            Tensor<2,dim> interface_lambda;
            Tensor<1,dim> sigma_n;
            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() != 0)
                {
                    fe_face_values.reinit (cell, face_n);

                    for (unsigned int d_i=0; d_i<dim; ++d_i)
                        fe_face_values[stresses[d_i]].get_function_values (interface_fe_function, interface_values[d_i]);

                    fe_face_values[velocity].get_function_values (interface_fe_function, interface_values_flux);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            local_rhs(i) += -(fe_face_values[velocity].value (i, q) *
                                              fe_face_values.normal_vector(q) *
                                              interface_values_flux[q] * get_normal_direction(cell->face(face_n)->boundary_id()-1) *
                                              fe_face_values.normal_vector(q) *
                                              fe_face_values.JxW(q));

                            for (unsigned int d_i=0; d_i<dim; ++d_i)
                                sigma[d_i] = fe_face_values[stresses[d_i]].value (i, q);

                            for (unsigned int d_i=0; d_i<dim; ++d_i)
                                local_rhs(i) += fe_face_values[stresses[d_i]].value (i, q) *
                                                fe_face_values.normal_vector(q) *
                                                interface_values[d_i][q] * get_normal_direction(cell->face(face_n)->boundary_id()-1) *
                                                fe_face_values.normal_vector(q) *
                                                fe_face_values.JxW(q);
                        }
                }

            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                system_rhs_star(local_dof_indices[i]) += local_rhs(i);
            //
        }
    }


    // MixedBiotProblemDD - assemble RHS of star problem corrsponding to Elast
    template <int dim>
    void MixedBiotProblemDD<dim>::assemble_rhs_star_elast (FEFaceValues<dim> &fe_face_values)
    {
        TimerOutput::Scope t(computing_timer, "Assemble RHS star elast");
        system_rhs_star_elast = 0;

        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());

        for (unsigned int d=0; d<dim; ++d)
        {
            const FEValuesExtractors::Vector tmp_stress(d*dim);
            stresses[d].first_vector_component = tmp_stress.first_vector_component;

        }


        std::vector<std::vector<Tensor<1, dim>>> interface_values(dim, std::vector<Tensor<1, dim>> (n_face_q_points));

        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
        for (;cell!=endc;++cell)
        {
            local_rhs = 0;
            cell->get_dof_indices (local_dof_indices);

            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() != 0)
                {
                    fe_face_values.reinit (cell, face_n);

                    for (unsigned int d_i=0; d_i<dim; ++d_i)
                        fe_face_values[stresses[d_i]].get_function_values (interface_fe_function, interface_values[d_i]);


                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        	if(local_dof_indices[i]<n_Elast)
                        	{

								for (unsigned int d_i=0; d_i<dim; ++d_i)
									local_rhs(i) += fe_face_values[stresses[d_i]].value (i, q) *
													fe_face_values.normal_vector(q) *
													interface_values[d_i][q] * get_normal_direction(cell->face(face_n)->boundary_id()-1) *
													fe_face_values.normal_vector(q) *
													fe_face_values.JxW(q);
                        }
                }


            for (unsigned int i=0; i<dofs_per_cell; ++i)
            	if(local_dof_indices[i]<n_Elast)
            		system_rhs_star_elast(local_dof_indices[i]) += local_rhs(i);
            //
        }
    }

    // MixedBiotProblemDD - assemble RHS of star problem cosrresponding to Flow problem
      template <int dim>
      void MixedBiotProblemDD<dim>::assemble_rhs_star_darcy (FEFaceValues<dim> &fe_face_values)
      {
          TimerOutput::Scope t(computing_timer, "Assemble RHS star Darcy");
          system_rhs_star_darcy = 0;

          const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
          const unsigned int dofs_per_cell = fe.dofs_per_cell;

          Vector<double>       local_rhs (dofs_per_cell);
          std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


          const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));

          std::vector<Tensor<1, dim>> interface_values_flux(n_face_q_points);

          typename DoFHandler<dim>::active_cell_iterator
                  cell = dof_handler.begin_active(),
                  endc = dof_handler.end();
          for (;cell!=endc;++cell)
          {
              local_rhs = 0;
              cell->get_dof_indices (local_dof_indices);

              for (unsigned int face_n=0;
                   face_n<GeometryInfo<dim>::faces_per_cell;
                   ++face_n)
                  if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() != 0)
                  {
                      fe_face_values.reinit (cell, face_n);

                      fe_face_values[velocity].get_function_values (interface_fe_function, interface_values_flux);

                      for (unsigned int q=0; q<n_face_q_points; ++q)
                          for (unsigned int i=0; i<dofs_per_cell; ++i)
                        	  if(local_dof_indices[i]>=n_Elast)
							  {
								  local_rhs(i) += -(fe_face_values[velocity].value (i, q) *
													fe_face_values.normal_vector(q) *
													interface_values_flux[q] * get_normal_direction(cell->face(face_n)->boundary_id()-1) *
													fe_face_values.normal_vector(q) *
													fe_face_values.JxW(q));
							  }
                  }


              for (unsigned int i=0; i<dofs_per_cell; ++i)
            	  if(local_dof_indices[i]>=n_Elast)
            		  system_rhs_star_darcy(local_dof_indices[i]-n_Elast) += local_rhs(i);
              //
          }
      }
    // MixedBiotProblemDD::solvers
    template <int dim>
    void MixedBiotProblemDD<dim>::solve_bar ()
    {
        TimerOutput::Scope t(computing_timer, "Solve bar");

        if (cg_iteration == 0 && prm.time == prm.time_step)
        {

//          A_direct.initialize(system_matrix);
          pcout << "  ...factorized..." << "\n";
          A_direct.initialize(system_matrix);
        }

        A_direct.vmult (solution_bar, system_rhs_bar);


    }

    template <int dim>
    void MixedBiotProblemDD<dim>::solve_bar_elast()
    {
        TimerOutput::Scope t(computing_timer, "Solve bar Elast");

        if (cg_iteration == 0 && prm.time == prm.time_step)
        {
          A_direct_elast.initialize(system_matrix_elast);
          pcout << "  ...factorized Elast..." << "\n";
        }
//
        A_direct_elast.vmult (solution_bar_elast, system_rhs_bar_elast);
    }

    template <int dim>
     void MixedBiotProblemDD<dim>::solve_bar_darcy ()
     {
         TimerOutput::Scope t(computing_timer, "Solve bar Darcy");

         if (cg_iteration == 0 && prm.time == prm.time_step)
         {
           A_direct_darcy.initialize(system_matrix_darcy);
           pcout << "  ...factorized Darcy..." << "\n";
         }

         A_direct_darcy.vmult (solution_bar_darcy, system_rhs_bar_darcy);
     }

    template <int dim>
    void MixedBiotProblemDD<dim>::solve_star ()
    {
        TimerOutput::Scope t(computing_timer, "Solve star");

        A_direct.vmult (solution_star, system_rhs_star);

    }

    template <int dim>
      void MixedBiotProblemDD<dim>::solve_star_elast ()
      {
          TimerOutput::Scope t(computing_timer, "Solve star Elast");

          A_direct_elast.vmult (solution_star_elast, system_rhs_star_elast);

      }

    template <int dim>
      void MixedBiotProblemDD<dim>::solve_star_darcy ()
      {
          TimerOutput::Scope t(computing_timer, "Solve star Darcy");

          A_direct_darcy.vmult (solution_star_darcy, system_rhs_star_darcy);

      }

    template<int dim>
    void MixedBiotProblemDD<dim>::solve_timestep(unsigned int maxiter)
    {
    	if(split_flag==0)
    	{
		  if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
		  {

			assemble_rhs_bar ();
			//old_solution.print(std::cout);
			//system_rhs_bar = 0;

			solve_bar ();
			solution = solution_bar;
			system_rhs_bar = 0;
		  }
		  else
		  {
			pcout << "\nStarting GMRES iterations, time t=" << prm.time << "s..." << "\n";
			assemble_rhs_bar ();
	//        local_cg(maxiter);
			local_gmres (maxiter);

			if (cg_iteration > max_cg_iteration)
			  max_cg_iteration = cg_iteration;

			system_rhs_bar = 0;
			system_rhs_star = 0;
			  cg_iteration = 0;
		  }
    	}
    	else if(split_flag==1) //drained split: solving elasticity first with pressure from previous step
        	{
    		  if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
    		  {
    			  //solving Elasticity part
    			intermediate_solution.block(4)= old_solution.block(4);
    			assemble_rhs_bar_elast ();
    			//old_solution.print(std::cout);
    			//system_rhs_bar = 0;
    			solve_bar_elast ();
    			//updating solution
    			solution.block(0) = solution_bar_elast.block(0);
    			solution.block(1) = solution_bar_elast.block(1);
    			solution.block(2) = solution_bar_elast.block(2);
    			system_rhs_bar_elast = 0;
    			//end of solving elasiticity part and updating solution

    			//solving Darcy part
    			intermediate_solution.block(0)=solution.block(0);
    			assemble_rhs_bar_darcy ();
    			solve_bar_darcy();
    			//updating solution
    			solution.block(3) = solution_bar_darcy.block(0);
    			solution.block(4) = solution_bar_darcy.block(1);
    			system_rhs_bar_darcy=0;

    		  }
    		  else
    		  {
    			pcout << "\nStarting Elast CG iterations, time t=" << prm.time << "s..." << "\n";
    			//solving Elasticity part
    			intermediate_solution.block(4)= old_solution.block(4);
    			assemble_rhs_bar_elast ();
    	        local_cg(maxiter,0);
//    			local_cg_elast(maxiter);
    	        system_rhs_bar_elast=0;
    	        system_rhs_star_elast=0;
      			if (cg_iteration > max_cg_iteration)
      			  max_cg_iteration = cg_iteration;
      			cg_iteration = 0;

      			//solving Darcy part
      			 intermediate_solution.block(0)=solution.block(0);
      			 assemble_rhs_bar_darcy ();
      			pcout << "\nStarting Darcy CG iterations, time t=" << prm.time << "s..." << "\n";
      			local_cg(maxiter,1);
//      			 local_cg_darcy(maxiter);
      			 system_rhs_bar_darcy=0;
      			 system_rhs_star_darcy=0;
     			if (cg_iteration > max_cg_iteration_darcy)
           			  max_cg_iteration_darcy = cg_iteration;
      			 cg_iteration=0;



    		  }
        	}

    	else if(split_flag==2) //fixed stress: solving flow problem first with trace of stress from previous step.
        	{
    		  if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
    		  {

    			  //solving Darcy part
    			 intermediate_solution.block(0)=old_solution.block(0);
    			 assemble_rhs_bar_darcy ();
    			 solve_bar_darcy();
    			 //updating solution
    			 solution.block(3) = solution_bar_darcy.block(0);
    			 solution.block(4) = solution_bar_darcy.block(1);
    			 system_rhs_bar_darcy=0;


    			  //solving Elasticity part
    			intermediate_solution.block(4)= solution.block(4);
    			assemble_rhs_bar_elast ();
    			//old_solution.print(std::cout);
    			//system_rhs_bar = 0;
    			solve_bar_elast ();
    			//updating solution
    			solution.block(0) = solution_bar_elast.block(0);
    			solution.block(1) = solution_bar_elast.block(1);
    			solution.block(2) = solution_bar_elast.block(2);
    			system_rhs_bar_elast = 0;
    			//end of solving elasiticity part and updating solution



    		  }
    		  else
    		  {

    			  //solving Darcy part
    			 intermediate_solution.block(0)=old_solution.block(0);
    			 assemble_rhs_bar_darcy ();
    			 pcout << "\nStarting Darcy CG iterations, time t=" << prm.time << "s..." << "\n";
    			 local_cg(maxiter,1);
    			//local_cg_darcy(maxiter);
    			system_rhs_bar_darcy=0;
    			system_rhs_star_darcy=0;
    			pcout << "\nStarting Elast CG iterations, time t=" << prm.time << "s..." << "\n";
    			if (cg_iteration > max_cg_iteration_darcy)
    			   max_cg_iteration_darcy = cg_iteration;
    			cg_iteration=0;

    			//solving Elasticity part
    			intermediate_solution.block(4)= solution.block(4);
    			assemble_rhs_bar_elast ();
    	        local_cg(maxiter,0);
//    			local_cg_elast(maxiter);
    	        system_rhs_bar_elast=0;
    	        system_rhs_star_elast=0;
      			if (cg_iteration > max_cg_iteration)
      			  max_cg_iteration = cg_iteration;
      			 cg_iteration=0;



    		  }
        	}
    }

    template <int dim>
    void MixedBiotProblemDD<dim>::compute_multiscale_basis ()
    {
        TimerOutput::Scope t(computing_timer, "Compute multiscale basis");
        ConstraintMatrix constraints;
        QGauss<dim-1> quad(qdegree);
        FEFaceValues<dim> fe_face_values (fe, quad,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);

        std::vector<size_t> block_sizes {solution_bar_mortar.block(0).size(), solution_bar_mortar.block(1).size()};
        long n_interface_dofs = 0;

        for (auto vec : interface_dofs)
            for (auto el : vec)
                n_interface_dofs += 1;

        multiscale_basis.resize(n_interface_dofs);
        BlockVector<double> tmp_basis (solution_bar_mortar);

        interface_fe_function.reinit(solution_bar);

        unsigned int ind = 0;
        for (unsigned int side=0; side<GeometryInfo<dim>::faces_per_cell; ++side)
            for (unsigned int i=0; i<interface_dofs[side].size(); ++i)
            {
                interface_fe_function = 0;
                multiscale_basis[ind].reinit(solution_bar_mortar);
                multiscale_basis[ind] = 0;

                tmp_basis = 0;
                tmp_basis[interface_dofs[side][i]] = 1.0;
                project_mortar(P_coarse2fine, dof_handler_mortar, tmp_basis, quad, constraints, neighbors, dof_handler, interface_fe_function);

                interface_fe_function.block(1) = 0;
                interface_fe_function.block(2) = 0;
                interface_fe_function.block(4) = 0;
                assemble_rhs_star(fe_face_values);
                solve_star();

                project_mortar(P_fine2coarse, dof_handler, solution_star, quad, constraints, neighbors, dof_handler_mortar, multiscale_basis[ind]);
                ind += 1;
            }

    }

    //Functions for GMRES:-------------------


      //finding the l2 norm of a std::vector<double> vector
      template <int dim>
      double
	  MixedBiotProblemDD<dim>::vect_norm(std::vector<double> v){
      	double result = 0;
      	for(unsigned int i=0; i<v.size(); ++i){
      		result+= v[i]*v[i];
      	}
      	return sqrt(result);

      }
      //Calculating the given rotation matrix
      template <int dim>
      void
	  MixedBiotProblemDD<dim>::givens_rotation(double v1, double v2, double &cs, double &sn){

      	if(fabs(v1)<1e-15){
      		cs=0;
      		sn=1;
      	}
      	else{
      		double t = sqrt(v1*v1 + v2*v2);
      		cs = fabs(v1)/t;
      		sn=cs*v2/v1;
      	}


      }

      //Applying givens rotation to H column
      template <int dim>
      void
	  MixedBiotProblemDD<dim>::apply_givens_rotation(std::vector<double> &h, std::vector<double> &cs, std::vector<double> &sn,
      							unsigned int k_iteration){
    	  int k=k_iteration;
      	assert(h.size()>k+1); //size should be k+2
      	double temp;
      	for( int i=0; i<=k-1; ++i){

      		temp= cs[i]* h[i]+ sn[i]*h[i+1];
    //  		pcout<<"\n temp value is: "<<temp<<"\n";
      		h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1];
      		h[i] = temp;
      	}
      	assert(h.size()==k+2);
      	//update the next sin cos values for rotation
      	double cs_k=0, sn_k=0;
      	 givens_rotation(h[k],h[k+1],cs_k,sn_k);


      	 //Eliminate H(i+1,i)
      	 h[k] = cs_k*h[k] + sn_k*h[k+1];
      	 h[k+1] = 0.0;
      	 cs[k]=cs_k;
      	 sn[k]=sn_k;

      }


      template <int dim>
      void
	  MixedBiotProblemDD<dim>::back_solve(std::vector<std::vector<double>> H, std::vector<double> beta, std::vector<double> &y, unsigned int k_iteration){
      	 int k = k_iteration;
      	 assert(y.size()==k_iteration+1);
      	 for(int i=0; i<k_iteration;i++)
      		 y[i]=0;
      	for( int i =k-1; i>=0;i-- ){
      		y[i]= beta[i]/H[i][i];
      		for( int j = i+1; j<=k-1;j++){
      			y[i]-= H[j][i]*y[j]/H[i][i];
      		}
      	}

      }

    //local GMRES function.
      template <int dim>
        void
		MixedBiotProblemDD<dim>::local_gmres(const unsigned int maxiter)
        {
          TimerOutput::Scope t(computing_timer, "Local GMRES");

          const unsigned int this_mpi =
            Utilities::MPI::this_mpi_process(mpi_communicator);
          const unsigned int n_processes =
            Utilities::MPI::n_mpi_processes(mpi_communicator);
          const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

          std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
          std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
          std::vector<std::vector<double>> interface_data(n_faces_per_cell);
          std::vector<std::vector<double>> lambda(n_faces_per_cell);


          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
              {
                    interface_data_receive[side].resize(interface_dofs[side].size(),
                                                        0);
                    interface_data_send[side].resize(interface_dofs[side].size(), 0);
                    interface_data[side].resize(interface_dofs[side].size(), 0);

              }

          // Extra for projections from mortar to fine grid and RHS assembly
          Quadrature<dim - 1> quad;
          quad = QGauss<dim - 1>(qdegree);


          ConstraintMatrix  constraints;
          constraints.clear();
          constraints.close();
          FEFaceValues<dim> fe_face_values(fe,
                                           quad,
                                           update_values | update_normal_vectors |
                                             update_quadrature_points |
											 update_JxW_values);
          int temp_array_size = maxiter/4;
          //GMRES structures and parameters
          std::vector<double>	sn(temp_array_size);
          std::vector<double>	cs(temp_array_size);
    //      std::vector<double>	e1;
          std::vector<double>	Beta(temp_array_size); //beta for each side
          std::vector<std::vector<double>>	H(temp_array_size,Beta);
    //      std::vector<double> error_iter_side(n_faces_per_cell); //saves error in each iteration
          std::vector<double> e_all_iter(temp_array_size+1); //error will be saved here after each iteration
          double combined_error_iter =0; //sum of error_iter_side







          // CG structures and parameters
          std::vector<double> alpha_side(n_faces_per_cell, 0),
            alpha_side_d(n_faces_per_cell, 0), beta_side(n_faces_per_cell, 0),
            beta_side_d(n_faces_per_cell, 0); //to be deleted
          std::vector<double> alpha(2, 0), beta(2, 0); //to be deleted

          std::vector<std::vector<double>> r(n_faces_per_cell); //to be deleted probably: p?
          std::vector<double> r_norm_side(n_faces_per_cell,0);
          std::vector<std::vector<std::vector<double>>>	Q_side(n_faces_per_cell) ;
          std::vector<std::vector<double>>  Ap(n_faces_per_cell);

          //defing q  to push_back to Q (reused in Arnoldi algorithm)
          std::vector<std::vector<double>> q(n_faces_per_cell);

          solve_bar();
//          std::ofstream rhs_output_file("solution_bar.txt");
//          for(int i =0; i<solution_bar.size();i++)
//        	  rhs_output_file<<i<<" : "<<solution_bar[i]<<"\n";

          interface_fe_function.reinit(solution_bar);

          if (mortar_flag == 1)
          {
              interface_fe_function_mortar.reinit(solution_bar_mortar);
              project_mortar(P_fine2coarse, dof_handler, solution_bar, quad, constraints, neighbors, dof_handler_mortar, solution_bar_mortar);
          }
          else if (mortar_flag == 2)
          {
              interface_fe_function_mortar.reinit(solution_bar_mortar);
              solution_star_mortar = 0;

              // The computation of multiscale basis must necessarilly be after solve_bar() call,
              // as in solve bar we factorize the system matrix into matrix A and clear the system matrix
              // for the sake of memory. Same for solve_star() calls, they should only appear after the solve_bar()
              compute_multiscale_basis();
              pcout << "Done computing multiscale basis\n";
              project_mortar(P_fine2coarse, dof_handler, solution_bar, quad, constraints, neighbors, dof_handler_mortar, solution_bar_mortar);

              // Instead of solving subdomain problems we compute the response using basis
              unsigned int j=0;
              for (unsigned int side=0; side<n_faces_per_cell; ++side)
                  for (unsigned int i=0;i<interface_dofs[side].size();++i)
                  {
//                      solution_star_mortar.sadd(1.0, interface_fe_function_mortar[interface_dofs[side][i]], multiscale_basis[j]);
                	  solution_star_mortar.block(0).sadd(1.0, interface_fe_function_mortar[interface_dofs[side][i]], multiscale_basis[j].block(0));
                      solution_star_mortar.block(3).sadd(1.0, interface_fe_function_mortar[interface_dofs[side][i]], multiscale_basis[j].block(3));
                      j += 1;
                  }
          }


          double l0 = 0.0;
          // CG with rhs being 0 and initial guess lambda = 0
          for (unsigned side = 0; side < n_faces_per_cell; ++side)

            if (neighbors[side] >= 0)
              {

                // Something will be here to initialize lambda correctly, right now it
                // is just zero
                Ap[side].resize(interface_dofs[side].size(), 0);
                lambda[side].resize(interface_dofs[side].size(), 0);
                if (true || prm.time == prm.time_step)
						{
							Ap[side].resize(interface_dofs[side].size(), 0);
							lambda[side].resize(interface_dofs[side].size(), 0);
						}
						else
						{
							Ap = Alambda_guess;
							lambda = lambda_guess;
						}

                q[side].resize(interface_dofs[side].size());
                r[side].resize(interface_dofs[side].size(), 0);
                std::vector<double> r_receive_buffer(r[side].size());
                //temporarily fixing a size for Q_side matrix
                Q_side[side].resize(temp_array_size+1,q[side]);


                // Right now it is effectively solution_bar - A\lambda (0)
                if(mortar_flag)
                	for (unsigned int i=0;i<interface_dofs[side].size();++i){
                	                      r[side][i] = get_normal_direction(side) * solution_bar_mortar[interface_dofs[side][i]]
                	                                   - get_normal_direction(side) * Ap[side][i];
//                	                      pcout<<"r[side][i] is: "<<solution_bar_mortar[interface_dofs[side][i]]<<"\n";
                	}
                else
                	for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
						r[side][i] = get_normal_direction(side) *
									   solution_bar[interface_dofs[side][i]] -
									   get_normal_direction(side) *solution_star[interface_dofs[side][i]] ;



                MPI_Send(&r[side][0],
                         r[side].size(),
                         MPI_DOUBLE,
                         neighbors[side],
                         this_mpi,
                         mpi_communicator);
                MPI_Recv(&r_receive_buffer[0],
                         r_receive_buffer.size(),
                         MPI_DOUBLE,
                         neighbors[side],
                         neighbors[side],
                         mpi_communicator,
                         &mpi_status);

                for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                  {
                    r[side][i] += r_receive_buffer[i];
                  }
                r_norm_side[side] = vect_norm(r[side]);



              }


          //Calculating r-norm(same as b-norm)-----
          double r_norm =0;
          for(unsigned int side=0; side<n_faces_per_cell;++side)
        	  if (neighbors[side] >= 0)
        		  r_norm+=r_norm_side[side]*r_norm_side[side];
          double r_norm_buffer =0;
          MPI_Allreduce(&r_norm,
        		  &r_norm_buffer,
    			  1,
    			  MPI_DOUBLE,
    			  MPI_SUM,
                  mpi_communicator);
          r_norm = sqrt(r_norm_buffer);
          //end -----------of calclatig r-norm------------------

          //Making the first element of matrix Q[side] same as r_side[side/r_norm
          for(unsigned int side=0; side<n_faces_per_cell;++side)
             	  if (neighbors[side] >= 0){
             		  for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                                   q[side][i]= r[side][i]/r_norm ;
             		 //adding q[side] as first element of Q[side]
//             		  Q_side[side].push_back(q[side]);
             		 Q_side[side][0] = q[side];
             	  }


          //end----------- of caluclating first element of Q[side]-----------------
//          e_all_iter.push_back(1.0);
          e_all_iter[0]=r_norm;
          pcout<<"\n\n r_norm is \n\n"<<r_norm<<"\n\n";
//          Beta.push_back(r_norm);
          Beta[0]=r_norm;






//          std::ofstream rhs_output_file("interface_data_side1.txt");

          unsigned int k_counter = 0; //same as the count of the iteration
          while (k_counter < maxiter)
            {
        	  //resizing cs,sn, Beta,H and Q if needed
        	  if(temp_array_size<k_counter-2){
        		  temp_array_size*=2;
        		  cs.resize(temp_array_size);
        		  sn.resize(temp_array_size);
        		  e_all_iter.resize(temp_array_size);
        		  Beta.resize(temp_array_size);
        		  H.resize(temp_array_size,Beta);
        		  for(unsigned int side=0; side<n_faces_per_cell;++side)
        		               	  if (neighbors[side] >= 0){
        		               		  std::vector<double> tmp_vector(interface_dofs[side].size());
        		               		  Q_side[side].resize(temp_array_size+1,tmp_vector);
        		               	  }
        	  }// END OF resizng vectors and arrays


      //////------solving the  star problem to find AQ(k)---------------------

              //Performing the Arnoldi algorithm
              //interface data will be given as Q_side[side][k_counter];
              for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            	  if (neighbors[side] >= 0)
            		  interface_data[side]=Q_side[side][k_counter];


              if (mortar_flag == 1)
              {
                  for (unsigned int side=0;side<n_faces_per_cell;++side)
                      for (unsigned int i=0;i<interface_dofs[side].size();++i)
                          interface_fe_function_mortar[interface_dofs[side][i]] = interface_data[side][i];

                  project_mortar(P_coarse2fine, dof_handler_mortar,
                                 interface_fe_function_mortar,
                                 quad,
                                 constraints,
                                 neighbors,
                                 dof_handler,
                                 interface_fe_function);

                  interface_fe_function.block(2) = 0;

                  assemble_rhs_star(fe_face_values);
                  solve_star();
              }
              else if (mortar_flag == 2)
              {
                  solution_star_mortar = 0;
                  unsigned int j=0;
                  for (unsigned int side=0; side<n_faces_per_cell; ++side)
                      for (unsigned int i=0;i<interface_dofs[side].size();++i)
                      {

//                          solution_star_mortar.sadd(1.0, interface_data[side][i], multiscale_basis[j]);
                          solution_star_mortar.block(0).sadd(1.0, interface_data[side][i], multiscale_basis[j].block(0));
                        solution_star_mortar.block(3).sadd(1.0, interface_data[side][i], multiscale_basis[j].block(3));
                          j += 1;
                      }

              }
              else
              {
                  for (unsigned int side=0; side<n_faces_per_cell; ++side)
                      for (unsigned int i=0; i<interface_dofs[side].size(); ++i)
                          interface_fe_function[interface_dofs[side][i]] = interface_data[side][i];

                  interface_fe_function.block(2) = 0;
                  assemble_rhs_star(fe_face_values);
                  solve_star();
              }




              cg_iteration++;
              if (mortar_flag == 1){
                        project_mortar(P_fine2coarse,
                                       dof_handler,
                                       solution_star,
                                       quad,
                                       constraints,
                                       neighbors,
                                       dof_handler_mortar,
                                       solution_star_mortar);

              }

              //defing q  to push_back to Q (Arnoldi algorithm)
              //defing h  to push_back to H (Arnoldi algorithm)
              std::vector<double> h(k_counter+2,0);


              for (unsigned int side = 0; side < n_faces_per_cell; ++side)
                if (neighbors[side] >= 0)
                  {

                    // Create vector of u\dot n to send
                    if (mortar_flag)
                        for (unsigned int i=0; i<interface_dofs[side].size(); ++i)
                            interface_data_send[side][i] = get_normal_direction(side) * solution_star_mortar[interface_dofs[side][i]];
                    else
                        for (unsigned int i=0; i<interface_dofs[side].size(); ++i)
                            interface_data_send[side][i] = get_normal_direction(side) * solution_star[interface_dofs[side][i]];

                    MPI_Send(&interface_data_send[side][0],
                             interface_dofs[side].size(),
                             MPI_DOUBLE,
                             neighbors[side],
                             this_mpi,
                             mpi_communicator);
                    MPI_Recv(&interface_data_receive[side][0],
                             interface_dofs[side].size(),
                             MPI_DOUBLE,
                             neighbors[side],
                             neighbors[side],
                             mpi_communicator,
                             &mpi_status);

                    // Compute Ap and with it compute alpha
                    for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
                      {
                        Ap[side][i] = -(interface_data_send[side][i] +
                                        interface_data_receive[side][i]);


                      }

                    q[side].resize(Ap[side].size(),0);
                    assert(Ap[side].size()==Q_side[side][k_counter].size());
                    q[side] = Ap[side];
                    for(unsigned int i=0; i<=k_counter; ++i){
                             	for(unsigned int j=0; j<q[side].size();++j){
                             		h[i]+=q[side][j]*Q_side[side][i][j];
                             		}

                  }


                  } //////-----------end of loop over side, q is calculated as AQ[][k] and ARnoldi Algorithm continued-------------------------------



              //Arnoldi Algorithm continued
                    //combining summing h[i] over all subdomains
                    std::vector<double> h_buffer(k_counter+2,0);

                	MPI_Allreduce(&h[0],
                			&h_buffer[0],
    						k_counter+2,
    						MPI_DOUBLE,
    						MPI_SUM,
    						mpi_communicator);

                	h=h_buffer;
                	for (unsigned int side = 0; side < n_faces_per_cell; ++side)
                		if (neighbors[side] >= 0)
                			for(unsigned int i=0; i<=k_counter; ++i)
                				for(unsigned int j=0; j<q[side].size();++j){
                					q[side][j]-=h[i]*Q_side[side][i][j];
                				}//end first loop for arnolod algorithm
                	double h_dummy = 0;

                	//calculating h(k+1)=norm(q) as summation over side,subdomains norm_squared(q[side])
                	for (unsigned int side = 0; side < n_faces_per_cell; ++side)
                	            		if (neighbors[side] >= 0)
                	            			h_dummy+=vect_norm(q[side])*vect_norm(q[side]);
                	double h_k_buffer=0;

                	MPI_Allreduce(&h_dummy,
                            			&h_k_buffer,
                						1,
                						MPI_DOUBLE,
                						MPI_SUM,
                						mpi_communicator);
                	h[k_counter+1]=sqrt(h_k_buffer);


                	for (unsigned int side = 0; side < n_faces_per_cell; ++side)
                		if (neighbors[side] >= 0){
                			for(unsigned int i=0;i<q[side].size();++i)
                				q[side][i]/=h[k_counter+1];
                		//Pushing back q[side] h to Q  as Q(k+1)
//                		Q_side[side].push_back(q[side]);
                			Q_side[side][k_counter+1]=q[side];
                		}

                		//Pushing back  h to H as H(k)}
//                	H.push_back(h);
                	H[k_counter]=h;
               //---end of Arnoldi Algorithm

              //Eliminating the last element in H ith row and updating the rotation matrix.
              apply_givens_rotation(H[k_counter],cs,sn,
              						k_counter);
              //Updating the residual vector
//              Beta.push_back(-sn[k_counter]*Beta[k_counter]);
              Beta[k_counter+1]=-sn[k_counter]*Beta[k_counter];
              Beta[k_counter]*=cs[k_counter];

              //Combining error at kth iteration
              combined_error_iter=fabs(Beta[k_counter+1])/r_norm;


              //saving the combined error at each iteration
              e_all_iter[k_counter+1]=(combined_error_iter);





//              pcout << "\r  ..." << cg_iteration
//                    << " iterations completed, (residual = " << combined_error_iter
//                    << ")..." << std::flush;
              // Exit criterion
              if (combined_error_iter/e_all_iter[0] < tolerance)
                {
                  pcout << "\n  GMRES converges in " << cg_iteration << " iterations!\n and residual is"<<combined_error_iter/e_all_iter[0]<<"\n";
                  Alambda_guess = Ap;
                  lambda_guess = lambda;
                  break;
                }
              else if(k_counter>maxiter-2)
            	  pcout << "\n  GMRES doesn't converge after  " << k_counter << " iterations!\n";



              //maxing interface_data_receive and send zero so it can be used is solving for Ap(or A*Q([k_counter]).
              for (unsigned int side = 0; side < n_faces_per_cell; ++side)
                {

                  interface_data_receive[side].resize(interface_dofs[side].size(), 0);
                  interface_data_send[side].resize(interface_dofs[side].size(), 0);


                }
              Ap.resize(n_faces_per_cell);
              k_counter++;
            }//end of the while loop(k_counter<max iteration)

          //Calculating the final result from H ,Q_side and Beta
          //Finding y which has size k_counter using back sove function
          std::vector<double> y(k_counter+1,0);
//          assert(Beta.size()==k_counter+2);
          back_solve(H,Beta,y,k_counter);

          //updating X(lambda) to get the final lambda value before solving the final star problem
          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
                  if (neighbors[side] >= 0)
                      for (unsigned int i = 0; i < interface_data[side].size(); ++i)
                         for(unsigned int j=0; j<=k_counter; ++j)
                          lambda[side][i] += Q_side[side][j][i]*y[j];
          //we can replace lambda here and just add interface_data(skip one step below)


          if (mortar_flag)
                 {
                     interface_data = lambda;
                     for (unsigned int side=0;side<n_faces_per_cell;++side)
                         for (unsigned int i=0;i<interface_dofs[side].size();++i)
                             interface_fe_function_mortar[interface_dofs[side][i]] = interface_data[side][i];

                     project_mortar(P_coarse2fine,
                                    dof_handler_mortar,
                                    interface_fe_function_mortar,
                                    quad,
                                    constraints,
                                    neighbors,
                                    dof_handler,
                                    interface_fe_function);
                     interface_fe_function.block(2) = 0;
                 }
                 else
                 {
                     interface_data = lambda;
                     for (unsigned int side=0; side<n_faces_per_cell; ++side)
                         for (unsigned int i=0; i<interface_dofs[side].size(); ++i)
                             interface_fe_function[interface_dofs[side][i]] = interface_data[side][i];
                 }


          assemble_rhs_star(fe_face_values);
          solve_star();
          solution.reinit(solution_bar);
          solution = solution_bar;
          solution.sadd(1.0, solution_star);

          solution_star.sadd(1.0, solution_bar);
          pcout<<"finished local_gmres"<<"\n";


        }






    // TODO: allow for nonzero initial lambda, use (k-1)-st lambda for k-th guess
    template <int dim>
    void
	MixedBiotProblemDD<dim>::local_cg(const unsigned int maxiter, unsigned int split_order_flag)
    {
//    	if(split_order_flag==0)
//    		TimerOutput::Scope t(computing_timer, "Local CG Elast");
//    	else if(split_order_flag==1)
//    		TimerOutput::Scope t(computing_timer, "Local CG Darcy");
    	TimerOutput::Scope t(computing_timer, "Local CG ");


      const unsigned int this_mpi =
        Utilities::MPI::this_mpi_process(mpi_communicator);
      const unsigned int n_processes =
        Utilities::MPI::n_mpi_processes(mpi_communicator);
      const unsigned int n_faces_per_cell = GeometryInfo<dim>::faces_per_cell;

      std::vector<std::vector<double>> interface_data_receive(n_faces_per_cell);
      std::vector<std::vector<double>> interface_data_send(n_faces_per_cell);
      std::vector<std::vector<double>> interface_data(n_faces_per_cell);
      std::vector<std::vector<double>> lambda(n_faces_per_cell);

      for (unsigned int side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
          {
        	if(split_order_flag==0){
                interface_data_receive[side].resize(interface_dofs_elast[side].size(),0);
                interface_data_send[side].resize(interface_dofs_elast[side].size(), 0);
                interface_data[side].resize(interface_dofs_elast[side].size(), 0);
        	}
        	else if(split_order_flag==1){
        		interface_data_receive[side].resize(interface_dofs_darcy[side].size(),0);
        		interface_data_send[side].resize(interface_dofs_darcy[side].size(), 0);
        		interface_data[side].resize(interface_dofs_darcy[side].size(), 0);

        	}


          }

      // Extra for projections from mortar to fine grid and RHS assembly
      Quadrature<dim - 1> quad;
      quad = QGauss<dim - 1>(qdegree);


      ConstraintMatrix  constraints;
      constraints.clear();
      constraints.close();
      FEFaceValues<dim> fe_face_values(fe,
                                       quad,
                                       update_values | update_normal_vectors |
                                         update_quadrature_points |
                                         update_JxW_values);

      // CG structures and parameters
      std::vector<double> alpha_side(n_faces_per_cell, 0),
        alpha_side_d(n_faces_per_cell, 0), beta_side(n_faces_per_cell, 0),
        beta_side_d(n_faces_per_cell, 0);
      std::vector<double> alpha(2, 0), beta(2, 0);

      std::vector<std::vector<double>> r(n_faces_per_cell), p(n_faces_per_cell);
      std::vector<std::vector<double>> Ap(n_faces_per_cell);

      if(split_order_flag==0)
    	  solve_bar_elast();
      else if (split_order_flag==1)
    	  solve_bar_darcy();
      interface_fe_function.reinit(solution);


      double l0 = 0.0;
      // CG with rhs being 0 and initial guess lambda = 0
      for (unsigned side = 0; side < n_faces_per_cell; ++side)
        if (neighbors[side] >= 0)
          {

            // Something will be here to initialize lambda correctly, right now it
            // is just zero
//        	std::vector<double> r_receive_buffer;
        	if(split_order_flag==0)
        	{
        		if (true || prm.time == prm.time_step)
        		              {
        		                Ap[side].resize(interface_dofs_elast[side].size(), 0);
        		                lambda[side].resize(interface_dofs_elast[side].size(), 0);
        		              }
        		              else
        		              {
        		                Ap = Alambda_guess_elast;
        		                lambda = lambda_guess_elast;
        		              }
            r[side].resize(interface_dofs_elast[side].size(), 0);


            // Right now it is effectively solution_bar - A\lambda (0)
              for (unsigned int i = 0; i < interface_dofs_elast[side].size(); ++i){
                r[side][i] = get_normal_direction(side) *
                               solution_bar_elast[interface_dofs_elast[side][i]] -
                             get_normal_direction(side) * solution_star_elast[interface_dofs_elast[side][i]];
              }


        	}
        	 else if(split_order_flag==1)
        	 {
        		  if (true || prm.time == prm.time_step)
        		              {
        		                Ap[side].resize(interface_dofs_darcy[side].size(), 0);
        		                lambda[side].resize(interface_dofs_darcy[side].size(), 0);
        		              }
        		              else
        		              {
        		                Ap = Alambda_guess_darcy;
        		                lambda = lambda_guess_darcy;
        		              }

                 r[side].resize(interface_dofs_darcy[side].size(), 0);

                 // Right now it is effectively solution_bar - A\lambda (0)
                   for (unsigned int i = 0; i < interface_dofs_darcy[side].size(); ++i)
                     r[side][i] = get_normal_direction(side) *
                                    solution_bar_darcy[interface_dofs_darcy[side][i]-n_Elast] -
                                  get_normal_direction(side) *  solution_star_darcy[interface_dofs_darcy[side][i]-n_Elast];

             	}
        	std::vector<double> r_receive_buffer(r[side].size(),0);

        	  MPI_Send(&r[side][0],
        	           r[side].size(),
        				MPI_DOUBLE,
        				neighbors[side],
        				this_mpi,
        				mpi_communicator);
        	  MPI_Recv(&r_receive_buffer[0],
        	            r_receive_buffer.size(),
        				MPI_DOUBLE, neighbors[side],
        				neighbors[side],
        				mpi_communicator,
        				&mpi_status);
            for (unsigned int i = 0; i < r[side].size(); ++i)
              {
                r[side][i] += r_receive_buffer[i];
              }
          }
      p = r;

      double normB    = 0;
      double normRold = 0;

      unsigned int iteration_counter = 0;
      while (iteration_counter < maxiter)
        {
          alpha[0] = 0.0;
          alpha[1] = 0.0;
          beta[0]  = 0.0;
          beta[1]  = 0.0;

          iteration_counter++;
          interface_data = p;

              for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            	  if(split_order_flag==0){
            		  for (unsigned int i = 0; i < interface_dofs_elast[side].size(); ++i)
            			  interface_fe_function[interface_dofs_elast[side][i]] = interface_data[side][i];
            	  }
            	  else if(split_order_flag==1){
            		  for (unsigned int i = 0; i < interface_dofs_darcy[side].size(); ++i)
            		              			  interface_fe_function[interface_dofs_darcy[side][i]] = interface_data[side][i];
            	  }


              if(split_order_flag==0){
            	  interface_fe_function.block(2) = 0;
            	  assemble_rhs_star_elast(fe_face_values);
            	  solve_star_elast();
              }
              else if(split_order_flag==1){
                       	  assemble_rhs_star_darcy(fe_face_values);
                       	  solve_star_darcy();
              }
//              solve_star();


          cg_iteration++;


          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
              {
                alpha_side[side]   = 0;
                alpha_side_d[side] = 0;
                beta_side[side]    = 0;
                beta_side_d[side]  = 0;

                // Create vector of u\dot n to send
                if(split_order_flag==0){
                  for (unsigned int i = 0; i < interface_dofs_elast[side].size(); ++i)
                    interface_data_send[side][i] =
                      get_normal_direction(side) *
                      solution_star_elast[interface_dofs_elast[side][i]];
                }
                else if(split_order_flag==1){
                                  for (unsigned int i = 0; i < interface_dofs_darcy[side].size(); ++i)
                                    interface_data_send[side][i] =
                                      get_normal_direction(side) *
                                      solution_star_darcy[interface_dofs_darcy[side][i]-n_Elast];
                                }
                unsigned int interface_tmp_size;
                if(split_order_flag==0)
                	 interface_tmp_size= interface_dofs_elast[side].size();
                else if(split_order_flag==1)
                		interface_tmp_size = interface_dofs_darcy[side].size();
                MPI_Send(&interface_data_send[side][0],
                         interface_tmp_size,
                         MPI_DOUBLE,
                         neighbors[side],
                         this_mpi,
                         mpi_communicator);
                MPI_Recv(&interface_data_receive[side][0],
                         interface_tmp_size,
                         MPI_DOUBLE,
                         neighbors[side],
                         neighbors[side],
                         mpi_communicator,
                         &mpi_status);

                // Compute Ap and with it compute alpha

                for (unsigned int i = 0; i < interface_data_send[side].size(); ++i)
                  {
                    Ap[side][i] = -(interface_data_send[side][i] +
                                    interface_data_receive[side][i]);

                    alpha_side[side] += r[side][i] * r[side][i];
                    alpha_side_d[side] += p[side][i] * Ap[side][i];
                  }
              }

          // Fancy some lambdas, huh?
          std::for_each(alpha_side.begin(), alpha_side.end(), [&](double n) {
            alpha[0] += n;
          });
          std::for_each(alpha_side_d.begin(), alpha_side_d.end(), [&](double n) {
            alpha[1] += n;
          });
          std::vector<double> alpha_buffer(2, 0);

          MPI_Allreduce(&alpha[0],
                        &alpha_buffer[0],
                        2,
                        MPI_DOUBLE,
                        MPI_SUM,
                        mpi_communicator);

          alpha = alpha_buffer;

          if (cg_iteration == 1){
            normB = alpha[0];
            normRold = alpha[0];
          }

          normRold = alpha[0];

          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            if (neighbors[side] >= 0)
              {
                for (unsigned int i = 0; i < interface_data[side].size(); ++i)
                  {
                    lambda[side][i] += (alpha[0] * p[side][i]) / alpha[1];
                    r[side][i] -= (alpha[0] * Ap[side][i]) / alpha[1];
                  }

                for (unsigned int i = 0; i < interface_data[side].size(); ++i)
                  beta_side[side] += r[side][i] * r[side][i];
              }
          if(split_order_flag==0){
//          pcout << "\r  ..." << cg_iteration
//                << " Elast iterations completed, (Elast residual = " << fabs(alpha[0])
//                << ")..." << std::flush;
          // Exit criterion
          if (fabs(alpha[0]) / normB < tolerance )
//          if (sqrt(alpha[0]/normB<1.e-8) )
            {
              pcout << "\n  Elast CG converges in " << cg_iteration << " iterations!\n";
              Alambda_guess_elast = Ap;
              lambda_guess_elast = lambda;
              break;
            }
          }
          else if(split_order_flag==1){
//                   pcout << "\r  ..." << cg_iteration
//                         << " Darcy iterations completed, (Darcy residual = " << fabs(alpha[0])
//                         << ")..." << std::flush;
                   // Exit criterion
                   if (fabs(alpha[0]) / normB < tolerance )
//        	  if (sqrt(alpha[0]/normB<1.e-8) )
                     {
                       pcout << "\n  Darcy CG converges in " << cg_iteration << " iterations!\n";
                       Alambda_guess_darcy = Ap;
                       lambda_guess_darcy = lambda;
                       break;
                     }
                   }
//          normRold = alpha[0];

          std::for_each(beta_side.begin(), beta_side.end(), [&](double n) {
            beta[0] += n;
          });
          double beta_buffer = 0;

          MPI_Allreduce(
            &beta[0], &beta_buffer, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

          beta[0] = beta_buffer;
          beta[1] = normRold;

          for (unsigned int side = 0; side < n_faces_per_cell; ++side)
            {
              if (neighbors[side] >= 0)
                for (unsigned int i = 0; i < interface_data[side].size(); ++i)
                  p[side][i] = r[side][i] + (beta[0] / beta[1]) * p[side][i];

              if(split_order_flag==0){
              interface_data_receive[side].resize(interface_dofs_elast[side].size(), 0);
              interface_data_send[side].resize(interface_dofs_elast[side].size(), 0);
              }
              if(split_order_flag==1){
                            interface_data_receive[side].resize(interface_dofs_darcy[side].size(), 0);
                            interface_data_send[side].resize(interface_dofs_darcy[side].size(), 0);
                            }


            }
          Ap.resize(n_faces_per_cell);
        }


          interface_data = lambda;
          //
          if(split_order_flag==0)
          {
			  for (unsigned int side = 0; side < n_faces_per_cell; ++side)
				for (unsigned int i = 0; i < interface_dofs_elast[side].size(); ++i)
				  interface_fe_function[interface_dofs_elast[side][i]] =
					interface_data[side][i];


		  assemble_rhs_star_elast(fe_face_values);
		  solve_star_elast();
          }
          else if(split_order_flag==1)
                {
      			  for (unsigned int side = 0; side < n_faces_per_cell; ++side)
      				for (unsigned int i = 0; i < interface_dofs_darcy[side].size(); ++i)
      				  interface_fe_function[interface_dofs_darcy[side][i]] =
      					interface_data[side][i];


      		  assemble_rhs_star_darcy(fe_face_values);
      		  solve_star_darcy();
                }
      if(split_order_flag==0){
    	  solution_elast.reinit(solution_bar_elast);
    	  solution_elast = solution_bar_elast;
    	  solution_elast.sadd(1.0, solution_star_elast);
    	  solution_star_elast.sadd(1.0, solution_bar_elast);
    	  solution.block(0)=solution_elast.block(0);
    	  solution.block(1)=solution_elast.block(1);
    	  solution.block(2)=solution_elast.block(2);

      }
      else if(split_order_flag==1){
         	  solution_darcy.reinit(solution_bar_darcy);
         	  solution_darcy = solution_bar_darcy;
         	  solution_darcy.sadd(1.0, solution_star_darcy);
         	  solution_star_darcy.sadd(1.0, solution_bar_darcy);
         	 solution.block(3)=solution_darcy.block(0);
         	 solution.block(4)=solution_darcy.block(1);

           }

 }





    // MixedBiotProblemDD::compute_interface_error
    template <int dim>
    std::vector<double> MixedBiotProblemDD<dim>::compute_interface_error()
    {
        system_rhs_star = 0;
        std::vector<double> return_vector(2,0);

        QGauss<dim-1> quad (qdegree);
        QGauss<dim-1> project_quad (qdegree);
        FEFaceValues<dim> fe_face_values (fe, quad,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);

        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int dofs_per_cell_mortar = fe_mortar.dofs_per_cell;

        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
        const FEValuesExtractors::Vector velocity (dim*dim + dim + 0.5*dim*(dim-1));
//        const FEValuesExtractors::Scalar pressure(dim*dim + dim + 0.5*dim*(dim-1)+dim);

        DisplacementBoundaryValues<dim> displacement_boundary_values;
        PressureBoundaryValues<dim>     pressure_boundary_values;

        displacement_boundary_values.set_time(prm.time);
        pressure_boundary_values.set_time(prm.time);

        for (unsigned int d=0; d<dim; ++d)
        {
            const FEValuesExtractors::Vector tmp_stress(d*dim);
            stresses[d].first_vector_component = tmp_stress.first_vector_component;

        }

        std::vector<std::vector<Tensor<1, dim>>> interface_values_mech(dim, std::vector<Tensor<1, dim>> (n_face_q_points));
        std::vector<Tensor<1, dim>> interface_values_flux(n_face_q_points);
        std::vector<std::vector<Tensor<1, dim>>> solution_values_mech(dim, std::vector<Tensor<1, dim>> (n_face_q_points));
        std::vector<Tensor<1, dim>> solution_values_flow(n_face_q_points);
        std::vector<Vector<double>> displacement_values (n_face_q_points, Vector<double> (dim));
        std::vector<double> pressure_values(n_face_q_points);
//        Vector<double> pressure_values(n_face_q_points);

        // Assemble rhs for star problem with data = u - lambda_H on interfaces
        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
        for (;cell!=endc;++cell)
        {
            local_rhs = 0;
            cell->get_dof_indices (local_dof_indices);

            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() != 0)
                {
                    fe_face_values.reinit (cell, face_n);

                    for (unsigned int d_i=0; d_i<dim; ++d_i)
                        fe_face_values[stresses[d_i]].get_function_values (interface_fe_function, interface_values_mech[d_i]);

                    fe_face_values[velocity].get_function_values (interface_fe_function, interface_values_flux);

                    displacement_boundary_values.vector_value_list(fe_face_values.get_quadrature_points(),
                                                     displacement_values);
                    pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), pressure_values);
//                    Vector<double >pressure_values(boundary_values_flow.begin(),boundary_values_flow.end());

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {

                        	 local_rhs(i) += (fe_face_values[velocity].value (i, q) *
                        	                 fe_face_values.normal_vector(q) *
                        	                 (interface_values_flux[q] * get_normal_direction(cell->face(face_n)->boundary_id()-1) *
                        	                 fe_face_values.normal_vector(q) - pressure_values[q])) *
                        	                 fe_face_values.JxW(q);


                            for (unsigned int d_i=0; d_i<dim; ++d_i)
                                local_rhs(i) += fe_face_values[stresses[d_i]].value (i, q) *
                                                fe_face_values.normal_vector(q) *
                                                (displacement_values[q][d_i] - interface_values_mech[d_i][q] * get_normal_direction(cell->face(face_n)->boundary_id()-1) *
                                                                               fe_face_values.normal_vector(q)) *
                                                fe_face_values.JxW(q);
                        }
                }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
                system_rhs_star(local_dof_indices[i]) += local_rhs(i);

        }

        // Solve star problem with data given by (u,p) - (lambda_u,lambda_p).
        solve_star();

        // Project the solution to the mortar space
        ConstraintMatrix constraints;
        constraints.clear();
        constraints.close();
        project_mortar(P_fine2coarse, dof_handler, solution_star, project_quad, constraints, neighbors, dof_handler_mortar, solution_star_mortar);

//        double res = 0;

        FEFaceValues<dim> fe_face_values_mortar (fe_mortar, quad,
                                                 update_values    | update_normal_vectors |
                                                 update_quadrature_points  | update_JxW_values);

        // Compute the discrete interface norm
        cell = dof_handler_mortar.begin_active(),
                endc = dof_handler_mortar.end();
        for (;cell!=endc;++cell)
        {
            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() != 0)
                {
                    fe_face_values_mortar.reinit (cell, face_n);

                    for (unsigned int d_i=0; d_i<dim; ++d_i)
                    {
                        fe_face_values_mortar[stresses[d_i]].get_function_values (solution_star_mortar, solution_values_mech[d_i]);
                        fe_face_values_mortar[stresses[d_i]].get_function_values (interface_fe_function_mortar, interface_values_mech[d_i]);
                    }

                    fe_face_values_mortar[velocity].get_function_values(solution_star_mortar,solution_values_flow);
                    fe_face_values_mortar[velocity].get_function_values (interface_fe_function_mortar, interface_values_flux);

                    displacement_boundary_values.vector_value_list(fe_face_values_mortar.get_quadrature_points(),
                                                     displacement_values);
                    pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), pressure_values);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                        for (unsigned int d_i=0; d_i<dim; ++d_i)
                            return_vector[0] += fabs(fe_face_values_mortar.normal_vector(q) * solution_values_mech[d_i][q] *
                                        (displacement_values[q][d_i] - fe_face_values_mortar.normal_vector(q) * interface_values_mech[d_i][q] * get_normal_direction(cell->face(face_n)->boundary_id()-1)) *
                                        fe_face_values_mortar.JxW(q));
                        return_vector[1] += fabs(fe_face_values_mortar.normal_vector(q) * solution_values_flow[q] *
                                                                (pressure_values[q] - fe_face_values_mortar.normal_vector(q) * interface_values_flux[q] * get_normal_direction(cell->face(face_n)->boundary_id()-1)) *
                                                                fe_face_values_mortar.JxW(q));
                    }
                }
        }
//        std::vector<double> return_vector(1,sqrt(res));
//        return sqrt(res);
//        return_vector[1]+= return_vector[0];
        return_vector[0]=sqrt(return_vector[0]);
        return_vector[1]=sqrt(return_vector[1]);
        return return_vector;
    }


    // MixedBiotProblemDD::compute_errors
    template <int dim>
    void MixedBiotProblemDD<dim>::compute_errors (const unsigned int cycle)
    {
      TimerOutput::Scope t(computing_timer, "Compute errors");

      const unsigned int total_dim = static_cast<unsigned int>(dim*dim + dim + 0.5*dim*(dim-1) + dim + 1);

      const ComponentSelectFunction<dim> stress_mask(std::make_pair(0,dim*dim), total_dim);
      const ComponentSelectFunction<dim> displacement_mask(std::make_pair(dim*dim,dim*dim+dim), total_dim);
      const ComponentSelectFunction<dim> rotation_mask(std::make_pair(dim*dim+dim,dim*dim+dim+0.5*dim*(dim-1)), total_dim);

      const ComponentSelectFunction<dim> velocity_mask(std::make_pair(dim*dim+dim+0.5*dim*(dim-1), dim*dim+dim+0.5*dim*(dim-1)+dim), total_dim);
      const ComponentSelectFunction<dim> pressure_mask(static_cast<unsigned int>(dim*dim+dim+0.5*dim*(dim-1)+dim), total_dim);

      ExactSolution<dim> exact_solution;
      exact_solution.set_time(prm.time);

      // Vectors to temporarily store cellwise errros
      Vector<double> cellwise_errors (triangulation.n_active_cells());
      Vector<double> cellwise_norms (triangulation.n_active_cells());

      // Vectors to temporarily store cellwise componentwise div errors
      Vector<double> cellwise_div_errors (triangulation.n_active_cells());
      Vector<double> cellwise_div_norms (triangulation.n_active_cells());

      // Define quadrature points to compute errors at
      QTrapez<1>      q_trapez;
      QIterated<dim>  quadrature(q_trapez,degree+2);
      QGauss<dim>  quadrature_div(5);

      // This is used to show superconvergence at midcells
      QGauss<dim>   quadrature_super(1);

      // Since we want to compute the relative norm
      BlockVector<double> zerozeros(1, solution.size());

      // Pressure error and norm
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature,
                                         VectorTools::L2_norm,
                                         &pressure_mask);
      const double p_l2_error = cellwise_errors.norm_sqr();

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature,
                                         VectorTools::L2_norm,
                                         &pressure_mask);
      const double p_l2_norm = cellwise_norms.norm_sqr();

      // L2 in time error
      err.l2_l2_errors[1] += p_l2_error;
      err.l2_l2_norms[1] += p_l2_norm;

      // Linf in time error
      err.linf_l2_errors[1] = std::max(err.linf_l2_errors[1], sqrt(p_l2_error)/sqrt(p_l2_norm));
      //linf_l2_norms[1] = std::max(linf_l2_norms[1], p_l2_norm*p_l2_norm);

      // Pressure error and norm at midcells
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature_super,
                                         VectorTools::L2_norm,
                                         &pressure_mask);
      const double p_l2_mid_error = cellwise_errors.norm_sqr();

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature_super,
                                         VectorTools::L2_norm,
                                         &pressure_mask);
      const double p_l2_mid_norm = cellwise_norms.norm_sqr();

      // L2 in time error
      err.pressure_disp_l2_midcell_errors[0] +=p_l2_mid_error;
      err.pressure_disp_l2_midcell_norms[0] += p_l2_mid_norm;

      // Velocity L2 error and norm
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature,
                                         VectorTools::L2_norm,
                                         &velocity_mask);
      double u_l2_error = cellwise_errors.norm_sqr();

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature,
                                         VectorTools::L2_norm,
                                         &velocity_mask);

      double u_l2_norm = cellwise_norms.norm_sqr();

      // following is actually calculating H_div norm for velocity
//      err.l2_l2_errors[0] +=u_l2_error;
      err.l2_l2_norms[0] += u_l2_norm;
      double total_time = prm.time_step * prm.num_time_steps;
      {
        // Velocity Hdiv error and seminorm
        VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                           cellwise_errors, quadrature,
                                           VectorTools::Hdiv_seminorm,
                                           &velocity_mask);
        const double u_hd_error = cellwise_errors.norm_sqr();

        VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                           cellwise_norms, quadrature,
                                           VectorTools::Hdiv_seminorm,
                                           &velocity_mask);
        const double u_hd_norm = cellwise_norms.norm_sqr();

        //std::cout << u_hd_error << std::endl;

        // L2 in time error
        //if (std::fabs(time-5*time_step) > 1.0e-12) {
        err.velocity_stress_l2_div_errors[0] += u_hd_error;
        err.velocity_stress_l2_div_norms[0] += u_hd_norm;     // put += back!
        //}
        u_l2_error+=u_hd_error;
		u_l2_norm+=u_hd_norm;
      }
      err.l2_l2_errors[0] = std::max(err.l2_l2_errors[0],sqrt(u_l2_error)/sqrt(u_l2_norm));
//      err.linf_l2_errors[1] = std::max(err.linf_l2_errors[1], sqrt(p_l2_error)/sqrt(p_l2_norm));
      // Rotation error and norm
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature,
                                         VectorTools::L2_norm,
                                         &rotation_mask);
      const double r_l2_error = cellwise_errors.norm_sqr();

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature,
                                         VectorTools::L2_norm,
                                         &rotation_mask);
      const double r_l2_norm = cellwise_norms.norm_sqr();

      err.l2_l2_errors[4] += r_l2_error;
      err.l2_l2_norms[4] += r_l2_norm;

      // Displacement error and norm
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature,
                                         VectorTools::L2_norm,
                                         &displacement_mask);
      const double d_l2_error = cellwise_errors.norm_sqr();

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature,
                                         VectorTools::L2_norm,
                                         &displacement_mask);
      const double d_l2_norm = cellwise_norms.norm_sqr();

//      err.l2_l2_errors[3] += d_l2_error;
      err.l2_l2_errors[3] = std::max(err.l2_l2_errors[3], sqrt(d_l2_error)/sqrt(d_l2_norm));
      err.l2_l2_norms[3] += d_l2_norm;

      // Displacement error and norm at midcells
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature_super,
                                         VectorTools::L2_norm,
                                         &displacement_mask);
      const double d_l2_mid_error = cellwise_errors.norm_sqr();

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature_super,
                                         VectorTools::L2_norm,
                                         &displacement_mask);
      const double d_l2_mid_norm = cellwise_norms.norm_sqr();

      // L2 in time error
      err.pressure_disp_l2_midcell_errors[1] += d_l2_mid_error;
      err.pressure_disp_l2_midcell_norms[1] += d_l2_mid_norm;

      // Stress L2 error and norm
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature,
                                         VectorTools::L2_norm,
                                         &stress_mask);
      double s_l2_error = cellwise_errors.norm_sqr();

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature,
                                         VectorTools::L2_norm,
                                         &stress_mask);

      double s_l2_norm = cellwise_norms.norm_sqr();

      // Linf in time error
//      err.linf_l2_errors[2] = std::max(err.linf_l2_errors[2],sqrt(s_l2_error)/sqrt(s_l2_norm));
      //linf_l2_norms[2] = std::max(linf_l2_norms[2],s_l2_norm*s_l2_norm);

      err.l2_l2_errors[2] += s_l2_error;
      err.l2_l2_norms[2] += s_l2_norm;

      // Stress Hdiv seminorm
      cellwise_errors = 0;
      cellwise_norms = 0;

      double s_hd_error = 0;
      double s_hd_norm = 0;

      for (int i=0; i<dim; ++i){
        const ComponentSelectFunction<dim> stress_component_mask (std::make_pair(i*dim,(i+1)*dim), total_dim);

        VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                           cellwise_div_errors, quadrature,
                                           VectorTools::Hdiv_seminorm,
                                           &stress_component_mask);
        s_hd_error += cellwise_div_errors.norm_sqr();

        VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                           cellwise_div_norms, quadrature,
                                           VectorTools::Hdiv_seminorm,
                                           &stress_component_mask);
        s_hd_norm += cellwise_div_norms.norm_sqr();
      }
      s_l2_error+= s_hd_error;
      s_l2_norm+= s_hd_norm;
      err.linf_l2_errors[2] = std::max(err.linf_l2_errors[2],sqrt(s_l2_error)/sqrt(s_l2_norm));

//    s_hd_error = sqrt(s_hd_error);
//    s_hd_norm = sqrt(s_hd_norm);

//      std::cout << "Component function test: " << std::endl;
//      Vector<double> tmp(MixedBiotProblem::total_dim);
//      displacement_mask.vector_value(Point<dim>(), tmp);
//      std::cout << tmp << std::endl;

      err.velocity_stress_l2_div_errors[1] += s_hd_error;
      err.velocity_stress_l2_div_norms[1] += s_hd_norm;     // put += back!

      double l_int_error_elast=1, l_int_norm_elast=1;
      double l_int_error_darcy=1, l_int_norm_darcy=1;
//      double l_int_error=1, l_int_norm=1;
        if (mortar_flag)
        {
            DisplacementBoundaryValues<dim> displ_solution;
            displ_solution.set_time(prm.time);
            std::vector<double> tmp_err_vect(2,0);
            tmp_err_vect = compute_interface_error();
//            l_int_error_elast = compute_interface_error();
            l_int_error_elast =tmp_err_vect[0];
            l_int_error_darcy =tmp_err_vect[1];
//            if(split_flag==0){
//            	l_int_error=pow(l_int_error_elast,2)+pow(l_int_error_darcy,2);
//            	l_int_error= sqrt(l_int_error);
//            }

            interface_fe_function = 0;
            interface_fe_function_mortar = 0;
            tmp_err_vect = compute_interface_error();
//            l_int_norm_elast = compute_interface_error();
            l_int_norm_elast = tmp_err_vect[0];
            l_int_norm_darcy = tmp_err_vect[1];
//            if(split_flag==0){
//                l_int_norm=pow(l_int_norm_elast,2)+pow(l_int_norm_darcy,2);
//                l_int_norm= sqrt(l_int_norm);
//            }
        }


      // On the last time step compute actual errors
      if(std::fabs(prm.time-total_time) < 1.0e-12)
      {
        // Assemble convergence table
        const unsigned int n_active_cells=triangulation.n_active_cells();
        const unsigned int n_dofs=dof_handler.n_dofs();

        double send_buf_num[13] = {err.l2_l2_errors[0],
                                   err.velocity_stress_l2_div_errors[0],
                                   err.l2_l2_errors[1],
                                   err.pressure_disp_l2_midcell_errors[0],
                                   err.linf_l2_errors[1],
                                   err.l2_l2_errors[2],
                                   err.velocity_stress_l2_div_errors[1],
                                   err.linf_l2_errors[2],
                                   err.l2_l2_errors[3],
                                   err.pressure_disp_l2_midcell_errors[1],
                                   err.l2_l2_errors[4],
								   l_int_error_elast,
								   l_int_error_darcy};

        double send_buf_den[13] = {err.l2_l2_norms[0],
                                   err.velocity_stress_l2_div_norms[0],
                                   err.l2_l2_norms[1],
                                   err.pressure_disp_l2_midcell_norms[0],
                                   0,
                                   err.l2_l2_norms[2],
                                   err.velocity_stress_l2_div_norms[1],
                                   0,
                                   err.l2_l2_norms[3],
                                   err.pressure_disp_l2_midcell_norms[1],
                                   err.l2_l2_norms[4],
								   l_int_norm_elast,
								   l_int_norm_darcy};

        double recv_buf_num[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};
        double recv_buf_den[13] = {0,0,0,0,0,0,0,0,0,0,0,0,0};

        MPI_Reduce(&send_buf_num[0], &recv_buf_num[0], 13, MPI_DOUBLE, MPI_SUM, 0, mpi_communicator);
        MPI_Reduce(&send_buf_den[0], &recv_buf_den[0], 13, MPI_DOUBLE, MPI_SUM, 0, mpi_communicator);

        for (unsigned int i=0; i<11; ++i)
          if (i != 4 && i != 7 && i != 0 && i != 8)
            recv_buf_num[i] = sqrt(recv_buf_num[i])/sqrt(recv_buf_den[i]);
//          else
//            recv_buf_num[i] = recv_buf_num[i];
 //    Calculating the relative error in mortar displacement.
//        recv_buf_num[11] = recv_buf_num[11]/recv_buf_den[11];
//        recv_buf_num[12] = recv_buf_num[12]/recv_buf_den[12];

        convergence_table.add_value("cycle", cycle);
        if(split_flag==0)
        convergence_table.add_value("# GMRES", max_cg_iteration);
        else if(split_flag!=0){
        	convergence_table.add_value("# CG_Elast",max_cg_iteration);
        	convergence_table.add_value("# CG_Darcy",max_cg_iteration_darcy);
        }

        convergence_table.add_value("Velocity,L8-Hdiv", recv_buf_num[0]);
//        convergence_table.add_value("Velocity,L2-Hdiv", recv_buf_num[1]);

//        convergence_table.add_value("Pressure,L2-L2", recv_buf_num[2]);
//        convergence_table.add_value("Pressure,L2-L2mid", recv_buf_num[3]);
        convergence_table.add_value("Pressure,L8-L2", recv_buf_num[4]);

//        convergence_table.add_value("Stress,L2-L2", recv_buf_num[5]);
//        convergence_table.add_value("Stress,L2-Hdiv", recv_buf_num[6]);
        convergence_table.add_value("Stress,L8-Hdiv", recv_buf_num[7]);

        convergence_table.add_value("Displ,L8-L2", recv_buf_num[8]);
//        convergence_table.add_value("Displ,L2-L2mid", recv_buf_num[9]);

//        convergence_table.add_value("Rotat,L2-L2", recv_buf_num[10]);
        if (mortar_flag)
        {
          convergence_table.add_value("Lambda,Elast", recv_buf_num[11]/recv_buf_den[11]);
          convergence_table.add_value("Lambda,Darcy", recv_buf_num[12]/recv_buf_den[12]);
          if(split_flag==0)
          {
        	double combined_l_int_error =(pow(recv_buf_num[11],2) + pow(recv_buf_num[12],2))/(pow(recv_buf_den[11],2) + pow(recv_buf_den[12],2));
        	combined_l_int_error = sqrt(combined_l_int_error);
        	convergence_table.add_value("Lambda,Biot", combined_l_int_error);
          }
        }
      }
    }


    // MixedBiotProblemDD::output_results
    template <int dim>
    void MixedBiotProblemDD<dim>::output_results (const unsigned int cycle, const unsigned int refine)
    {
        TimerOutput::Scope t(computing_timer, "Output results");
        unsigned int n_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
        unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator);

        /* From here disabling for longer runs:
         */

      std::vector<std::string> solution_names;
      switch(dim)
      {
        case 2:
          solution_names.push_back ("s11");
          solution_names.push_back ("s12");
          solution_names.push_back ("s21");
          solution_names.push_back ("s22");
          solution_names.push_back ("d1");
          solution_names.push_back ("d2");
          solution_names.push_back ("r");
          solution_names.push_back ("u1");
          solution_names.push_back ("u2");
          solution_names.push_back ("p");
          break;

        case 3:
          solution_names.push_back ("s11");
          solution_names.push_back ("s12");
          solution_names.push_back ("s13");
          solution_names.push_back ("s21");
          solution_names.push_back ("s22");
          solution_names.push_back ("s23");
          solution_names.push_back ("s31");
          solution_names.push_back ("s32");
          solution_names.push_back ("s33");
          solution_names.push_back ("d1");
          solution_names.push_back ("d2");
          solution_names.push_back ("d3");
          solution_names.push_back ("r1");
          solution_names.push_back ("r2");
          solution_names.push_back ("r3");
          solution_names.push_back ("u1");
          solution_names.push_back ("u2");
          solution_names.push_back ("u3");
          solution_names.push_back ("p");
          break;

        default:
        Assert(false, ExcNotImplemented());
      }

      // Components interpretation of the mechanics solution (vector^dim - vector - rotation)
      std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim*dim+dim, DataComponentInterpretation::component_is_part_of_vector);
      switch (dim)
      {
        case 2:
          data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
          break;

        case 3:
          data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
          break;

        default:
        Assert(false, ExcNotImplemented());
          break;
      }

      // Components interpretation of the flow solution (vector - scalar)
      data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

      DataOut<dim> data_out;
      data_out.attach_dof_handler (dof_handler);
      data_out.add_data_vector (solution, solution_names,
                                DataOut<dim>::type_dof_data,
                                data_component_interpretation);

      data_out.build_patches ();


      int tmp = prm.time/prm.time_step;
      std::ofstream output ("solution_d" + Utilities::to_string(dim) + "_p"+Utilities::to_string(this_mpi,4)+"-" + std::to_string(tmp)+".vtu");
      data_out.write_vtu (output);
      //following lines create a file which paraview can use to link the subdomain results
            if (this_mpi == 0)
              {
                std::vector<std::string> filenames;
                for (unsigned int i=0;
                     i<Utilities::MPI::n_mpi_processes(mpi_communicator);
                     ++i)
                  filenames.push_back ("solution_d" + Utilities::to_string(dim) + "_p"+Utilities::to_string(i,4)+"-" + std::to_string(tmp)+".vtu");

                std::ofstream master_output (("solution_d" + Utilities::to_string(dim) + "-" + std::to_string(tmp) +
                                              ".pvtu").c_str());
                data_out.write_pvtu_record (master_output, filenames);
              }

     /* end of commenting out for disabling vtu outputs*/


      double total_time = prm.time_step * prm.num_time_steps;
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && cycle == refine-1 && std::abs(prm.time-total_time)<1.0e-12){
        convergence_table.set_precision("Velocity,L8-Hdiv", 3);
//        convergence_table.set_precision("Velocity,L2-Hdiv", 3);
//        convergence_table.set_precision("Pressure,L2-L2", 3);
//        convergence_table.set_precision("Pressure,L2-L2mid", 3);
        convergence_table.set_precision("Pressure,L8-L2", 3);

        convergence_table.set_scientific("Velocity,L8-Hdiv", true);
//        convergence_table.set_scientific("Velocity,L2-Hdiv", true);
//        convergence_table.set_scientific("Pressure,L2-L2", true);
//        convergence_table.set_scientific("Pressure,L2-L2mid", true);
        convergence_table.set_scientific("Pressure,L8-L2", true);

//        convergence_table.set_precision("Stress,L2-L2", 3);
//        convergence_table.set_precision("Stress,L2-Hdiv", 3);
        convergence_table.set_precision("Stress,L8-Hdiv", 3);
        convergence_table.set_precision("Displ,L8-L2", 3);
//        convergence_table.set_precision("Displ,L2-L2mid", 3);
//        convergence_table.set_precision("Rotat,L2-L2", 3);

//        convergence_table.set_scientific("Stress,L2-L2", true);
//        convergence_table.set_scientific("Stress,L2-Hdiv", true);
        convergence_table.set_scientific("Stress,L8-Hdiv", true);
        convergence_table.set_scientific("Displ,L8-L2", true);
//        convergence_table.set_scientific("Displ,L2-L2mid", true);
//        convergence_table.set_scientific("Rotat,L2-L2", true);

//        convergence_table.set_tex_caption("# CG", "\\# cg");
        if(split_flag==0)
        	convergence_table.set_tex_caption("# GMRES", "\\# gmres");
        else if (split_flag!=0){
        	convergence_table.set_tex_caption("# CG_Elast", "\\# cg_Elast");
        	convergence_table.set_tex_caption("# CG_Darcy", "\\# cg_Darcy");
        }

        convergence_table.set_tex_caption("Velocity,L8-Hdiv", "$ \\|z - z_h\\|_{L^{\\infty}(H_{div})} $");
//        convergence_table.set_tex_caption("Velocity,L2-Hdiv", "$ \\|\\nabla\\cdot(\\u - \\u_h)\\|_{L^2(L^2)} $");
//        convergence_table.set_tex_caption("Pressure,L2-L2", "$ \\|p - p_h\\|_{L^2(L^2)} $");
//        convergence_table.set_tex_caption("Pressure,L2-L2mid", "$ \\|Qp - p_h\\|_{L^2(L^2)} $");
        convergence_table.set_tex_caption("Pressure,L8-L2", "$ \\|p - p_h\\|_{L^{\\infty}(L^2)} $");

//        convergence_table.set_tex_caption("Stress,L2-L2", "$ \\|\\sigma - \\sigma_h\\|_{L^{\\infty}(L^2)} $");
//        convergence_table.set_tex_caption("Stress,L2-Hdiv", "$ \\|\\nabla\\cdot(\\sigma - \\sigma_h)\\|_{L^{\\infty}(L^2)} $");
        convergence_table.set_tex_caption("Stress,L8-Hdiv", "$ \\|\\sigma - \\sigma_h\\|_{L^{\\infty}(H_{div})} $");
        convergence_table.set_tex_caption("Displ,L8-L2", "$ \\|u - u_h\\|_{L^{\\infty}(L^2)} $");
//        convergence_table.set_tex_caption("Displ,L2-L2mid", "$ \\|Q\\bbeta - \\bbeta_h\\|_{L^{\\infty}(L^2)} $");
//        convergence_table.set_tex_caption("Rotat,L2-L2", "$ \\|r - r_h\\|_{L^{\\infty}(L^2)} $");

//        convergence_table.evaluate_convergence_rates("# CG", ConvergenceTable::reduction_rate_log2);
        if(split_flag==0)
        	convergence_table.evaluate_convergence_rates("# GMRES", ConvergenceTable::reduction_rate_log2);
        else if (split_flag!=0){
        	convergence_table.evaluate_convergence_rates("# CG_Elast", ConvergenceTable::reduction_rate_log2);
        	convergence_table.evaluate_convergence_rates("# CG_Darcy", ConvergenceTable::reduction_rate_log2);
        }

        convergence_table.evaluate_convergence_rates("Velocity,L8-Hdiv", ConvergenceTable::reduction_rate_log2);
//        convergence_table.evaluate_convergence_rates("Velocity,L2-Hdiv", ConvergenceTable::reduction_rate_log2);
//        convergence_table.evaluate_convergence_rates("Pressure,L2-L2", ConvergenceTable::reduction_rate_log2);
//        convergence_table.evaluate_convergence_rates("Pressure,L2-L2mid", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Pressure,L8-L2", ConvergenceTable::reduction_rate_log2);

//        convergence_table.evaluate_convergence_rates("Stress,L2-L2", ConvergenceTable::reduction_rate_log2);
//        convergence_table.evaluate_convergence_rates("Stress,L2-Hdiv", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Stress,L8-Hdiv", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Displ,L8-L2", ConvergenceTable::reduction_rate_log2);
//        convergence_table.evaluate_convergence_rates("Displ,L2-L2mid", ConvergenceTable::reduction_rate_log2);
//        convergence_table.evaluate_convergence_rates("Rotat,L2-L2", ConvergenceTable::reduction_rate_log2);

        if (mortar_flag)
        {
          convergence_table.set_precision("Lambda,Elast", 3);
          convergence_table.set_scientific("Lambda,Elast", true);
          convergence_table.set_tex_caption("Lambda,Elast", "$ \\|u - \\lambda_u_H\\|_{d_H} $");
          convergence_table.evaluate_convergence_rates("Lambda,Elast", ConvergenceTable::reduction_rate_log2);

          convergence_table.set_precision("Lambda,Darcy", 3);
          convergence_table.set_scientific("Lambda,Darcy", true);
          convergence_table.set_tex_caption("Lambda,Darcy", "$ \\|p - \\lambda_p_H\\|_{d_H} $");
          convergence_table.evaluate_convergence_rates("Lambda,Darcy", ConvergenceTable::reduction_rate_log2);

          if(split_flag==0){
        	  convergence_table.set_precision("Lambda,Biot", 3);
        	  convergence_table.set_scientific("Lambda,Biot", true);
        	  convergence_table.set_tex_caption("Lambda,Biot", "$ \\|(u,p) - \\lambda_H\\|_{d_H} $");
        	  convergence_table.evaluate_convergence_rates("Lambda,Biot", ConvergenceTable::reduction_rate_log2);
          }
        }

        std::ofstream error_table_file("error" + std::to_string(Utilities::MPI::n_mpi_processes(mpi_communicator)) + "domains.tex");

        pcout << std::endl;
        convergence_table.write_text(std::cout);
        convergence_table.write_tex(error_table_file);
      }
    }

    template <int dim>
    void MixedBiotProblemDD<dim>::reset_mortars()
    {
        triangulation.clear();
        dof_handler.clear();
        convergence_table.clear();
        faces_on_interface.clear();
        faces_on_interface_mortar.clear();
        interface_dofs.clear();
        interface_fe_function = 0;

        if (mortar_flag)
        {
            triangulation_mortar.clear();
//            P_fine2coarse.reset();
//            P_coarse2fine.reset();
        }

        dof_handler_mortar.clear();
    }

    // MixedBiotProblemDD::run
    template <int dim>
    void MixedBiotProblemDD<dim>::run (const unsigned int refine,
                                             const std::vector<std::vector<unsigned int>> &reps,
                                             double tol,
                                             unsigned int maxiter,
                                             unsigned int quad_degree)
    {
        tolerance = tol;
        qdegree = quad_degree;

        const unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator);
        const unsigned int n_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
        pcout<<"\n\n Total number of processes is "<<n_processes<<"\n\n";

        Assert(reps[0].size() == dim, ExcDimensionMismatch(reps[0].size(), dim));

        if (mortar_flag)
        {
            Assert(n_processes > 1, ExcMessage("Mortar MFEM is impossible with 1 subdomain"));
            Assert(reps.size() >= n_processes + 1, ExcMessage("Some of the mesh parameters were not provided"));
        }

        for (unsigned int cycle=0; cycle<refine; ++cycle)
        {
            cg_iteration = 0;
            interface_dofs.clear();
            if(split_flag!=0){
				interface_dofs_elast.clear();
				interface_dofs_darcy.clear();
            }

            if (cycle == 0)
            {
                // Partitioning into subdomains (simple bricks)
                find_divisors<dim>(n_processes, n_domains);

                // Dimensions of the domain (unit hypercube)
                std::vector<double> subdomain_dimensions(dim);
                for (unsigned int d=0; d<dim; ++d)
                    subdomain_dimensions[d] = 1.0/double(n_domains[d]);

                get_subdomain_coordinates(this_mpi, n_domains, subdomain_dimensions, p1, p2);

                if (mortar_flag)
                    GridGenerator::subdivided_hyper_rectangle(triangulation, reps[this_mpi], p1, p2);
                else
                {
                    GridGenerator::subdivided_hyper_rectangle(triangulation, reps[0], p1, p2);
                    if (this_mpi == 0 || this_mpi == 3)
                      GridTools::distort_random (0.1*(1+this_mpi), triangulation, true);
                }

                if (mortar_flag)
                {
                    GridGenerator::subdivided_hyper_rectangle(triangulation_mortar, reps[n_processes], p1, p2);
                    pcout << "Mortar mesh has " << triangulation_mortar.n_active_cells() << " cells" << std::endl;
                }


            }
            else
            {
                if (mortar_flag == 0)
                    triangulation.refine_global(1);
                else if (mortar_degree <= 2)
                    triangulation.refine_global(1);
                else if (mortar_degree > 2)
                    triangulation.refine_global(1);

                if (mortar_flag){
                    triangulation_mortar.refine_global(1);
                    pcout << "Mortar mesh has " << triangulation_mortar.n_active_cells() << " cells" << std::endl;
                }
            }
//            pcout<<"\n \n grid diameter is : "<<GridTools::minimal_cell_diameter(triangulation)<<"\n \n ";
            pcout << "Making grid and DOFs...\n";
            make_grid_and_dofs();
            lambda_guess.resize(GeometryInfo<dim>::faces_per_cell);
            Alambda_guess.resize(GeometryInfo<dim>::faces_per_cell);
            if(split_flag!=0){
            lambda_guess_elast.resize(GeometryInfo<dim>::faces_per_cell);
            Alambda_guess_elast.resize(GeometryInfo<dim>::faces_per_cell);

            lambda_guess_darcy.resize(GeometryInfo<dim>::faces_per_cell);
            Alambda_guess_darcy.resize(GeometryInfo<dim>::faces_per_cell);
            }


            //Functions::ZeroFunction<dim> ic(static_cast<unsigned int> (dim*dim+dim+0.5*dim*(dim-1)+dim+1));
            pcout << "Projecting the initial conditions...\n";
            {
              InitialCondition<dim> ic;

              ConstraintMatrix constraints;
              constraints.close();
              VectorTools::project (dof_handler,
                                    constraints,
                                    QGauss<dim>(degree+5),
                                    ic,
                                    old_solution);

              solution = old_solution;
              output_results(cycle,refine);
              if(split_flag==2)
            	  older_solution=old_solution;
            }

            pcout << "Assembling system..." << "\n";
//        	pcout<<"\n split_flag value is "<<split_flag<<"\n";

            if(split_flag==0)
            	assemble_system ();
            else if(split_flag!=0)
            {
            	assemble_system_elast();
            	assemble_system_darcy();
            }


            if (Utilities::MPI::n_mpi_processes(mpi_communicator) != 1)
            	if(split_flag==0)
            		get_interface_dofs();
            	else if(split_flag!=0){
            		get_interface_dofs_elast();
            		get_interface_dofs_darcy();
            	}


            for(unsigned int i=0; i<prm.num_time_steps; i++)
            {
              prm.time += prm.time_step;

              solve_timestep (maxiter);
              compute_errors(cycle);
              if(split_flag==2)
            	  older_solution=old_solution;
              old_solution = solution;
              output_results (cycle, refine);
              max_cg_iteration=0;
              if(split_flag!=0)
            	  max_cg_iteration_darcy=0;

            }

            set_current_errors_to_zero();
            prm.time = 0.0;

            computing_timer.print_summary();
            computing_timer.reset();
        }

        reset_mortars();
    }

    template class MixedBiotProblemDD<2>;
    template class MixedBiotProblemDD<3>;
}
