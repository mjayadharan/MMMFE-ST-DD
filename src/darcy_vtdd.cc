/* ---------------------------------------------------------------------*/
/* ---------------------------------------------------------------------
 This is part of a program that  implements DD for time dependent Darcy flow with variable time stepping and MMMFE on non-matching grid.
 Also outputs the solution in a space-time domain. Refer to the paper/report on the same for further information.
 Template: BiotDD which was co-authored by Eldar K, University of Pittsburgh.
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan, University of Pittsburgh: 2020
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
#include <deal.II/fe/fe_face.h>
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
#include "../inc/darcy_vtdd.h"
#include "../inc/utilities.h"
#include "../inc/data.h"


namespace vt_darcy
{
    using namespace dealii;

    // DarcyVT class constructor
    template <int dim>
    DarcyVTProblem<dim>::DarcyVTProblem (const unsigned int degree,
                                                 const BiotParameters &bprm,
                                                 const unsigned int mortar_flag,
                                                 const unsigned int mortar_degree,
												 std::vector<char> bc_condition_vect)
            :
            mpi_communicator (MPI_COMM_WORLD),
            P_coarse2fine (false),
            P_fine2coarse (false),
            n_domains(dim,0),
            prm (bprm),
			bc_condition_vect (bc_condition_vect),
            degree (degree),
            mortar_degree(mortar_degree),
            mortar_flag (mortar_flag),
            cg_iteration(0),
			gmres_iteration(0),
            max_cg_iteration(0),
            qdegree(11),
//            fe (FE_BDM<dim>(degree), 1,
//                FE_DGQ<dim>(degree-1), 1),
			 fe (FE_RaviartThomas<dim>(degree), 1,
			                FE_DGQ<dim>(degree), 1),
            dof_handler (triangulation),
//			fe_face_q(0),
			fe_st (FE_RaviartThomas<dim+1>(degree), 1,
			           FE_DGQ<dim+1>(degree), 1),
//			fe_st (FE_RaviartThomas<dim+1>(degree), 1,
//			           FE_Nothing<dim+1>(), 1),
			dof_handler_st(triangulation_st),
            fe_mortar (FE_RaviartThomas<dim+1>(mortar_degree), 1,
                       FE_Nothing<dim+1>(), 1),
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
    void DarcyVTProblem<dim>::set_current_errors_to_zero()
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
    void DarcyVTProblem<dim>::make_grid_and_dofs ()
    {

        TimerOutput::Scope t(computing_timer, "Make grid and DoFs");
        	system_matrix.clear();

        //double lower_left, upper_right;
        //const unsigned int n_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
        const unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator);

        // Find neighbors
        neighbors.resize(GeometryInfo<dim>::faces_per_cell, 0);
        find_neighbors(dim, this_mpi, n_domains, neighbors);

        // Make interface data structures
        faces_on_interface.resize(GeometryInfo<dim>::faces_per_cell,0);
        faces_on_interface_mortar.resize(GeometryInfo<dim>::faces_per_cell,0);
        faces_on_interface_st.resize(GeometryInfo<dim>::faces_per_cell,0);

        // Label interface faces and count how many of them there are per interface
        mark_interface_faces(triangulation, neighbors, p1, p2, faces_on_interface);
        if (mortar_flag){
            mark_interface_faces_space_time(triangulation_mortar, neighbors, p1, p2, faces_on_interface_mortar);
            mark_interface_faces_space_time(triangulation_st,neighbors,p1,p2,faces_on_interface_st);

        }

        dof_handler.distribute_dofs(fe);
        DoFRenumbering::component_wise (dof_handler);




        if (mortar_flag)
        {
            dof_handler_mortar.distribute_dofs(fe_mortar);
            DoFRenumbering::component_wise (dof_handler_mortar);

            dof_handler_st.distribute_dofs(fe_st);
            DoFRenumbering::component_wise(dof_handler_st);

        }

        std::vector<types::global_dof_index> dofs_per_component ( dim + 1);
        DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
//        unsigned int n_s=0, n_u=0, n_g=0;
        unsigned int n_z = 0, n_p = 0;


        n_z = dofs_per_component[0];
        n_p = dofs_per_component[dim];

        n_flux = n_z;
        n_pressure = n_p;

        //Adding essential Neumann BC
        {
        	constraint_bc.clear();
        	for (int i=0; i<bc_condition_vect.size(); ++i){
        		if (bc_condition_vect[i] == 'D')
        				dir_bc_ids.push_back(100+i+1); // Dirichlet bc: left: 101, bottom: 102, right: 103, top:104
				else if (bc_condition_vect[i] == 'N')
					nm_bc_ids.push_back(100+i+1);   // Neumann bc: left: 101, bottom: 102, right: 103, top:104
        	} //end of updating bc_ids

            //adding normal flux(velocity.n as an essential bc)
            typename FunctionMap<dim>::type velocity_bc;
            std::map<types::global_dof_index,double> boundary_values_velocity;

            ZeroFunction<dim> velocity_bc_func(dim+1);
            for (int i=0; i<nm_bc_ids.size(); ++i)
            	velocity_bc[nm_bc_ids[i]] = &velocity_bc_func;

            VectorTools::project_boundary_values (dof_handler,
                                                  velocity_bc,
                                                  QGauss<dim-1>(degree+3),
                                                  boundary_values_velocity);

//            for (auto it=boundary_values_velocity.begin(); it!=boundary_values_velocity.end(); ++it)
//                if (it->first < n_s+n_u+n_g || it->first>=n_s+n_u+n_g+n_z)
//                    boundary_values_velocity.erase(it->first);

            typename std::map<types::global_dof_index,double>::const_iterator boundary_value_vel =
                    boundary_values_velocity.begin();
            for ( ; boundary_value_vel !=boundary_values_velocity.end(); ++boundary_value_vel)
            {
                if (!constraint_bc.is_constrained(boundary_value_vel->first))
                {
                    constraint_bc.add_line (boundary_value_vel->first);
                    constraint_bc.set_inhomogeneity (boundary_value_vel->first,
                                                   boundary_value_vel->second);
                }
            }

            //---------------------------------------------------end of velocity boundary values

        }
        constraint_bc.close();

        BlockDynamicSparsityPattern dsp(2, 2);
        dsp.block(0, 0).reinit (n_z, n_z);
        dsp.block(1, 0).reinit (n_p, n_z);
        dsp.block(0, 1).reinit (n_z, n_p);
        dsp.block(1, 1).reinit (n_p, n_p);

			dsp.collect_sizes ();
			DoFTools::make_sparsity_pattern (dof_handler, dsp, constraint_bc, false);

			// Initialize system matrix
			sparsity_pattern.copy_from(dsp);
			system_matrix.reinit (sparsity_pattern);

			// Reinit solution and RHS vectors
			solution_bar.reinit (2);
			solution_bar.block(0).reinit (n_z);
			solution_bar.block(1).reinit (n_p);
			solution_bar.collect_sizes ();
			solution_bar = 0;


			// Reinit solution and RHS vectors
			solution_star.reinit (2);
			solution_star.block(0).reinit (n_z);
			solution_star.block(1).reinit (n_p);
			solution_star.collect_sizes ();
			solution_star = 0;

			system_rhs_bar.reinit (2);;
			system_rhs_bar.block(0).reinit (n_z);
			system_rhs_bar.block(1).reinit (n_p);
			system_rhs_bar.collect_sizes ();
			system_rhs_bar = 0;

			// Required for essential(Neumann bc), used in assemble_system
			system_rhs_bar_bc.reinit (2);;
			system_rhs_bar_bc.block(0).reinit (n_z);
			system_rhs_bar_bc.block(1).reinit (n_p);
			system_rhs_bar_bc.collect_sizes ();
			system_rhs_bar_bc = 0;

			system_rhs_star.reinit (2);
			system_rhs_star.block(0).reinit (n_z);
			system_rhs_star.block(1).reinit (n_p);
			system_rhs_star.collect_sizes ();
			system_rhs_star = 0;

			//adding vectors required for storing mortar and space-time subdomain solutions.
			if (mortar_flag)
			        {
				//Mortar part.
			            std::vector<types::global_dof_index> dofs_per_component_mortar (dim+1 + 1);
			            DoFTools::count_dofs_per_component (dof_handler_mortar, dofs_per_component_mortar);
			            unsigned int  n_z_mortar=0, n_p_mortar=0;

			            n_z_mortar = dofs_per_component_mortar[0]; //For RT mortar space
			            n_p_mortar = dofs_per_component_mortar[dim+1];


			            solution_bar_mortar.reinit(2);
			            solution_bar_mortar.block(0).reinit (n_z_mortar);
			            solution_bar_mortar.block(1).reinit (n_p_mortar);
			            solution_bar_mortar.collect_sizes ();
			            solution_bar_mortar=0;

			            solution_star_mortar.reinit(2);
			            solution_star_mortar.block(0).reinit (n_z_mortar);
			            solution_star_mortar.block(1).reinit (n_p_mortar);
			            solution_star_mortar.collect_sizes ();
			            solution_star_mortar=0;

						//Space-time part.
			            std::vector<types::global_dof_index> dofs_per_component_st (dim+1 + 1);
			            DoFTools::count_dofs_per_component (dof_handler_st, dofs_per_component_st);

			            n_flux_st = dofs_per_component_st[0]; //For RT mortar space
			            n_pressure_st= dofs_per_component_st[dim+1];


			            solution_bar_st.reinit(2);
			            solution_bar_st.block(0).reinit (n_flux_st);
			            solution_bar_st.block(1).reinit (n_pressure_st);
			            solution_bar_st.collect_sizes ();
			            solution_bar_st=0;

			            solution_star_st.reinit(2);
			            solution_star_st.block(0).reinit (n_flux_st);
			            solution_star_st.block(1).reinit (n_pressure_st);
			            solution_star_st.collect_sizes ();
			            solution_star_st=0;

			            solution_st.reinit(solution_bar_st);
			            solution_st.collect_sizes();
			            solution_st =0;

			            solution_bar_collection.resize(prm.num_time_steps,solution_bar);
			        }



        solution.reinit (2);
        solution.block(0).reinit (n_z);
        solution.block(1).reinit (n_p);
        solution.collect_sizes ();
        solution = 0;
        old_solution.reinit(solution);
        initialc_solution.reinit(solution);
        old_solution_for_jump.reinit(solution);
        pressure_projection.reinit(solution);
        old_pressure_projection.reinit(solution);

        initialc_solution=0;
        old_solution_for_jump=0;
        pressure_projection=0;
        old_pressure_projection=0;

        pcout << "N flux subdom dofs: " << n_flux << std::endl;
        pcout << "N pressure subdom dofs: " << n_pressure << std::endl;

        pcout << "N flux space-time dofs: " << n_flux_st << std::endl;
        pcout << "N pressure space-time dofs: " << n_pressure_st << std::endl;
    }


    // MixedBiotProblemDD - assemble_system
    template <int dim>
    void DarcyVTProblem<dim>::assemble_system ()
    {
        TimerOutput::Scope t(computing_timer, "Assemble system");
        system_matrix = 0;
        system_rhs_bar_bc = 0;

        QGauss<dim>   quadrature_formula(degree+2);

        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    | update_gradients |
                                 update_quadrature_points  | update_JxW_values);

        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();

        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        const KInverse<dim> k_inverse;
        std::vector<Tensor<2,dim>>  k_inverse_values(n_q_points);

        // Velocity and Pressure DoFs
        const FEValuesExtractors::Vector velocity (0);
        const FEValuesExtractors::Scalar pressure (dim);


        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            local_matrix = 0;

            k_inverse.value_list (fe_values.get_quadrature_points(), k_inverse_values);

            // Velocity and pressure
            std::vector<Tensor<1,dim>>                phi_u(dofs_per_cell);
            std::vector <double>                      div_phi_u(dofs_per_cell);
            std::vector <double>                      phi_p(dofs_per_cell);


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
                {

                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {


                        local_matrix(i, j) += ( phi_u[i] * k_inverse_values[q] * phi_u[j] - phi_p[j] * div_phi_u[i]                                     // Darcy law
                                               + prm.time_step*div_phi_u[j] * phi_p[i] + prm.c_0*phi_p[i]*phi_p[j] )
                                              * fe_values.JxW(q);
                    }
                }
            }

            cell->get_dof_indices (local_dof_indices);
//            for (unsigned int i=0; i<dofs_per_cell; ++i)
//                for (unsigned int j=0; j<dofs_per_cell; ++j)
//                    system_matrix.add (local_dof_indices[i],
//                                       local_dof_indices[j],
//                                       local_matrix(i,j));
            Vector<double> local_rhs(dofs_per_cell);
            local_rhs = 0;
            constraint_bc.distribute_local_to_global (local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs_bar_bc);
        }

        pcout << "  ...factorized..." << "\n";
        A_direct.initialize(system_matrix);
    }


    // MixedBiotProblemDD - initialize the interface data structure for coupled monolithic sheme
    template <int dim>
    void DarcyVTProblem<dim>::get_interface_dofs ()
    {
        TimerOutput::Scope t(computing_timer, "Get interface DoFs");

        { //getting interace dof normal subdomain part for no-mortar a
        	//a nd getting interace dofs for mortar grid in case of mortar.

        interface_dofs.resize(GeometryInfo<dim>::faces_per_cell, std::vector<types::global_dof_index> ());


        std::vector<types::global_dof_index> local_face_dof_indices;

//        typename DoFHandler<dim>::active_cell_iterator cell, endc;

        if (mortar_flag == 0)
        {
        	typename DoFHandler<dim>::active_cell_iterator cell, endc;
            cell = dof_handler.begin_active(), endc = dof_handler.end();
            local_face_dof_indices.resize(fe.dofs_per_face);
            for (;cell!=endc;++cell)
            {
                for (unsigned int face_n=0;
                     face_n<GeometryInfo<dim>::faces_per_cell;
                     ++face_n)
                    if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() < 100)
                    {
                        cell->face(face_n)->get_dof_indices (local_face_dof_indices, 0);

                        for (auto el : local_face_dof_indices)
                                interface_dofs[cell->face(face_n)->boundary_id()-1].push_back(el);
                    }
             }
        }
        else
        {
        	typename DoFHandler<dim+1>::active_cell_iterator cell, endc;
        	cell = dof_handler_mortar.begin_active(),
                    endc = dof_handler_mortar.end();
            local_face_dof_indices.resize(fe_mortar.dofs_per_face);
            for (;cell!=endc;++cell)
            {
                for (unsigned int face_n=0;
                     face_n<GeometryInfo<dim>::faces_per_cell;
                     ++face_n)
                    if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() < 100)
                    {
                        cell->face(face_n)->get_dof_indices (local_face_dof_indices, 0);
                        for (auto el : local_face_dof_indices)

                                interface_dofs[cell->face(face_n)->boundary_id()-1].push_back(el);
                    }
             }
        	}
        }// end of getting normal/mortar inteface_dofs.

        if(mortar_flag){ //getting interace dof normal subdomain part for no-mortar a
        	//a nd getting interace dofs for mortar grid in case of mortar.

			interface_dofs_subd.resize(GeometryInfo<dim>::faces_per_cell, std::vector<types::global_dof_index> ());
	        face_dofs_subdom.resize(GeometryInfo<dim>::faces_per_cell, std::vector<types::global_dof_index> ());

			std::vector<types::global_dof_index> local_face_dof_indices;

			typename DoFHandler<dim>::active_cell_iterator cell, endc;

				cell = dof_handler.begin_active(),
						endc = dof_handler.end();
				local_face_dof_indices.resize(fe.dofs_per_face);


			for (;cell!=endc;++cell)
			{
				for (unsigned int face_n=0;
					 face_n<GeometryInfo<dim>::faces_per_cell;
					 ++face_n)
				{

				//start of getting face dofs.
					cell->face(face_n)->get_dof_indices (local_face_dof_indices, 0);

					for (auto el : local_face_dof_indices)
					{
						face_dofs_subdom[face_n].push_back(el);
					}
				//end of getting face dofs

					//start of getting interface dofs.
					if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() < 100)
					{

						for (auto el : local_face_dof_indices)
								interface_dofs_subd[cell->face(face_n)->boundary_id()-1].push_back(el);
					}
				//end of getting interface dofs
			 }
			}
			}// end of getting subdomain interface dofs in mortar case: used for space-time mortar.
    }

    template <int dim>
    void DarcyVTProblem<dim>::get_interface_dofs_st()
    {
        TimerOutput::Scope t(computing_timer, "Get interface DoFs S-T");
        pcout<<"inteface_dofs_subd size is : "<<interface_dofs_subd.size()<<"\n";
        unsigned int n_faces = GeometryInfo<dim>::faces_per_cell;
        interface_dofs_st.resize(GeometryInfo<dim>::faces_per_cell, std::vector<types::global_dof_index> ());
        face_dofs_st.resize(GeometryInfo<dim>::faces_per_cell, std::vector<types::global_dof_index> ());

        std::vector<types::global_dof_index> local_face_dof_indices;
        typename DoFHandler<dim+1>::active_cell_iterator cell, endc;

        cell = dof_handler_st.begin_active(),
               endc = dof_handler_st.end();
        local_face_dof_indices.resize(fe_st.dofs_per_face);

        for (;cell!=endc;++cell)
        {
            for (unsigned int face_n=0; face_n<n_faces; ++face_n)
            {
            	//start of getting face dofs
                        cell->face(face_n)->get_dof_indices (local_face_dof_indices, 0);
                        for (auto el : local_face_dof_indices){
                                face_dofs_st[face_n].push_back(el);
                        }
                    //end of getting face dofs

            	//start of getting interface dofs
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() < 100)
                {
                    for (auto el : local_face_dof_indices){
                            interface_dofs_st[cell->face(face_n)->boundary_id()-1].push_back(el);

                    }
                } //end of getting interface dofs
            }
        }
    }





  // MixedBiotProblemDD - assemble RHS of star problems
  template <int dim>
  void DarcyVTProblem<dim>::assemble_rhs_bar ()
  {
//      TimerOutput::Scope t(computing_timer, "Assemble RHS bar");
      system_rhs_bar = 0;

      QGauss<dim>   quadrature_formula(degree+2);
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

      PressureBoundaryValues<dim>     pressure_boundary_values;
      pressure_boundary_values.set_time(prm.time);
      std::vector<double>         boundary_values_flow (n_face_q_points);

      RightHandSidePressure<dim>      right_hand_side_pressure(prm.c_0,prm.alpha);
      right_hand_side_pressure.set_time(prm.time);
      std::vector<double>         rhs_values_flow (n_q_points);


      typename DoFHandler<dim>::active_cell_iterator
              cell = dof_handler.begin_active(),
              endc = dof_handler.end();
      for (; cell!=endc; ++cell)
      {
          local_rhs = 0;
          fe_values.reinit (cell);

          right_hand_side_pressure.value_list(fe_values.get_quadrature_points(), rhs_values_flow);

          // Velocity and Pressure DoFs
          const FEValuesExtractors::Vector velocity (0);
          const FEValuesExtractors::Scalar pressure (dim);


          std::vector <double>                      phi_p(dofs_per_cell);

          std::vector<double> old_pressure_values(n_q_points);

          if(std::fabs(prm.time-prm.time_step)<1.0e-10)
        	  fe_values[pressure].get_function_values (initialc_solution, old_pressure_values);
          else
        	  fe_values[pressure].get_function_values (old_solution, old_pressure_values);

          for (unsigned int q=0; q<n_q_points; ++q)
          {

              for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                  // Evaluate test functions
                  phi_p[k] = fe_values[pressure].value (k, q);

              }

              for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                  local_rhs(i) += ( prm.time_step*phi_p[i] * rhs_values_flow[q]
                                            + prm.c_0*old_pressure_values[q] * phi_p[i] )
                                           * fe_values.JxW(q);

              }
          }


          Tensor<2,dim> sigma;
          Tensor<1,dim> sigma_n;
          for (unsigned int face_no=0;
               face_no<GeometryInfo<dim>::faces_per_cell;
               ++face_no)
          {
              if (cell->at_boundary(face_no))
              {
            	  bool at_dir_boundary; //to check whether the given boundary is part of Dirichlet bc.
            	  at_dir_boundary = is_inside<int>(dir_bc_ids, cell->face(face_no)->boundary_id());
            	  if (at_dir_boundary)// Dirichlet(pressure) part of the boundary.
				  {
					  fe_face_values.reinit (cell, face_no);

					  pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), boundary_values_flow);

					  for (unsigned int q=0; q<n_face_q_points; ++q)
						  for (unsigned int i=0; i<dofs_per_cell; ++i)
						  {
							  local_rhs(i) += -(fe_face_values[velocity].value (i, q) *
														 fe_face_values.normal_vector(q) *
														 boundary_values_flow[q] *
														 fe_face_values.JxW(q));

						  }
				  }
              }
          }


          cell->get_dof_indices (local_dof_indices);
//          for (unsigned int i=0; i<dofs_per_cell; ++i)
//              system_rhs_bar(local_dof_indices[i]) += local_rhs(i);
          FullMatrix<double> local_matrix(dofs_per_cell);
          local_matrix = 0;
          constraint_bc.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs_bar);
      }
  }

    // MixedBiotProblemDD - assemble RHS of star problems
    template <int dim>
    void DarcyVTProblem<dim>::assemble_rhs_star ()
    {
//        TimerOutput::Scope t(computing_timer, "Assemble RHS star");
        system_rhs_star = 0;

        QGauss<dim>   quadrature_formula(degree+2);
        QGauss<dim-1> face_quadrature_formula(qdegree);

        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);

        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();

        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


        const FEValuesExtractors::Vector velocity (0);
        const FEValuesExtractors::Scalar pressure (dim);

        std::vector<Tensor<1, dim>> interface_values_flux(n_face_q_points);

        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();
        for (;cell!=endc;++cell)
        {
            local_rhs = 0;
            fe_values.reinit (cell);

            std::vector <double> phi_p(dofs_per_cell);
            std::vector<double> old_pressure_values(n_q_points);

            if(std::fabs(prm.time-prm.time_step)>1.0e-10)
            {
            	fe_values[pressure].get_function_values (old_solution, old_pressure_values);

				for (unsigned int q=0; q<n_q_points; ++q)
				   {

					   for (unsigned int k=0; k<dofs_per_cell; ++k)
					   {
						   // Evaluate test functions
						   phi_p[k] = fe_values[pressure].value (k, q);

					   }

					   for (unsigned int i=0; i<dofs_per_cell; ++i)
						   local_rhs(i) += ( prm.c_0*old_pressure_values[q] * phi_p[i] )* fe_values.JxW(q);
				   }
            }


            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() < 100)
                {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values[velocity].get_function_values (interface_fe_function_subdom, interface_values_flux);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            local_rhs(i) += -(fe_face_values[velocity].value (i, q) *
                                              fe_face_values.normal_vector(q) *
                                              interface_values_flux[q] * get_normal_direction(cell->face(face_n)->boundary_id()-1) *
                                              fe_face_values.normal_vector(q) *
                                              fe_face_values.JxW(q));
                        }
                }

            cell->get_dof_indices (local_dof_indices);
//            for (unsigned int i=0; i<dofs_per_cell; ++i)
//                system_rhs_star(local_dof_indices[i]) += local_rhs(i);
            FullMatrix<double> local_matrix(dofs_per_cell);
            local_matrix = 0;
            constraint_bc.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs_star);
        }
    }



    // MixedBiotProblemDD::solvers
    template <int dim>
    void DarcyVTProblem<dim>::solve_bar ()
    {
    	system_rhs_bar.sadd(1.0, system_rhs_bar_bc);
        A_direct.vmult (solution_bar, system_rhs_bar);

        constraint_bc.distribute(solution_bar);

    }



    template <int dim>
    void DarcyVTProblem<dim>::solve_star ()
    {
//        TimerOutput::Scope t(computing_timer, "Solve star");

        A_direct.vmult (solution_star, system_rhs_star);

    }

    template<int dim>
    void DarcyVTProblem<dim>::solve_darcy_vt(unsigned int maxiter)
    {
    			prm.time=0.0;
			  for(unsigned int time_level=0; time_level<prm.num_time_steps; time_level++)
			  {
				  prm.time +=prm.time_step;
				  solve_timestep(0,time_level);
			  }//end of solving the bar problems.
			  prm.time=0.0;

			  pcout << "\nStarting GMRES iterations.........\n";
			  local_gmres(maxiter);
    }

    template<int dim>
    void DarcyVTProblem<dim>::solve_timestep(int star_bar_flag, unsigned int time_level)
    {
    	switch(star_bar_flag){

			case 0: //solving bar problem at time_level and carrying solution to st mesh boundary.
				assemble_rhs_bar();
				solve_bar();
				if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
				{
					solution =solution_bar;
					compute_errors(refinement_index, time_level);
					output_results(refinement_index,total_refinements);
				}
				old_solution = solution_bar;
				system_rhs_bar=0;
				//transferring solution to st mesh.
				if(mortar_flag){
					solution_bar_collection[time_level]= solution_bar;
					subdom_to_st_distribute(solution_bar_st, solution_bar, time_level,prm.time_step);
					solution_bar=0;
				}
				break; //break for case 0

			case 1: //solving star problem at time_level and carrying solution to st mesh boundary multiple times during local_gmres.
				st_to_subdom_distribute(interface_fe_function_st, interface_fe_function_subdom, time_level,prm.time_step);
				assemble_rhs_star();
				solve_star();

				interface_fe_function_subdom=0;
				old_solution = solution_star;

				subdom_to_st_distribute(solution_star_st, solution_star, time_level,prm.time_step);

				solution_star=0;


				break; //break for case 1

			case 2: //solving star problem final time after gmres converges: also compute error and output result corresponding to time_level.
				st_to_subdom_distribute(interface_fe_function_st, interface_fe_function_subdom, time_level,prm.time_step);
				assemble_rhs_star();
				solve_star();
				interface_fe_function_subdom=0;
				old_solution = solution_star;
				solution=0;
				solution.sadd(1.0,solution_star);
				solution.sadd(1.0,solution_bar_collection[time_level]);
				final_solution_transfer(solution_st, solution, time_level, prm.time_step);

				compute_errors(refinement_index, time_level);
				output_results(refinement_index,total_refinements);
				old_solution_for_jump = solution;

				break; //break for case 2

    	}

    }

//    template <int dim>
//    void DarcyVTProblem<dim>::compute_multiscale_basis ()
//    {
//        TimerOutput::Scope t(computing_timer, "Compute multiscale basis");
//        ConstraintMatrix constraints;
//        QGauss<dim-1> quad(qdegree);
//        FEFaceValues<dim> fe_face_values (fe, quad,
//                                          update_values    | update_normal_vectors |
//                                          update_quadrature_points  | update_JxW_values);
//
//        std::vector<size_t> block_sizes {solution_bar_mortar.block(0).size(), solution_bar_mortar.block(1).size()};
//        long n_interface_dofs = 0;
//
//        for (auto vec : interface_dofs)
//            for (auto el : vec)
//                n_interface_dofs += 1;
//
//        multiscale_basis.resize(n_interface_dofs);
//        BlockVector<double> tmp_basis (solution_bar_mortar);
//
//        interface_fe_function_subdom.reinit(solution_bar);
//
//        unsigned int ind = 0;
//        for (unsigned int side=0; side<GeometryInfo<dim>::faces_per_cell; ++side)
//            for (unsigned int i=0; i<interface_dofs[side].size(); ++i)
//            {
//                interface_fe_function_subdom = 0;
//                multiscale_basis[ind].reinit(solution_bar_mortar);
//                multiscale_basis[ind] = 0;
//
//                tmp_basis = 0;
//                tmp_basis[interface_dofs[side][i]] = 1.0;
//                project_mortar(P_coarse2fine, dof_handler_mortar, tmp_basis, quad, constraints, neighbors, dof_handler, interface_fe_function);
//
//                interface_fe_function_subdom.block(1) = 0;
////                interface_fe_function.block(2) = 0;
////                interface_fe_function.block(4) = 0;
//                assemble_rhs_star();
//                solve_star();
//
//                project_mortar(P_fine2coarse, dof_handler, solution_star, quad, constraints, neighbors, dof_handler_mortar, multiscale_basis[ind]);
//                ind += 1;
//            }
//
//    }

    // from space-time subdomain mesh to 2d sub-domain space mesh
    template<int dim>
    void DarcyVTProblem<dim>::st_to_subdom_distribute (BlockVector<double> &vector_st,
    												   BlockVector<double> &vector_subdom,
													   unsigned int &time_level, double scale_factor)
    {
        for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell; ++side)
              if (neighbors[side] >= 0){
            	  int interface_dofs_side_size = interface_dofs_subd[side].size();
            	  for(int i=0; i<interface_dofs_side_size; i++)
            		  vector_subdom[ interface_dofs_subd[side][i]] =(1/scale_factor)* vector_st[ interface_dofs_st[side][interface_dofs_side_size*time_level +i]];
              }

    }

    // // from 2-d subdomain space mesh to 3-d subdomain space-time mesh
    template<int dim>
    void DarcyVTProblem<dim>::subdom_to_st_distribute (BlockVector<double> &vector_st,
    												   BlockVector<double> &vector_subdom,
													   unsigned int &time_level, double scale_factor)
    {
        for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell; ++side)
            if (neighbors[side] >= 0){
               	  int interface_dofs_side_size = interface_dofs_subd[side].size();
               	  for(int i=0; i<interface_dofs_side_size; i++)
               		vector_st[ interface_dofs_st[side][interface_dofs_side_size*time_level +i]]= scale_factor*vector_subdom[ interface_dofs_subd[side][i]];
                 }

    }

    // // transfering solution from 2d to space-time 3d mesh
    template<int dim>
    void DarcyVTProblem<dim>::final_solution_transfer (BlockVector<double> &solution_st,
    												   BlockVector<double> &solution_subdom,
													   unsigned int &time_level, double scale_factor)
    {
    	//transferring pressure solution.
    	assert(n_pressure_st== prm.num_time_steps*n_pressure);
    	for(unsigned int i=0; i<n_pressure; i++)
    	{
    		solution_st.block(1)[(time_level*n_pressure) + i]= solution_subdom.block(1)[i];
    	}

    	//transferring velocity solution.
        for (unsigned int side = 0; side < GeometryInfo<dim>::faces_per_cell; ++side)
        {
               	  int face_dofs_side_size = face_dofs_subdom[side].size();
               	  for(int i=0; i<face_dofs_side_size; i++)
               		solution_st[ face_dofs_st[side][face_dofs_side_size*time_level +i]]= scale_factor*solution_subdom[ face_dofs_subdom[side][i]];
        }

    }

    //Functions for GMRES:-------------------


      //finding the l2 norm of a std::vector<double> vector
      template <int dim>
      double
	  DarcyVTProblem<dim>::vect_norm(std::vector<double> v){
      	double result = 0;
      	for(unsigned int i=0; i<v.size(); ++i){
      		result+= v[i]*v[i];
      	}
      	return sqrt(result);

      }
      //Calculating the given rotation matrix
      template <int dim>
      void
	  DarcyVTProblem<dim>::givens_rotation(double v1, double v2, double &cs, double &sn){

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
	  DarcyVTProblem<dim>::apply_givens_rotation(std::vector<double> &h, std::vector<double> &cs, std::vector<double> &sn,
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
	  DarcyVTProblem<dim>::back_solve(std::vector<std::vector<double>> H, std::vector<double> beta, std::vector<double> &y, unsigned int k_iteration){
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
		DarcyVTProblem<dim>::local_gmres(const unsigned int maxiter)
        {
//          TimerOutput::Scope t(computing_timer, "Local GMRES");

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
              {	// Edit 1 : In the following places to interface_dofs_mortar iff interface_dofs is replaced with interface_dofs_mortar.
                    interface_data_receive[side].resize(interface_dofs[side].size(),
                                                        0);
                    interface_data_send[side].resize(interface_dofs[side].size(), 0);
                    interface_data[side].resize(interface_dofs[side].size(), 0);

              }

          // Extra for projections from mortar to fine grid and RHS assembly
          //projections with quad_mortar
          Quadrature<dim > quad;
          quad = QGauss<dim >(qdegree);

          Quadrature<dim > quad_project;
          quad_project = QGauss<dim >(qdegree);

          ConstraintMatrix  constraints;
          constraints.clear();
          constraints.close();
          int temp_array_size = maxiter/4;
          //GMRES structures and parameters
          std::vector<double>	sn(temp_array_size);
          std::vector<double>	cs(temp_array_size);
          std::vector<double>	Beta(temp_array_size); //beta for each side
          std::vector<std::vector<double>>	H(temp_array_size,Beta);
          std::vector<double> e_all_iter(temp_array_size+1); //error will be saved here after each iteration
          double combined_error_iter =0; //sum of error_iter_side




          std::vector<std::vector<double>> r(n_faces_per_cell);
          std::vector<double> r_norm_side(n_faces_per_cell,0);
          std::vector<std::vector<std::vector<double>>>	Q_side(n_faces_per_cell) ;
          std::vector<std::vector<double>>  Ap(n_faces_per_cell);

          //defing q  to push_back to Q (reused in Arnoldi algorithm)
          std::vector<std::vector<double>> q(n_faces_per_cell);

          interface_fe_function_st.reinit(solution_bar_st);
          interface_fe_function_subdom.reinit(solution_bar);

          if (mortar_flag == 1)
          {
              interface_fe_function_mortar.reinit(solution_bar_mortar);
              interface_fe_function_mortar=0;
              project_mortar<dim>(P_fine2coarse, dof_handler_st, solution_bar_st, quad_project, constraints, neighbors, dof_handler_mortar, solution_bar_mortar);




          }



          double l0 = 0.0;
          // GMRES with rhs being 0 and initial guess lambda = 0
          for (unsigned side = 0; side < n_faces_per_cell; ++side)

            if (neighbors[side] >= 0)
              {

                Ap[side].resize(interface_dofs[side].size(), 0);
                lambda[side].resize(interface_dofs[side].size(), 0);
                q[side].resize(interface_dofs[side].size());
                r[side].resize(interface_dofs[side].size(), 0);
                std::vector<double> r_receive_buffer(r[side].size());
                //temporarily fixing a size for Q_side matrix
                Q_side[side].resize(temp_array_size+1,q[side]);

                if(mortar_flag)
                	for (unsigned int i=0;i<interface_dofs[side].size();++i){
                	                      r[side][i] = get_normal_direction(side) * solution_bar_mortar[interface_dofs[side][i]];

                	}
                else
                	for (unsigned int i = 0; i < interface_dofs[side].size(); ++i)
						r[side][i] = get_normal_direction(side) *
									   solution_bar[interface_dofs[side][i]] ;


                  MPI_Sendrecv(&r[side][0],
                           r[side].size(),
                           MPI_DOUBLE,
                           neighbors[side],
                           this_mpi,

						   &r_receive_buffer[0],
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
             		 Q_side[side][0] = q[side];
             	  }


          //end----------- of caluclating first element of Q[side]-----------------
          e_all_iter[0]=r_norm;
          pcout<<"\n\n r_norm is \n\n"<<r_norm<<"\n\n";
          Beta[0]=r_norm;

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
            	  interface_fe_function_mortar=0;
                  for (unsigned int side=0;side<n_faces_per_cell;++side)
                      for (unsigned int i=0;i<interface_dofs[side].size();++i)
                          interface_fe_function_mortar[interface_dofs[side][i]] = interface_data[side][i];

                  project_mortar(P_coarse2fine, dof_handler_mortar,
                                 interface_fe_function_mortar,
                                 quad_project,
                                 constraints,
                                 neighbors,
                                 dof_handler_st,
                                 interface_fe_function_st);

				//Solving the star problems.
                  prm.time=0.0;
        		  for(unsigned int time_level=0; time_level<prm.num_time_steps; time_level++)
        			  {
//                      pcout<<"reached top of solve_time_step 0 1\n";

        				  prm.time +=prm.time_step;
        				  solve_timestep(1,time_level);

        			  }//end of solving the star problems
        			  prm.time=0.0;

              }

              else
              {
                  for (unsigned int side=0; side<n_faces_per_cell; ++side)
                      for (unsigned int i=0; i<interface_dofs[side].size(); ++i)
                          interface_fe_function_subdom[interface_dofs[side][i]] = interface_data[side][i];

                  assemble_rhs_star();
                  solve_star();
              }




              cg_iteration++;
              if (mortar_flag == 1){

                        project_mortar<2>(P_fine2coarse,
                                       dof_handler_st,
                                       solution_star_st,
                                       quad_project,
                                       constraints,
                                       neighbors,
                                       dof_handler_mortar,
                                       solution_star_mortar);

              }

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
                    MPI_Sendrecv(&interface_data_send[side][0],
                            interface_dofs[side].size(),
                             MPI_DOUBLE,
                             neighbors[side],
                             this_mpi,
							 &interface_data_receive[side][0],
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
                			Q_side[side][k_counter+1]=q[side];
                		}

                	//Pushing back  h to H as H(k)}

                		H[k_counter]=h;
               //---end of Arnoldi Algorithm

              //Eliminating the last element in H ith row and updating the rotation matrix.
              apply_givens_rotation(H[k_counter],cs,sn,
              						k_counter);
              //Updating the residual vector
              Beta[k_counter+1]=-sn[k_counter]*Beta[k_counter];
              Beta[k_counter]*=cs[k_counter];

              //Combining error at kth iteration
              combined_error_iter=fabs(Beta[k_counter+1])/r_norm;


              //saving the combined error at each iteration
              e_all_iter[k_counter+1]=(combined_error_iter);

              pcout << "\r  ..." << cg_iteration
                    << " iterations completed, (residual = " << combined_error_iter
                    << ")..." << std::flush;
              // Exit criterion
              if (combined_error_iter/e_all_iter[0] < tolerance)
                {
                  pcout << "\n  GMRES converges in " << cg_iteration << " iterations!\n and residual is"<<combined_error_iter/e_all_iter[0]<<"\n";
                  break;
                }
              else if(k_counter>maxiter-2)
            	  pcout << "\n  GMRES doesn't converge after  " << k_counter << " iterations!\n";

              //maxing interface_data_receive and send zero so it can be used is solving for Ap(or A*Q([k_counter]).
              for (unsigned int side = 0; side < n_faces_per_cell; ++side)
                {
            	  for(int i=0; i<interface_data_send[side].size(); i++){
            		  interface_data_receive[side][i]=0;
            		  interface_data_send[side][i]=0;
            	  }


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
                                    quad_project,
                                    constraints,
                                    neighbors,
                                    dof_handler_st,
                                    interface_fe_function_st);

                 }
                 else
                 {
                     interface_data = lambda;
                     for (unsigned int side=0; side<n_faces_per_cell; ++side)
                         for (unsigned int i=0; i<interface_dofs[side].size(); ++i)
                             interface_fe_function_subdom[interface_dofs[side][i]] = interface_data[side][i];
                 }

          //Finally solving star problems.
          max_cg_iteration=cg_iteration;
		  for(unsigned int time_level=0; time_level<prm.num_time_steps; time_level++)
			  {
				  prm.time +=prm.time_step;
				  solve_timestep(2,time_level);



			  }//end of solving the star problems at the endand outputting results for all time levels.
			  prm.time=0.0;
          pcout<<"finished local_gmres"<<"\n";


        }

    // MixedBiotProblemDD::compute_interface_error in the l2 norm
    template <int dim>
    double DarcyVTProblem<dim>::compute_interface_error_l2()
    {
//        system_rhs_star = 0;
        double error_calculated =0;

        QGauss<dim> quad (qdegree);
        QGauss<dim> project_quad (qdegree);
        FEFaceValues<dim+1> fe_face_values (fe_mortar, quad,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);

        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
        const unsigned int dofs_per_cell_mortar = fe_mortar.dofs_per_cell;
        const FEValuesExtractors::Vector velocity (0);
        PressureBoundaryValues<dim+1>     pressure_boundary_values;
        pressure_boundary_values.set_time(prm.time);
        std::vector<Tensor<1, dim+1>> interface_values_flux(n_face_q_points);
        std::vector<Tensor<1, dim+1>> solution_values_flow(n_face_q_points);
        std::vector<double> pressure_values(n_face_q_points);

        typename DoFHandler<dim+1>::active_cell_iterator
                cell = dof_handler_mortar.begin_active(),
                endc = dof_handler_mortar.end();
        double error_add;
        for (;cell!=endc;++cell)
        {
            for (unsigned int face_n=0;
                 face_n<GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
                if (cell->at_boundary(face_n) && cell->face(face_n)->boundary_id() < 100)
                {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values[velocity].get_function_values (interface_fe_function_mortar, interface_values_flux);
                    pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), pressure_values);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell_mortar; ++i)
                        {
                        	 	 	 error_add = pow((
                	                 (interface_values_flux[q] * get_normal_direction(cell->face(face_n)->boundary_id()-1) *
                	                 fe_face_values.normal_vector(q) - pressure_values[q])),2)*
                	                 fe_face_values.JxW(q);
                        	 error_calculated+= error_add;
                        }
                }

        }

        return error_calculated;
    }

    // MixedBiotProblemDD::compute_interface_error in the l2 norm
       template <int dim>
       double DarcyVTProblem<dim>::compute_jump_error()
       {
           double error_calculated =0;

           QTrapez<1>      q_trapez;
           QIterated<dim>  quad(q_trapez,degree+2);
           FEValues<dim> fe_values (fe, quad,
                                             update_values  | update_quadrature_points  |
											 update_JxW_values);

           const unsigned int n_q_points = fe_values.get_quadrature().size();
           const unsigned int dofs_per_cell = fe.dofs_per_cell;
           const FEValuesExtractors::Scalar pressure_mask (dim);
           std::vector<double> pressure_values_current(n_q_points);
           std::vector<double> pressure_values_old(n_q_points);

           typename DoFHandler<dim>::active_cell_iterator
                   cell = dof_handler.begin_active(),
                   endc = dof_handler.end();
           for (;cell!=endc;++cell)
           {

                       fe_values.reinit (cell);
                       fe_values[pressure_mask].get_function_values (solution, pressure_values_current);
                       fe_values[pressure_mask].get_function_values (old_solution_for_jump, pressure_values_old);


                       for (unsigned int q=0; q<n_q_points; ++q)
                           for (unsigned int i=0; i<dofs_per_cell; ++i)
                        	   error_calculated+=  pow((pressure_values_current[q] - pressure_values_old[q]),2)*
                        	                      fe_values.JxW(q);
           }

           return error_calculated;
       }

    // MixedBiotProblemDD::compute_errors
    template <int dim>
    void DarcyVTProblem<dim>::compute_errors (const unsigned int refinement_index, unsigned int time_level)
    {
//      TimerOutput::Scope t(computing_timer, "Compute errors");

      const unsigned int total_dim = static_cast<unsigned int>(dim + 1);
      const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), total_dim);
      const ComponentSelectFunction<dim> pressure_mask(static_cast<unsigned int>(dim), total_dim);

      ExactSolution<dim> exact_solution;
      exact_solution.set_time(prm.time);

      PressureBoundaryValues<dim> pressure_function; //to calculate the projectin of exact projection on to piecewise constants.
      pressure_function.set_time(prm.time + (prm.time_step*0.5));
      std::vector<Point<dim>> pressure_support_points(n_pressure);
      std::vector<double> projected_pressure_values(n_pressure);

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
      zerozeros=0;
      ZeroFunction<dim> zero_function(dim+1);

      // Computing pressure error and norm.
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature,
                                         VectorTools::L2_norm,
                                         &pressure_mask);
//      const double p_l2_error = cellwise_errors.norm_sqr();
      const double p_l2_error = VectorTools::compute_global_error(triangulation,
    		  	  	  	  	  	  	  	  	  	  	  	  	  	  cellwise_errors,
																  VectorTools::L2_norm);

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature,
                                         VectorTools::L2_norm,
                                         &pressure_mask);
//      const double p_l2_norm = cellwise_norms.norm_sqr();
      const double p_l2_norm = VectorTools::compute_global_error(triangulation,
    		  	  	  	  	  	  	  	  	  	  	  	  	  	  cellwise_norms,
																  VectorTools::L2_norm);
      // L2 in time error
      err.l2_l2_errors[1] += prm.time_step*p_l2_error*p_l2_error;
      err.l2_l2_norms[1] += prm.time_step*p_l2_norm*p_l2_norm;

      //calculating the error in DG norm (jumps L2 and final time L2)
       FE_DGQ<dim> fe_dgq_2d(0);
       DoFHandler<dim> dof_handler_2d(triangulation);
       dof_handler_2d.distribute_dofs(fe_dgq_2d);
       MappingQGeneric<dim> mapping_generic(1);
       DoFTools::map_dofs_to_support_points(mapping_generic,dof_handler_2d,pressure_support_points);
       pressure_function.value_list(pressure_support_points, projected_pressure_values);
       for(unsigned int dum_i=0; dum_i<n_pressure; dum_i++)
    	   pressure_projection.block(1)[dum_i]= projected_pressure_values[dum_i];
       dof_handler_2d.clear();

       if(time_level!=0) //calculating the jump error
		   {
           for(unsigned int dum_i=0; dum_i<n_pressure; dum_i++)
        	   old_solution_for_jump.block(1)[dum_i]+= pressure_projection.block(1)[dum_i]-solution.block(1)[dum_i]
											           -old_pressure_projection.block(1)[dum_i];
		  VectorTools::integrate_difference (dof_handler, old_solution_for_jump, zero_function,
											 cellwise_errors, quadrature,
											 VectorTools::L2_norm,
											 &pressure_mask);
		  const double p_l2_jump = VectorTools::compute_global_error(triangulation,
																	  cellwise_errors,
																	  VectorTools::L2_norm);

	//    	    err.linf_l2_errors[1] += compute_jump_error();
			   err.linf_l2_errors[1]+= p_l2_jump*p_l2_jump;
		   }
       if(std::fabs(prm.time-prm.final_time) < 1.0e-12)//adding L2 error at final step.

       {
    	    err.linf_l2_errors[1]+= p_l2_error*p_l2_error;
    	    err.linf_l2_norms[1] += p_l2_norm*p_l2_norm;
       }
       old_pressure_projection = pressure_projection;
       //end of caluclating the erro in DG norm


      // Velocity L2 error and norm
      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_errors, quadrature,
                                         VectorTools::L2_norm,
                                         &velocity_mask);
//      double u_l2_error = cellwise_errors.norm_sqr();
      const double u_l2_error = VectorTools::compute_global_error(triangulation,
    		  	  	  	  	  	  	  	  	  	  	  	  	  	  cellwise_errors,
																  VectorTools::L2_norm);

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_norms, quadrature,
                                         VectorTools::L2_norm,
                                         &velocity_mask);

      const double u_l2_norm = VectorTools::compute_global_error(triangulation,
    		  	  	  	  	  	  	  	  	  	  	  	  	  	  cellwise_norms,
																  VectorTools::L2_norm);

      // following is actually calculating H_div norm for velocity
      err.l2_l2_errors[0] +=u_l2_error*u_l2_error;
      err.l2_l2_norms[0] += u_l2_norm*u_l2_norm;
      double total_time = prm.time_step * prm.num_time_steps;
//      {

      double l_int_error_darcy=1, l_int_norm_darcy=1;;


      // On the last time step compute actual errors
      if(std::fabs(prm.time-total_time) < 1.0e-12)
      {
        // Assemble convergence table
    	          if (mortar_flag) //finding the convergence of lambda.
    	          {
     	              err.l2_l2_errors[2]= compute_interface_error_l2();
    	              interface_fe_function_mortar = 0;
    	              err.l2_l2_norms[2] = compute_interface_error_l2();
    	          }

        double send_buf_num[7] = {err.l2_l2_errors[0],
                                   err.velocity_stress_l2_div_errors[0],
                                   err.l2_l2_errors[1],
                                   err.pressure_disp_l2_midcell_errors[0],
                                   err.linf_l2_errors[1],
								   l_int_error_darcy,
								   err.l2_l2_errors[2]};

        double send_buf_den[7] = {err.l2_l2_norms[0],
                                   err.velocity_stress_l2_div_norms[0],
                                   err.l2_l2_norms[1],
                                   err.pressure_disp_l2_midcell_norms[0],
                                   err.linf_l2_norms[1],
								   l_int_norm_darcy,
								   err.l2_l2_norms[2]};

        double recv_buf_num[7] = {0,0,0,0,0,0,0};
        double recv_buf_den[7] = {0,0,0,0,0,0,0};

        MPI_Reduce(&send_buf_num[0], &recv_buf_num[0], 7, MPI_DOUBLE, MPI_SUM, 0, mpi_communicator);
        MPI_Reduce(&send_buf_den[0], &recv_buf_den[0], 7, MPI_DOUBLE, MPI_SUM, 0, mpi_communicator);


        for (unsigned int i=0; i<7; ++i)
          if (i != 10  ){
            recv_buf_num[i] = sqrt(recv_buf_num[i])/sqrt(recv_buf_den[i]);
          }
        convergence_table.add_value("cycle", refinement_index);
        convergence_table.add_value("# GMRES", max_cg_iteration);
        convergence_table.add_value("Velocity,L2-L2", recv_buf_num[0]);
        convergence_table.add_value("Pressure,DG", recv_buf_num[4]);
        convergence_table.add_value("Pressure,L2-L2", recv_buf_num[2]);

        if (mortar_flag)
        {
        	convergence_table.add_value("Lambda,Darcy_L2", recv_buf_num[6]);


        }
      }
    }


    // MixedBiotProblemDD::output_results
    template <int dim>
    void DarcyVTProblem<dim>::output_results (const unsigned int cycle, const unsigned int refine)
	{
//	        TimerOutput::Scope t(computing_timer, "Output results");
	        unsigned int n_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
	        unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator);

	        /* From here disabling for longer runs:
	         */

	        if(cycle==total_refinements-1)
			 {
			  std::vector<std::string> solution_names;
			  switch(dim)
			  {
				case 2:
				  solution_names.push_back ("u1");
				  solution_names.push_back ("u2");
				  solution_names.push_back ("p");

				  break;

				case 3:
				  solution_names.push_back ("u1");
				  solution_names.push_back ("u2");
				  solution_names.push_back ("u3");
				  solution_names.push_back ("p");
				  break;

				default:
				Assert(false, ExcNotImplemented());
			  }


			  // The following lines are used to output the solution at each time steps: disabled now since we are only
			  //interested in space-time solutions as of now.

//			  // Components interpretation of the flow solution (vector - scalar)
//			  std::vector<DataComponentInterpretation::DataComponentInterpretation>
//			  data_component_interpretation (dim,
//							  DataComponentInterpretation::component_is_part_of_vector);
//			  data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
//
//			  DataOut<dim> data_out;
//			  data_out.attach_dof_handler (dof_handler);
//			  data_out.add_data_vector (solution, solution_names,
//										DataOut<dim>::type_dof_data,
//										data_component_interpretation);
//
//			  data_out.build_patches ();
//
//
//			  int tmp = prm.time/prm.time_step;
//			  std::ofstream output ("solution_d" + Utilities::to_string(dim) + "_p"+Utilities::to_string(this_mpi,4)+"-" + std::to_string(tmp)+".vtu");
//			  data_out.write_vtu (output);
	//	      //following lines create a file which paraview can use to link the subdomain results
	//	            if (this_mpi == 0)
	//	              {
	//	                std::vector<std::string> filenames;
	//	                for (unsigned int i=0;
	//	                     i<Utilities::MPI::n_mpi_processes(mpi_communicator);
	//	                     ++i)
	//	                  filenames.push_back ("solution_d" + Utilities::to_string(dim) + "_p"+Utilities::to_string(i,4)+"-" + std::to_string(tmp)+".vtu");
	//
	//	                std::ofstream master_output (("solution_d" + Utilities::to_string(dim) + "-" + std::to_string(tmp) +
	//	                                              ".pvtu").c_str());
	//	                data_out.write_pvtu_record (master_output, filenames);
	//	              }

			 /* end of commenting out for disabling vtu outputs*/


			   if(std::fabs(prm.time-prm.final_time)<1.0e-12){ //outputting the 3d space-time solution:
				  std::vector<std::string> solution_names_st;
				  switch(dim)
				  {
					case 2:
					  solution_names_st.push_back ("u1");
					  solution_names_st.push_back ("u2");
					  solution_names_st.push_back ("u3");
					  solution_names_st.push_back ("p");
					  break;

					default:
					Assert(false, ExcNotImplemented());
				  }


				  // Components interpretation of the flow solution (vector - scalar)
				  std::vector<DataComponentInterpretation::DataComponentInterpretation>
				  data_component_interpretation_st (dim+1,
								  DataComponentInterpretation::component_is_part_of_vector);
				  data_component_interpretation_st.push_back(DataComponentInterpretation::component_is_scalar);

				  DataOut<dim+1> data_out_2;
				  data_out_2.attach_dof_handler (dof_handler_st);
				  data_out_2.add_data_vector (solution_st, solution_names_st,
											DataOut<dim+1>::type_dof_data,
											data_component_interpretation_st);

				  data_out_2.build_patches ();
				  std::ofstream output_st ("st_solution_d" + Utilities::to_string(dim+1) + "_p"+Utilities::to_string(this_mpi,4) +".vtu");
				  data_out_2.write_vtu (output_st);
						  //following lines create a file which paraview can use to link the subdomain results
								if (this_mpi == 0)
								  {
									std::vector<std::string> filenames_st;
									for (unsigned int i=0;
										 i<Utilities::MPI::n_mpi_processes(mpi_communicator);
										 ++i)
									  filenames_st.push_back ("st_solution_d" + Utilities::to_string(dim+1) + "_p"+Utilities::to_string(i,4)+".vtu");

									std::ofstream master_output_st (("st_solution_d" + Utilities::to_string(dim+1)  + ".pvtu").c_str());
									data_out_2.write_pvtu_record (master_output_st, filenames_st);
								  }
	//
			  }  //end of outputting 3d space-time solution.


	        }//end of outputting plots at the final refinement.
	      double total_time = prm.time_step * prm.num_time_steps;
	      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && refinement_index == refine-1 && std::fabs(prm.time-total_time)<1.0e-12){
	        convergence_table.set_precision("Velocity,L2-L2", 3);
	        convergence_table.set_precision("Pressure,DG", 3);
	        convergence_table.set_precision("Pressure,L2-L2", 3);

	        convergence_table.set_scientific("Velocity,L2-L2", true);
	        convergence_table.set_scientific("Pressure,DG", true);
	        convergence_table.set_scientific("Pressure,L2-L2", true);

	        convergence_table.set_tex_caption("# GMRES", "\\# gmres");
	        convergence_table.set_tex_caption("Velocity,L2-L2", "$ \\|z - z_h\\|_{L^{2}(L^2)} $");
	        convergence_table.set_tex_caption("Pressure,DG", "$ \\|p - p_h\\|_{DG} $");
	        convergence_table.set_tex_caption("Pressure,L2-L2", "$ \\|p - p_h\\|_{L^{2}(L^2)} $");

	        convergence_table.evaluate_convergence_rates("# GMRES", ConvergenceTable::reduction_rate_log2);
	        convergence_table.evaluate_convergence_rates("Velocity,L2-L2", ConvergenceTable::reduction_rate_log2);
	        convergence_table.evaluate_convergence_rates("Pressure,DG", ConvergenceTable::reduction_rate_log2);
	        convergence_table.evaluate_convergence_rates("Pressure,L2-L2", ConvergenceTable::reduction_rate_log2);


	        if (mortar_flag)
	        {
	          convergence_table.set_precision("Lambda,Darcy_L2", 3);
	          convergence_table.set_scientific("Lambda,Darcy_L2", true);
	          convergence_table.set_tex_caption("Lambda,Darcy_L2", "$ \\|p - \\lambda_p_H\\|_{L^{2}(L^2)} $");
	          convergence_table.evaluate_convergence_rates("Lambda,Darcy_L2", ConvergenceTable::reduction_rate_log2);

	        }

	        std::ofstream error_table_file("error" + std::to_string(Utilities::MPI::n_mpi_processes(mpi_communicator)) + "domains.tex");

	        pcout << std::endl;
	        convergence_table.write_text(std::cout);
	        convergence_table.write_tex(error_table_file);
	      }
	    }



    template <int dim>
    void DarcyVTProblem<dim>::reset_mortars()
    {
        triangulation.clear();
        dof_handler.clear();
        convergence_table.clear();
        faces_on_interface.clear();
        faces_on_interface_mortar.clear();
        interface_dofs.clear();
        interface_dofs_st.clear();
        interface_dofs_subd.clear();
        face_dofs_subdom.clear();
        face_dofs_st.clear();
        interface_fe_function_subdom = 0;
        interface_fe_function_st=0;

        if (mortar_flag)
        {
            triangulation_mortar.clear();
            triangulation_st.clear();
        }

        dof_handler_mortar.clear();
        dof_handler_st.clear();
    }

    // MixedBiotProblemDD::run
    template <int dim>
    void DarcyVTProblem<dim>::run (const unsigned int refine,
											 const std::vector<std::vector<int>> &reps_st,
                                             double tol,
                                             unsigned int maxiter,
                                             unsigned int quad_degree)
    {
        tolerance = tol;
        qdegree = quad_degree;
        total_refinements = refine;


        const unsigned int this_mpi = Utilities::MPI::this_mpi_process(mpi_communicator);
        const unsigned int n_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
        pcout<<"\n\n Total number of processes is "<<n_processes<<"\n\n";

        Assert(reps_st[0].size() == dim+1, ExcDimensionMismatch(reps_st[0].size(), dim));



        std::vector<std::vector<unsigned int>> reps_local(reps_st.size()), reps_st_local(reps_st.size()); //local copy of mesh partition information.
        for(int i=0; i<reps_st_local.size(); i++)
        {
        	reps_local[i].resize(2);
        	reps_st_local[i].resize(3);

        	reps_st_local[i][0] = reps_st[i][0] ;
        	reps_st_local[i][1] = reps_st[i][1] ;
        	reps_st_local[i][2] = reps_st[i][2] ;

        	reps_local[i][0] = reps_st_local[i][0] ;
        	reps_local[i][1] = reps_st_local[i][1] ;
        }

        if (mortar_flag)
        {
        	pcout<<"number of processors is "<<n_processes<<std::endl;
            Assert(n_processes > 1, ExcMessage("Mortar MFEM is impossible with 1 subdomain"));
            Assert(reps_st.size() >= n_processes + 1, ExcMessage("Some of the mesh parameters were not provided"));
        }

        for (refinement_index=0; refinement_index<total_refinements; ++refinement_index)
        {
            cg_iteration = 0;
            interface_dofs.clear();
            interface_dofs_st.clear();
            interface_dofs_subd.clear();
            face_dofs_subdom.clear();
            face_dofs_st.clear();

            if (refinement_index == 0)
            {
                //finding the num_time_steps and time_step_size using final_time and number of time steps required.
                prm.num_time_steps = reps_st_local[this_mpi][2];
                prm.time_step = prm.final_time/double(prm.num_time_steps);
                pcout<<"Final time= "<<prm.final_time<<"\n";
                pcout<<"number of time_steps for subdomain is: "<<prm.num_time_steps<<"\n";

                // Partitioning into subdomains (simple bricks)
                find_divisors<dim>(n_processes, n_domains);

                // Dimensions of the domain (unit hypercube)
                std::vector<double> subdomain_dimensions(dim);
                for (unsigned int d=0; d<dim; ++d)
                    subdomain_dimensions[d] = 1.0/double(n_domains[d]);

                get_subdomain_coordinates(this_mpi, n_domains, subdomain_dimensions, p1, p2);
                //corners of the space time sub-domain.
                p1_st = {p1[0],p1[1],0}, p2_st={p2[0],p2[1],prm.final_time};

                if (mortar_flag){
                    GridGenerator::subdivided_hyper_rectangle(triangulation, reps_local[this_mpi], p1, p2);
                    GridGenerator::subdivided_hyper_rectangle(triangulation_st, reps_st_local[this_mpi], p1_st, p2_st);

                    GridGenerator::subdivided_hyper_rectangle(triangulation_mortar, reps_st_local[n_processes], p1_st, p2_st);
                    pcout << "Mortar mesh has " << triangulation_mortar.n_active_cells() << " cells" << std::endl;
                }
                else
                {
                    GridGenerator::subdivided_hyper_rectangle(triangulation, reps_local[0], p1, p2);
                    if (this_mpi == 0 || this_mpi == 3)
                      GridTools::distort_random (0.1*(1+this_mpi), triangulation, true);
                }

            }
            else
            {

                if (mortar_flag == 0)
                    triangulation.refine_global(1);
                else  //(if there is mortar flag)
                {
                	triangulation.clear();
                	triangulation_st.clear();
                	triangulation_mortar.clear();
                    for(unsigned int dum_i=0; dum_i<reps_st_local.size()-1;dum_i++){ //refining space-time meshes.
                        	reps_st_local[dum_i][0]*=2;
                        	reps_st_local[dum_i][1]*=2;
                        	reps_st_local[dum_i][2]*=2;
                    }
                    //refining mortar mesh
                    if(mortar_degree==1)
                    {
                    	reps_st_local[reps_st_local.size()-1][0]*=2;
                    	reps_st_local[reps_st_local.size()-1][1]*=2;
                    	reps_st_local[reps_st_local.size()-1][2]*=2;
                    }
                    else if(refinement_index!=0 && refinement_index%2==0) //refining mortar mesh only every other time to get H=sqrt(h)
                    {
                    	reps_st_local[reps_st_local.size()-1][0]*=2;
                    	reps_st_local[reps_st_local.size()-1][1]*=2;
                    	reps_st_local[reps_st_local.size()-1][2]*=2;

                    }
                    for(unsigned int dum_i=0; dum_i<reps_local.size();dum_i++){ //refining space  meshes.
                    	reps_local[dum_i][0]= reps_st_local[dum_i][0];
                    	reps_local[dum_i][1]= reps_st_local[dum_i][1];
                    }

                    //finding the num_time_steps and time_step_size using final_time and number of time steps required.
                    prm.num_time_steps = reps_st_local[this_mpi][2];
                    prm.time_step = prm.final_time/double(prm.num_time_steps);
                    pcout<<"Final time= "<<prm.final_time<<"\n";
                    pcout<<"number of time_steps for subdomain is: "<<prm.num_time_steps<<"\n";

                    GridGenerator::subdivided_hyper_rectangle(triangulation, reps_local[this_mpi], p1, p2);
                    GridGenerator::subdivided_hyper_rectangle(triangulation_st, reps_st_local[this_mpi], p1_st, p2_st);
                    GridGenerator::subdivided_hyper_rectangle(triangulation_mortar, reps_st_local[n_processes], p1_st, p2_st);
                    pcout << "Mortar mesh has " << triangulation_mortar.n_active_cells() << " cells" << std::endl;

                }

            }
            pcout << "Making grid and DOFs...\n";
            make_grid_and_dofs();

            pcout << "Projecting the initial conditions...\n";
            {
              InitialCondition<dim> ic;

              ConstraintMatrix constraints;
              constraints.clear();
              constraints.close();
              VectorTools::project (dof_handler,
                                    constraints,
                                    QGauss<dim>(degree+5),
                                    ic,
                                    initialc_solution);

              solution = initialc_solution;
              output_results(refinement_index,refine);
            }

            pcout << "Assembling system..." << "\n";
            	assemble_system ();

            if (Utilities::MPI::n_mpi_processes(mpi_communicator) != 1)
            {
				get_interface_dofs();
				get_interface_dofs_st();
            }

            solve_darcy_vt(maxiter);
            max_cg_iteration=0;

            set_current_errors_to_zero();
            prm.time = 0.0;

            computing_timer.print_summary();
            computing_timer.reset();

        }

        reset_mortars();
    }


    template class DarcyVTProblem<2>;
//    template class DarcyVTProblem<3>;
}
