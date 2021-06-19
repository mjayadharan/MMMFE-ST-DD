/* ---------------------------------------------------------------------
 * Declaration of DarcyVT class template: see source files for more details
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan, Eldar Khattatov, University of Pittsburgh, 2018-2019
 */

#ifndef ELASTICITY_MFEDD_ELASTICITY_MFEDD_H
#define ELASTICITY_MFEDD_ELASTICITY_MFEDD_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iostream>
#include <cstdlib>

#include "projector.h"



namespace vt_darcy
{
    using namespace dealii;

    struct BiotParameters
    {
      BiotParameters(const double dt, const unsigned int nt, const double f_time,
                     const double c = 1.0, const double a = 1.0, const double coe_a_in=1.0)
              :
              time(0.0),
	      time_step(dt),
	      final_time(f_time),
	      num_time_steps(nt),
	      c_0(c),
	      alpha(a),
		  coe_a(coe_a_in)
	{}

      mutable double time;
      double time_step;
      const double final_time;
      unsigned int num_time_steps;
      const double c_0;
      const double alpha;
      const double coe_a; //coefficient for controlling variation in time
    };

    struct BiotErrors
    {
      BiotErrors()
              :
        l2_l2_norms(3,0),
        l2_l2_errors(3,0),
        linf_l2_norms(2,0),
        linf_l2_errors(2,0),
        velocity_stress_l2_div_norms(2,0),
        velocity_stress_l2_div_errors(2,0),
        velocity_stress_linf_div_norms(2,0),
        velocity_stress_linf_div_errors(2,0),
        pressure_disp_l2_midcell_norms(2,0),
        pressure_disp_l2_midcell_errors(2,0),
        pressure_disp_linf_midcell_norms(2,0),
        pressure_disp_linf_midcell_errors(2,0)
      {}

      std::vector<double> l2_l2_norms;
      std::vector<double> l2_l2_errors;

      std::vector<double> linf_l2_norms;
      std::vector<double> linf_l2_errors;//In case of stress, this is actually linf_h_div_error

      std::vector<double> velocity_stress_l2_div_norms;
      std::vector<double> velocity_stress_l2_div_errors;

      std::vector<double> velocity_stress_linf_div_norms;
      std::vector<double> velocity_stress_linf_div_errors;

      std::vector<double> pressure_disp_l2_midcell_norms;
      std::vector<double> pressure_disp_l2_midcell_errors;

      std::vector<double> pressure_disp_linf_midcell_norms;
      std::vector<double> pressure_disp_linf_midcell_errors;
    };

    // Mixed Biot Domain Decomposition class template
    template<int dim=2>
    class DarcyVTProblem
    {
    public:
        DarcyVTProblem(const unsigned int degree, const BiotParameters& bprm, const unsigned int mortar_flag = 0,
                           const unsigned int mortar_degree = 0, std::vector<char> bc_condition_vect={'D','D','D','D'},
						   std::vector<double>bc_const_functs={0.,0.,0.,0.}, const bool is_manufact_soln = true,
						   const bool need_each_time_step_plot=false);

        void run(const unsigned int refine,
        		 const std::vector <std::vector<int>> &reps_st,
				 const std::vector <std::vector<int>> &reps_st_mortar, double tol,
                 unsigned int maxiter, unsigned int quad_degree=3);

    private:
        MPI_Comm mpi_communicator;
        MPI_Status mpi_status;

        Projector::Projector <dim+1> P_coarse2fine;
        Projector::Projector <dim+1> P_fine2coarse;

        void make_grid_and_dofs();
        void assemble_system();
        void get_interface_dofs();
        void get_interface_dofs_st(); //get inteface dofs from the space time interface.
        void assemble_rhs_bar();
//        void assemble_rhs_star(FEFaceValues<dim> &fe_face_values);
        void assemble_rhs_star();
        void solve_bar();


        void solve_star();
        void solve_timestep(int star_bar_flag, unsigned int time_level); //star_bar_flag == 0:solving bar problem, 1: solving star problem, 2: solving star problem at end after gmres converges, compute final solution, error and output.
        void solve_darcy_vt(unsigned int maxiter);

        void compute_multiscale_basis();
        std::vector<double> compute_interface_error_dh(); //return_vector[0] gives interface_error for elast part and return_vector[1] gives that of flow part.
        double compute_interface_error_l2();
        double compute_jump_error(); //return L2 error of jump of pressure across time levels.
        void compute_errors(const unsigned int refinement_index, unsigned int time_level);
        void output_results(const unsigned int cycle, const unsigned int refine, const unsigned int time_level);

        void set_current_errors_to_zero();
        void reset_mortars();
        //For implementing GMRES
        void
    	givens_rotation(double v1, double v2, double &cs, double &sn);

        void
    	apply_givens_rotation(std::vector<double> &h, std::vector<double> &cs, std::vector<double> &sn,
        							unsigned int k_iteration);

        void
    	back_solve(std::vector<std::vector<double>> H, std::vector<double> beta, std::vector<double> &y, unsigned int k_iteration);

        void
        local_gmres(const unsigned int maxiter);
        double vect_norm(std::vector<double> v);
        //distribute solution vectors between 2-d space and 3-d space-time subdomain mesh.
        void st_to_subdom_distribute (BlockVector<double> &vector_st,
        							  BlockVector<double> &vector_subdom, unsigned int &time_level, double scale_factor);
        void subdom_to_st_distribute (BlockVector<double> &vector_st,
               						  BlockVector<double> &vector_subdom, unsigned int &time_level, double scale_factor);
        //distribute local to global solution(now only the pressure part).
        void final_solution_transfer (BlockVector<double> &solution_st,
               						  BlockVector<double> &solution_subdom, unsigned int &time_level, double scale_factor);

    
        // Number of subdomains in the computational domain
        std::vector<unsigned int> n_domains;

        // Physical parameters
        BiotParameters prm;
        BiotErrors err;

        //Boundary condition vector: D means Dirichlet bc, N means Neumann bc starting from left, bottom, right, top respectively.
        std::vector<char> bc_condition_vect;  //= {D, D, D, N} = {left, bottom, right} has Dirichlet boundayr condition and
											//bottom has neumann bc( essential)
        std::vector<double> bc_const_functs; //vector containing v.n for Neumann boundary condition,
        									//in case its a constant fuction(this is assumed by default).
        //manufactured_sol
        const bool is_manufact_solution;
        //flag to check whether the computer solution needs plotting at each time step.
        const bool need_each_time_step_plot;

        std::vector<int> dir_bc_ids, nm_bc_ids;

        // FE degree and DD parameters
        const unsigned int degree;
        const unsigned int mortar_degree;
        const unsigned int mortar_flag;

        unsigned int gmres_iteration;
        double grid_diameter;
        unsigned int cg_iteration;
        unsigned int max_cg_iteration;
        double tolerance;
        unsigned int qdegree;
        unsigned int refinement_index;
        unsigned int total_refinements;


        // Neighbors and interface information
        std::vector<int> neighbors;
        std::vector<unsigned int> faces_on_interface;
        std::vector<unsigned int> faces_on_interface_mortar;
        std::vector<unsigned int> faces_on_interface_st;
        std::vector <std::vector<unsigned int>> interface_dofs; //dofs on the mortar space time interface.
        std::vector <std::vector<unsigned int>> interface_dofs_subd; //dofs on 2d-subdomain interface.
        std::vector <std::vector<unsigned int>> interface_dofs_st; //for 3d space-tiem subdomain mesh.
        std::vector <std::vector<unsigned int>> face_dofs_st; //RT0 3d dofs living on the faces of a cell: for transfering solution to space-time mesh.
        std::vector <std::vector<unsigned int>> face_dofs_subdom; // //RT0 2d dofs living on the faces of a cell: for transfering solution to space-time mesh.



        unsigned long n_flux;
        unsigned long n_pressure;

        unsigned long n_flux_st;
        unsigned long n_pressure_st;

        // Subdomain coordinates (assuming logically rectangular blocks)
        Point <dim> p1;
        Point <dim> p2;

        //  space-time grid diagonal coordinates (assuming logically rectangular blocks)
         Point <dim+1> p1_st;
         Point <dim+1> p2_st;


        // Fine triangulation
        Triangulation <dim> triangulation;
        FESystem <dim> fe;
        DoFHandler <dim> dof_handler;

        //3d Space time triangulation for subdomain.
        Triangulation<dim+1> triangulation_st;
        FESystem<dim+1> fe_st;
        DoFHandler<dim+1> dof_handler_st;

        // Mortar triangulation
        Triangulation <dim+1> triangulation_mortar;
        FESystem <dim+1> fe_mortar;
        DoFHandler <dim+1> dof_handler_mortar;

        // Star and bar problem data structures
        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        SparseDirectUMFPACK A_direct;

        BlockVector<double> solution_bar;
        BlockVector<double> solution_star;
        BlockVector<double> solution;
        BlockVector<double> solution_st; //for the 3d space-time solution

        BlockVector<double> old_solution;
        BlockVector<double> old_solution_for_jump;  //storing old solution to calculate the jump in pressure error.
        BlockVector<double> initialc_solution;

        BlockVector<double> pressure_projection; //projection of exact pressure to piecewise constant space.
        BlockVector<double> old_pressure_projection; //pressure_projection from previous time step.

        BlockVector<double> system_rhs_bar;
        BlockVector<double> system_rhs_bar_bc; //used in assemble_system: required for essential(Neumann bc)
        BlockVector<double> system_rhs_star;
//        BlockVector<double> interface_fe_function; //need to decide whether to keep this.
        BlockVector<double> interface_fe_function_subdom;

        //Constrain matrix for essential (Neumann) bc
        AffineConstraints<double> constraint_bc;
//        ConstraintMatrix constraint_bc;

        // Mortar data structures
        BlockVector<double> interface_fe_function_mortar;
        BlockVector<double> solution_bar_mortar;
        BlockVector<double> solution_star_mortar;
        std::vector <BlockVector<double>> multiscale_basis;

        // 3d Space-time data structures
        BlockVector<double> interface_fe_function_st;
        BlockVector<double> solution_bar_st;
        BlockVector<double> solution_star_st;
        std::vector<BlockVector<double>> solution_bar_collection;

        // Output extra
        ConditionalOStream pcout;
        ConvergenceTable convergence_table;
        TimerOutput computing_timer;
    };
}

#endif //ELASTICITY_MFEDD_ELASTICITY_MFEDD_H
