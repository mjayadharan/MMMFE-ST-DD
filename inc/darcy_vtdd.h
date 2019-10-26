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
                     const double c = 1.0, const double a = 1.0)
              :
              time(0.0),
              time_step(dt),
			  final_time(f_time),
              num_time_steps(nt),
              c_0(c),
              alpha(a)
      {}

      mutable double time;
      const double time_step;
      const unsigned int num_time_steps;
      const double final_time;
      const double c_0;
      const double alpha;
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
                           const unsigned int mortar_degree = 0);

        void run(const unsigned int refine, const std::vector <std::vector<unsigned int>> &reps,
        		 const std::vector <std::vector<unsigned int>> &reps_st, double tol,
                 unsigned int maxiter, unsigned int quad_degree = 11);

    private:
        MPI_Comm mpi_communicator;
        MPI_Status mpi_status;

        Projector::Projector <dim> P_coarse2fine;
        Projector::Projector <dim> P_fine2coarse;

        void make_grid_and_dofs();
        void assemble_system();
        void get_interface_dofs();
        void get_interface_dofs_st(); //get inteface dofs from the space time interface.
        void assemble_rhs_bar();
//        void assemble_rhs_star(FEFaceValues<dim> &fe_face_values);
        void assemble_rhs_star();
        void solve_bar();


        void solve_star();
        void solve_timestep(int star_bar_flag, int time_level); //star_bar_flag == 0:solving bar problem, 1: solving star problem, 3: solving bar problem at end after gmres converges, compute final solution, error and output.
        void solve_darcy_vt(unsigned int maxiter);

        void compute_multiscale_basis();
        std::vector<double> compute_interface_error_dh(); //return_vector[0] gives interface_error for elast part and return_vector[1] gives that of flow part.
        double compute_interface_error_l2();
        void compute_errors(const unsigned int refinement_index);
        void output_results(const unsigned int cycle, const unsigned int refine);

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
        //just to test the local_gmres algorithm
          void
      	testing_gmres(const unsigned int &maxiter);


        unsigned int       gmres_iteration;
        // Number of subdomains in the computational domain



        // Physical parameters
        const BiotParameters prm;
        BiotErrors err;

        double grid_diameter;

        // Number of subdomains in the computational domain
        std::vector<unsigned int> n_domains;

        // FE degree and DD parameters
        const unsigned int degree;
        const unsigned int mortar_degree;
        const unsigned int mortar_flag;
        unsigned int cg_iteration;
        unsigned int max_cg_iteration;
        double tolerance;
        unsigned int qdegree;
        unsigned int refinement_index;
        const unsigned int total_refinements;


        // Neighbors and interface information
        std::vector<int> neighbors;
        std::vector<unsigned int> faces_on_interface;
        std::vector<unsigned int> faces_on_interface_mortar;
        std::vector<unsigned int> faces_on_interface_st;
        std::vector <std::vector<unsigned int>> interface_dofs; //dofs on the mortar space time interface.
        std::vector <std::vector<unsigned int>> interface_dofs_subd; //dofs on 2d-subdomain interface.
        std::vector <std::vector<unsigned int>> interface_dofs_st; //for 3d space-tiem subdomain mesh.
//        std::vector<std::vector <std::vector<unsigned int>>> interface_dofs_st; //for 3d space-tiem subdomain mesh. first component corresponds to time_level.



        unsigned long n_flux;
        unsigned long n_pressure;

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

        BlockVector<double> old_solution;
        BlockVector<double> initialc_solution;

        BlockVector<double> system_rhs_bar;
        BlockVector<double> system_rhs_star;
        BlockVector<double> interface_fe_function; //need to decide whether to keep this.
        BlockVector<double> interface_fe_function_subdom;


//        std::vector<std::vector<double>> lambda_guess;
//        std::vector<std::vector<double>> Alambda_guess;

        // Mortar data structures
        BlockVector<double> interface_fe_function_mortar;
        BlockVector<double> solution_bar_mortar;
        BlockVector<double> solution_star_mortar;
        std::vector <BlockVector<double>> multiscale_basis;

        // 3d Space-time data structures
        BlockVector<double> interface_fe_function_st;
        BlockVector<double> solution_bar_st;
        BlockVector<double> solution_star_st;

        // Output extra
        ConditionalOStream pcout;
        ConvergenceTable convergence_table;
        TimerOutput computing_timer;
    };
}

#endif //ELASTICITY_MFEDD_ELASTICITY_MFEDD_H
