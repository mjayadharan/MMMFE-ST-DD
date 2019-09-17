/* ---------------------------------------------------------------------
 * Declaration of MixedBiotProblemDD class template: see source files for more details
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
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iostream>
#include <cstdlib>

#include "projector.h"



namespace dd_biot
{
    using namespace dealii;

    struct BiotParameters
    {
      BiotParameters(const double dt, const unsigned int nt,
                     const double c = 1.0, const double a = 1.0)
              :
              time(0.0),
              time_step(dt),
              num_time_steps(nt),
              c_0(c),
              alpha(a)
      {}

      mutable double time;
      const double time_step;
      const unsigned int num_time_steps;
      const double c_0;
      const double alpha;
    };

    struct BiotErrors
    {
      BiotErrors()
              :
        l2_l2_norms(5,0),
        l2_l2_errors(5,0),
        linf_l2_norms(5,0),
        linf_l2_errors(5,0),
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
    template<int dim>
    class MixedBiotProblemDD
    {
    public:
        MixedBiotProblemDD(const unsigned int degree, const BiotParameters& bprm, const unsigned int mortar_flag = 0,
                           const unsigned int mortar_degree = 0, unsigned int split_flag=0);

        void run(const unsigned int refine, const std::vector <std::vector<unsigned int>> &reps, double tol,
                 unsigned int maxiter, unsigned int quad_degree = 11);

    private:
        MPI_Comm mpi_communicator;
        MPI_Status mpi_status;

        Projector::Projector <dim> P_coarse2fine;
        Projector::Projector <dim> P_fine2coarse;

        void make_grid_and_dofs();
        void assemble_system();
        void assemble_system_elast();
        void assemble_system_darcy();
        void get_interface_dofs();
        void get_interface_dofs_elast();
        void get_interface_dofs_darcy();
        void assemble_rhs_bar();
        void assemble_rhs_bar_elast();
        void assemble_rhs_bar_darcy();
        void assemble_rhs_star(FEFaceValues<dim> &fe_face_values);
        void assemble_rhs_star_elast(FEFaceValues<dim> &fe_face_values);
        void assemble_rhs_star_darcy(FEFaceValues<dim> &fe_face_values);

        void solve_bar();
        void solve_bar_elast();
        void solve_bar_darcy();

        void solve_star();
        void solve_star_elast();
        void solve_star_darcy();

        void solve_timestep(unsigned int maxiter);

        void compute_multiscale_basis();
        void local_cg(const unsigned int maxiter,  unsigned int split_order_flag=0); //split_order_flag=0 is Elasticity part, 1 is Darcy part
        void local_cg_elast(const unsigned int maxiter);
        void local_cg_darcy(const unsigned int maxiter);
        std::vector<double> compute_interface_error(); //return_vector[0] gives interface_error for elast part and return_vector[1] gives that of flow part.
        void compute_errors(const unsigned int cycle);
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
        const unsigned int split_flag; //monolithic: 0, drained split:1, fixed stress: 2
        unsigned int cg_iteration;
        unsigned int max_cg_iteration;
        unsigned int max_cg_iteration_darcy; //for Darcy CG in split
        double tolerance;
        unsigned int qdegree;


        // Neighbors and interface information
        std::vector<int> neighbors;
        std::vector<unsigned int> faces_on_interface;
        std::vector<unsigned int> faces_on_interface_mortar;
        std::vector <std::vector<unsigned int>> interface_dofs;
        std::vector <std::vector<unsigned int>> interface_dofs_elast;
        std::vector <std::vector<unsigned int>> interface_dofs_darcy;


        unsigned long n_stress;
        unsigned long n_disp;
        unsigned long n_rot;
        unsigned long n_flux;
        unsigned long n_pressure;
        unsigned long n_Elast;

        // Subdomain coordinates (assuming logically rectangular blocks)
        Point <dim> p1;
        Point <dim> p2;

        // Fine triangulation
        Triangulation <dim> triangulation;
        FESystem <dim> fe;
        FESystem <dim> fe_elast;
        FESystem <dim> fe_darcy;
        DoFHandler <dim> dof_handler;
        DoFHandler <dim> dof_handler_elast;
        DoFHandler <dim> dof_handler_darcy;

        // Mortar triangulation
        Triangulation <dim> triangulation_mortar;
        FESystem <dim> fe_mortar;
        DoFHandler <dim> dof_handler_mortar;

        // Star and bar problem data structures
        BlockSparsityPattern sparsity_pattern;
        BlockSparsityPattern sparsity_pattern_elast;
        BlockSparsityPattern sparsity_pattern_darcy;
        BlockSparseMatrix<double> system_matrix;
        BlockSparseMatrix<double> system_matrix_elast;
        BlockSparseMatrix<double> system_matrix_darcy;
        SparseDirectUMFPACK A_direct;
        SparseDirectUMFPACK A_direct_elast;
        SparseDirectUMFPACK A_direct_darcy;

        BlockVector<double> solution_bar;
        BlockVector<double> solution_bar_elast;
        BlockVector<double> solution_bar_darcy;

        BlockVector<double> solution_star;
        BlockVector<double> solution_star_elast;
        BlockVector<double> solution_star_darcy;

        BlockVector<double> solution;
        BlockVector<double> solution_elast;
        BlockVector<double> solution_darcy;

        BlockVector<double> old_solution;
        BlockVector<double> older_solution; //solution in previous to previous timestep(used in fixed stress split)
        BlockVector<double> intermediate_solution;
//        BlockVector<double> intermediate_solution_old; //solution in previous time step of intermediate solution(used only in fixed stress)

        BlockVector<double> system_rhs_bar;
        BlockVector<double> system_rhs_bar_elast;
        BlockVector<double> system_rhs_bar_darcy;

        BlockVector<double> system_rhs_star;
        BlockVector<double> system_rhs_star_elast;
        BlockVector<double> system_rhs_star_darcy;

        BlockVector<double> interface_fe_function;

        std::vector<std::vector<double>> lambda_guess;
        std::vector<std::vector<double>> lambda_guess_elast;
        std::vector<std::vector<double>> lambda_guess_darcy;
        std::vector<std::vector<double>> Alambda_guess;
        std::vector<std::vector<double>> Alambda_guess_elast;
        std::vector<std::vector<double>> Alambda_guess_darcy;

        // Mortar data structures
        BlockVector<double> interface_fe_function_mortar;
        BlockVector<double> solution_bar_mortar;
        BlockVector<double> solution_star_mortar;
        std::vector <BlockVector<double>> multiscale_basis;

        // Output extra
        ConditionalOStream pcout;
        ConvergenceTable convergence_table;
        TimerOutput computing_timer;
    };
}

#endif //ELASTICITY_MFEDD_ELASTICITY_MFEDD_H
