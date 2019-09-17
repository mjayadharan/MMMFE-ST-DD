/* ---------------------------------------------------------------------
 * Utilities:
 *  - Elasticity related - physical tensors and asymmetry operators
 *  - DD & mortar related - mesh subdivision, mortar projection
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2016 - 2017
 */
#ifndef ELASTICITY_MFEDD_UTILITIES_H
#define ELASTICITY_MFEDD_UTILITIES_H

#include "projector.h"

namespace dd_biot
{
    using namespace dealii;

    // Elasticity problem related utilities
    // create rank-2 tensor
    template <int dim>
    Tensor<2,dim> make_tensor(const std::vector<Tensor<1,dim> > &vec, Tensor<2, dim> &res)
    {
        for(int row=0;row<dim;++row)
            for(int col=0;col<dim;++col)
                res[row][col] = vec[row][col];

        return res;
    }

    // create asymmetry tensor
    template <int dim, int tensor_dim>
    void make_asymmetry_tensor(const std::vector<Tensor<1,dim> > &vec, Tensor<1, tensor_dim> &res)
    {
        Tensor<2,dim> mat;
        make_tensor(vec, mat);

        switch(dim)
        {
            case 2:
                res[0] = mat[0][1] - mat[1][0];
                break;
            case 3:
                res[0] = mat[2][1] - mat[1][2];
                res[1] = mat[0][2] - mat[2][0];
                res[2] = mat[1][0] - mat[0][1];
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    // create compliance tensor
    template <int dim>
    void compliance_tensor(const std::vector<Tensor<1,dim> > &vec, const double mu, const double lambda, Tensor<2,dim> &res)
    {
        double trace=0.;
        for (unsigned int i=0;i<dim;++i)
            trace += vec[i][i];

        for (unsigned int row=0;row<dim;++row)
            for (unsigned int col=0;col<dim;++col)
                if (row == col)
                    res[row][col] = 1/(2*mu)*(vec[row][col] - lambda/(2*mu + dim*lambda)*trace);
                else
                    res[row][col] = 1/(2*mu)*(vec[row][col]);
    }


    // Create compliance tensor for pressure variables
    template <int dim>
    void compliance_tensor_pressure(const double pres, const double mu, const double lambda, Tensor<2,dim> &res)
    {
      for(unsigned int i=0;i<dim;i++)
          res[i][i] = pres/(2*mu+dim*lambda);
    }

    // Domain decomposition related utilities
    // Function to get the true normal vector orientation (in deal.II they are all positive by default)
    double
    get_normal_direction (const unsigned int &face)
    {
        std::vector<double> normals(6);
        normals[0] = -1;
        normals[1] = 1;
        normals[2] = 1;
        normals[3] = -1;
        normals[4] = -1;
        normals[5] = 1;

        return normals[face];
    }

    // Split the number into two divisors closest to sqrt (in order to partition mesh into subdomains)
    void find_2_divisors (const unsigned int &n_proc, std::vector<unsigned int> &n_dom)
    {
        double tmp = floor(sqrt(n_proc));
        while(true)
        {
            if (fmod(double(n_proc), tmp) == 0)
            {
                n_dom[0] = n_proc/tmp;
                n_dom[1] = tmp;
                break;
            }
            else
            {
                tmp -= 1;
            }
        }
    }

    // Split number into three divisors (Try to make it recursive better)
    template <int dim>
    void
    find_divisors (const unsigned int &n_proc, std::vector<unsigned int> &n_dom)
    {
        Assert(dim == 2 || dim == 3, ExcNotImplemented());

        if (dim == 2)
            find_2_divisors(n_proc, n_dom);
        else if (dim == 3)
        {
            double tmp = floor(pow(n_proc,1.0/3.0));
            while(true)
            {
                if (fmod(static_cast<double>(n_proc), tmp) == 0)
                {
                    std::vector<unsigned int> two_divisors(2);
                    unsigned int two_proc = n_proc/tmp;
                    find_2_divisors(two_proc, two_divisors);

                    n_dom[0] = two_divisors[0];
                    n_dom[1] = two_divisors[1];
                    n_dom[2] = tmp;
                    break;
                }
                else
                {
                    tmp -= 1;
                }
            }
        }
    }

    // Compute the lower left and upper right coordinate of a block
    template <int dim>
    void
    get_subdomain_coordinates (const unsigned int &this_mpi, const std::vector<unsigned int> &n_doms, const std::vector<double> &dims, Point<dim> &p1, Point<dim> &p2)
    {
        switch (dim)
        {
            case 2:
                p1[0] = (this_mpi % n_doms[0])*dims[0];
                p1[1] = dims[1]*(floor(double(this_mpi)/double(n_doms[0])));
                p2[0] = (this_mpi % n_doms[0])*dims[0] + dims[0];
                p2[1] = dims[1]*(floor(double(this_mpi)/double(n_doms[0]))) + dims[1];
                break;

            case 3:
                p1[0] = (this_mpi % n_doms[0])*dims[0];
                p1[1] = dims[1]*(floor(double(this_mpi % (n_doms[0] * n_doms[1]))/double(n_doms[0])));
                p1[2] = dims[2]*(floor(double(this_mpi)/double(n_doms[0]*n_doms[1])));
                p2[0] = (this_mpi % n_doms[0])*dims[0] + dims[0];
                p2[1] = dims[1]*(floor(double(this_mpi % (n_doms[0] * n_doms[1]))/double(n_doms[0]))) + dims[1];
                p2[2] = dims[2]*(floor(double(this_mpi)/double(n_doms[0]*n_doms[1]))) + dims[2];
                break;

            default:
                Assert(false, ExcNotImplemented());
                break;
        }
    }

    // Find neighboring subdomains
    void
    find_neighbors (const int &dim, const unsigned int &this_mpi, const std::vector<unsigned int> &n_doms, std::vector<int> &neighbors)
    {
        Assert(neighbors.size() == 2.0 * dim, ExcDimensionMismatch(neighbors.size(), 2.0 * dim));

        switch (dim)
        {
            case 2:
                if (this_mpi % n_doms[0] == 0) // x = a line
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = this_mpi + 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = - 1;
                }
                else if ((this_mpi + 1) % n_doms[0] == 0 /*&& (this_mpi + 1) >= n_doms[0]*/ ) // x = b line
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = - 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = this_mpi - 1;
                }
                else
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = this_mpi + 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = this_mpi - 1;
                }

                break;

            case 3:
                if (this_mpi % (n_doms[0] * n_doms[1]) == 0) // corner at origin
                {
                    neighbors[0] = - 1;
                    neighbors[1] = this_mpi + 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                else if ((this_mpi - n_doms[0] + 1) % (n_doms[0]*n_doms[1]) == 0) // corner at x=a
                {
                    neighbors[0] = - 1;
                    neighbors[1] = - 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = this_mpi - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                else if ((this_mpi + n_doms[0]) % (n_doms[0] * n_doms[1]) == 0) // corner at y=a
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = this_mpi + 1;
                    neighbors[2] = - 1;
                    neighbors[3] = - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                else if ((this_mpi + 1) % (n_doms[0] * n_doms[1]) == 0) // corner at x=y=a
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = - 1;
                    neighbors[2] = - 1;
                    neighbors[3] = this_mpi - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                else if (this_mpi % n_doms[0] == 0) // plane x = a
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = this_mpi + 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                else if ((this_mpi + 1) % n_doms[0] == 0) // plane x = b
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = - 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = this_mpi - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                else if (this_mpi % (n_doms[0] * n_doms[1]) < n_doms[0]) // plane y = a
                {
                    neighbors[0] = - 1;
                    neighbors[1] = this_mpi + 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = this_mpi - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                else if (this_mpi % (n_doms[0] * n_doms[1]) > n_doms[0] * n_doms[1] - n_doms[0]) // plane y = b
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = this_mpi + 1;
                    neighbors[2] = - 1;
                    neighbors[3] = this_mpi - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                else
                {
                    neighbors[0] = this_mpi - n_doms[0];
                    neighbors[1] = this_mpi + 1;
                    neighbors[2] = this_mpi + n_doms[0];
                    neighbors[3] = this_mpi - 1;
                    neighbors[4] = this_mpi - n_doms[0] * n_doms[1];
                    neighbors[5] = this_mpi + n_doms[0] * n_doms[1];
                }
                break;

            default:
                Assert(false, ExcNotImplemented());
                break;
        }


        for (unsigned int i=0; i<neighbors.size(); ++i)
        {
            if (neighbors[i] < 0)
                neighbors[i] = -1;
            else if ( dim == 2 && neighbors[i] >= int(n_doms[0] * n_doms[1]) )
                neighbors[i] = -1;
            else if ( dim == 3 && neighbors[i] >= int(n_doms[0] * n_doms[1] * n_doms[2]) )
                neighbors[i] = -1;
        }
    }

    template <int dim, typename Number>
    void
    mark_interface_faces (const Triangulation<dim> &tria, const std::vector<int> &neighbors, const Point<dim> &p1, const Point<dim> &p2, std::vector<Number> &faces_per_interface)
    {
        Assert(faces_per_interface.size() == neighbors.size(), ExcDimensionMismatch(faces_per_interface.size(), neighbors.size()));

        // Label boundaries
        // On unit hypercube for example:
        //  1 - plane y=0, 2 - plane x=1, 3 - plane y=1, 4 - plane x=0, 5 - plane z=0, 6 - plane z=1 for interfaces,
        //  0 - for outside

        typename Triangulation<dim>::cell_iterator cell, endc;
        cell = tria.begin_active (),
                endc = tria.end();

        for (; cell!=endc; ++cell)
            for (unsigned int face_number=0;
                 face_number<GeometryInfo<dim>::faces_per_cell;
                 ++face_number)
            {
                // If left boundary of the subdomain in 2d or
                if ( std::fabs(cell->face(face_number)->center()(0) - p1[0]) < 1e-12 )
                {
                    // If it is outside boundary (no neighbor) or interface
                    if (neighbors[3] < 0 )
                        cell->face(face_number)->set_boundary_id (0);
                    else
                    {
                        cell->face(face_number)->set_boundary_id (4);
                        faces_per_interface[3] += 1;
                    }
                }
                    // If bottom boundary of the subdomain
                else if ( std::fabs(cell->face(face_number)->center()(1) - p1[1]) < 1e-12 )
                {
                    // If it is outside boundary (no neighbor) or interface
                    if (neighbors[0] < 0 )
                        cell->face(face_number)->set_boundary_id (0);
                    else
                    {
                        cell->face(face_number)->set_boundary_id (1);
                        faces_per_interface[0] += 1;
                    }
                }
                    // If right boundary of the subdomain
                else if ( std::fabs(cell->face(face_number)->center()(0) - p2[0]) < 1e-12 )
                {
                    // If it is outside boundary (no neighbor) or interface
                    if (neighbors[1] < 0 )
                        cell->face(face_number)->set_boundary_id (0);
                    else
                    {
                        cell->face(face_number)->set_boundary_id (2);
                        faces_per_interface[1] += 1;
                    }
                }
                    // If top boundary of the subdomain
                else if ( std::fabs(cell->face(face_number)->center()(1) - p2[1]) < 1e-12 )
                {
                    // If it is outside boundary (no neighbor) or interface
                    if (neighbors[2] < 0 )
                        cell->face(face_number)->set_boundary_id (0);
                    else
                    {
                        cell->face(face_number)->set_boundary_id (3);
                        faces_per_interface[2] += 1;
                    }
                }
                else if ( dim == 3 && std::fabs(cell->face(face_number)->center()(2) - p1[2]) < 1e-12 )
                {
                    // If it is outside boundary (no neighbor) or interface
                    if (neighbors[4] < 0 )
                        cell->face(face_number)->set_boundary_id (0);
                    else
                    {
                        cell->face(face_number)->set_boundary_id (5);
                        faces_per_interface[4] += 1;
                    }
                }
                else if ( dim == 3 && std::fabs(cell->face(face_number)->center()(2) - p2[2]) < 1e-12 )
                {
                    // If it is outside boundary (no neighbor) or interface
                    if (neighbors[5] < 0 )
                        cell->face(face_number)->set_boundary_id (0);
                    else
                    {
                        cell->face(face_number)->set_boundary_id (6);
                        faces_per_interface[5] += 1;
                    }
                }
            }
    }

    template <int dim>
    void
    project_mortar (Projector::Projector<dim> &proj,
                    const DoFHandler<dim>     &dof1,
                    BlockVector<double>       &in_vec,
                    const Quadrature<dim-1>   &quad,
                    ConstraintMatrix          &constraints,
                    const std::vector<int>    &neighbors,
                    const DoFHandler<dim>     &dof2,
                    BlockVector<double>       &out_vec)
    {
        out_vec = 0;

        Functions::FEFieldFunction<dim, DoFHandler<dim>, BlockVector<double>> fe_interface_data (dof1, in_vec);
        std::map<types::global_dof_index,double> boundary_values_velocity;

        typename FunctionMap<dim>::type boundary_functions_velocity;

        constraints.clear ();

        for (unsigned int side=0; side<GeometryInfo<dim>::faces_per_cell; ++side)
            if (neighbors[side] >= 0)
                boundary_functions_velocity[side+1] = &fe_interface_data;

        proj.project_boundary_values (dof2,
                                      boundary_functions_velocity,
                                      quad,
                                      constraints);

        constraints.close ();
        constraints.distribute (out_vec);
    }
}

#endif //ELASTICITY_MFEDD_UTILITIES_H
