/* ---------------------------------------------------------------------
 * Reimplementation of the deal.II project_boundary_values ()
 * Here I store the factorized boundary mass matrix as it is used
 * multiple times on every iteration of the interface CG.
 *
 * To hide the details it is wrapped into Projector class,
 * that calls the internal routines and stores the factorization of
 * the matrix.
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2017
 */

#ifndef ELASTICITY_MFEDD_PROJECTOR_H
#define ELASTICITY_MFEDD_PROJECTOR_H

// Internals
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_direct.h>


// Threads
#include <deal.II/base/work_stream.h>

namespace Projector
{
    using namespace dealii;

    struct Scratch {
        Scratch() = default;
    };

    template<typename DoFHandlerType, typename number>
    struct CopyData {
        CopyData();

        CopyData(CopyData const &data);

        unsigned int dofs_per_cell;
        std::vector <types::global_dof_index> dofs;
        std::vector <std::vector<bool>> dof_is_on_face;
        typename DoFHandlerType::active_cell_iterator cell;
        std::vector <FullMatrix<number>> cell_matrix;
        std::vector <Vector<number>> cell_vector;
    };


    template<typename DoFHandlerType, typename number>
    CopyData<DoFHandlerType, number>::CopyData()
            :
            dofs_per_cell(numbers::invalid_unsigned_int) {}


    template<typename DoFHandlerType, typename number>
    CopyData<DoFHandlerType, number>::CopyData(CopyData const &data)
            :
            dofs_per_cell(data.dofs_per_cell),
            dofs(data.dofs),
            dof_is_on_face(data.dof_is_on_face),
            cell(data.cell),
            cell_matrix(data.cell_matrix),
            cell_vector(data.cell_vector) {}


    template<int dim, int spacedim, typename number>
    void
    static inline
    create_boundary_mass_matrix_1(typename DoFHandler<dim, spacedim>::active_cell_iterator const &cell,
                                  Scratch const &,
                                  CopyData<DoFHandler < dim,
                                          spacedim>, number

                                  > &copy_data,
                                  Mapping <dim, spacedim> const &mapping,
                                  FiniteElement<dim, spacedim> const
                                  &fe,
                                  Quadrature<dim - 1> const &q,
                                  std::map<types::boundary_id, const Function <spacedim, number> *> const
                                  &boundary_functions,
                                  Function <spacedim, number> const *const coefficient,
                                  std::vector<unsigned int> const
                                  &component_mapping,
                                  const bool nomatrix
    )

    {
        // Most assertions for this function are in the calling function
        // before creating threads.
        const unsigned int n_components = fe.n_components();
        const unsigned int n_function_components = boundary_functions.begin()->second->n_components;
        const bool fe_is_system = (n_components != 1);
        const bool fe_is_primitive = fe.is_primitive();

        const unsigned int dofs_per_face = fe.dofs_per_face;

        copy_data.
                cell = cell;
        copy_data.
                dofs_per_cell = fe.dofs_per_cell;

        UpdateFlags update_flags = UpdateFlags(update_values |
                                               update_JxW_values |
                                               update_normal_vectors |
                                               update_quadrature_points);
        FEFaceValues <dim, spacedim> fe_values(mapping, fe, q, update_flags);

        // two variables for the coefficient, one for the two cases
        // indicated in the name
        std::vector <number> coefficient_values(fe_values.n_quadrature_points, 1.);
        std::vector <Vector<number>> coefficient_vector_values(fe_values.n_quadrature_points,
                                                               Vector<number>(n_components));
        const bool coefficient_is_vector = (coefficient != nullptr && coefficient->n_components != 1);

        std::vector <number> rhs_values_scalar(fe_values.n_quadrature_points);
        std::vector <Vector<number>> rhs_values_system(fe_values.n_quadrature_points,
                                                       Vector<number>(n_function_components));

        copy_data.dofs.
                resize(copy_data
                               .dofs_per_cell);
        cell->
                get_dof_indices (copy_data
                                         .dofs);

        std::vector <types::global_dof_index> dofs_on_face_vector(dofs_per_face);

        // Because CopyData objects are reused and emplace_back is
        // used, dof_is_on_face, cell_matrix, and cell_vector must be
        // cleared before they are reused
        copy_data.dof_is_on_face.

                clear();

        copy_data.cell_matrix.

                clear();

        copy_data.cell_vector.

                clear();

        for (
                unsigned int face = 0;
                face<GeometryInfo<dim>::faces_per_cell;
                ++face)
            // check if this face is on that part of the boundary we are
            // interested in
            if (boundary_functions.
                    find(cell
                                 ->
                                         face(face)
                                 ->

                                         boundary_id()

            ) !=
                boundary_functions.

                        end()

                    )
            {
                copy_data.cell_matrix.
                        emplace_back(copy_data
                                             .dofs_per_cell,
                                     copy_data.dofs_per_cell);
                copy_data.cell_vector.
                        emplace_back(copy_data
                                             .dofs_per_cell);
                fe_values.
                        reinit (cell, face
                );

                if (fe_is_system)
                    // FE has several components
                {
                    boundary_functions.
                                    find(cell
                                                 ->
                                                         face(face)
                                                 ->

                                                         boundary_id()

                            )
                            ->second->
                                    vector_value_list (fe_values
                                                               .

                                                                       get_quadrature_points(),
                                                       rhs_values_system

                            );

                    if (coefficient_is_vector)
                        // If coefficient is vector valued, fill all
                        // components
                        coefficient->
                                vector_value_list (fe_values
                                                           .

                                                                   get_quadrature_points(),
                                                   coefficient_vector_values

                        );
                    else
                    {
                        // If a scalar function is given, update the
                        // values, if not, use the default one set in the
                        // constructor above
                        if (coefficient != nullptr)
                            coefficient->
                                    value_list (fe_values
                                                        .

                                                                get_quadrature_points(),
                                                coefficient_values

                            );
                        // Copy scalar values into vector
                        for (
                                unsigned int point = 0;
                                point<fe_values.
                                        n_quadrature_points;
                                ++point)
                            coefficient_vector_values[point] = coefficient_values[point];
                    }

                    // Special treatment for Hdiv elements,
                    // where only normal components should be projected.
                    // For Hdiv we need to compute (u dot n, v dot n) which
                    // can be done as sum over dim components of
                    // u[c] * n[c] * v[c] * n[c] = u[c] * v[c] * normal_adjustment[c]
                    // Same approach does not work for Hcurl, so we throw an exception.
                    // Default value 1.0 allows for use with non Hdiv elements
                    std::vector <std::vector<double>> normal_adjustment(fe_values.n_quadrature_points,
                                                                        std::vector<double>(n_components, 1.));

                    for (
                            unsigned int comp = 0;
                            comp<n_components;
                            ++comp)
                    {
                        const FiniteElement <dim, spacedim> &base = fe.base_element(fe.component_to_base_index(comp).first);
                        const unsigned int bcomp = fe.component_to_base_index(comp).second;

                        if (!base.
                                conforms(FiniteElementData<dim>::H1)
                            &&
                            base.
                                    conforms(FiniteElementData<dim>::Hdiv)
                            &&
                            fe_is_primitive)
                        Assert(false,

                               ExcNotImplemented()

                        );

                        if (!base.
                                conforms(FiniteElementData<dim>::H1)
                            &&
                            base.
                                    conforms(FiniteElementData<dim>::Hcurl)
                                )
                        Assert(false,

                               ExcNotImplemented()

                        );

                        if (!base.
                                conforms(FiniteElementData<dim>::H1)
                            &&
                            base.
                                    conforms(FiniteElementData<dim>::Hdiv)
                                )
                            for (
                                    unsigned int point = 0;
                                    point<fe_values.
                                            n_quadrature_points;
                                    ++point)
                                normal_adjustment[point][comp] = fe_values.
                                        normal_vector(point)[bcomp]
                                                                 * fe_values.
                                        normal_vector(point)[bcomp];
                    }

                    for (
                            unsigned int point = 0;
                            point<fe_values.
                                    n_quadrature_points;
                            ++point)
                    {
                        const double weight = fe_values.JxW(point);
                        for (
                                unsigned int i = 0;
                                i<fe_values.
                                        dofs_per_cell;
                                ++i)
                            if (fe_is_primitive)
                            {
                                for (
                                        unsigned int j = 0;
                                        j<fe_values.
                                                dofs_per_cell;
                                        ++j)
                                {
                                    if (fe.
                                                    system_to_component_index(j)
                                                .first
                                        == fe.
                                                    system_to_component_index(i)
                                                .first && !nomatrix)
                                    {
                                        copy_data.cell_matrix.

                                                back()(i, j)

                                                += coefficient_vector_values[point](fe.
                                                        system_to_component_index(i)
                                                                                            .first)
                                                   *
                                                   weight
                                                   *fe_values
                                                           .
                                                                   shape_value(j, point
                                                           )
                                                   * fe_values.
                                                shape_value(i, point
                                        );
                                    }
                                }
                                copy_data.cell_vector.

                                        back()(i)

                                        += rhs_values_system[point](component_mapping[fe.
                                                system_to_component_index(i)
                                        .first])
                                           * fe_values.
                                        shape_value(i, point
                                )
                                           *
                                           weight;
                            }
                            else
                            {
                                for (
                                        unsigned int comp = 0;
                                        comp<n_components;
                                        ++comp)
                                {
                                    if (!nomatrix)
                                        for (
                                                unsigned int j = 0;
                                                j<fe_values.
                                                        dofs_per_cell;
                                                ++j)
                                            copy_data.cell_matrix.

                                                    back()(i, j)

                                                    += coefficient_vector_values[point](comp)
                                                       * fe_values.
                                                    shape_value_component(j, point, comp
                                            )
                                                       * fe_values.
                                                    shape_value_component(i, point, comp
                                            )
                                                       * normal_adjustment[point][comp]
                                                       *
                                                       weight;
                                    copy_data.cell_vector.

                                            back()(i)

                                            += rhs_values_system[point](component_mapping[comp])
                                               * fe_values.
                                            shape_value_component(i, point, comp
                                    )
                                               * normal_adjustment[point][comp]
                                               *
                                               weight;
                                }
                            }
                    }
                }
                else
                    // FE is a scalar one
                {
                    boundary_functions.
                                    find(cell
                                                 ->
                                                         face(face)
                                                 ->

                                                         boundary_id()

                            )
                            ->second->
                                    value_list (fe_values
                                                        .

                                                                get_quadrature_points(), rhs_values_scalar

                            );

                    if (coefficient != nullptr)
                        coefficient->
                                value_list (fe_values
                                                    .

                                                            get_quadrature_points(),
                                            coefficient_values

                        );
                    for (
                            unsigned int point = 0;
                            point<fe_values.
                                    n_quadrature_points;
                            ++point)
                    {
                        const double weight = fe_values.JxW(point);
                        for (
                                unsigned int i = 0;
                                i<fe_values.
                                        dofs_per_cell;
                                ++i)
                        {
                            const double v = fe_values.shape_value(i, point);
                            for (
                                    unsigned int j = 0;
                                    j<fe_values.
                                            dofs_per_cell;
                                    ++j)
                            {
                                const double u = fe_values.shape_value(j, point);
                                copy_data.cell_matrix.

                                        back()(i, j)

                                        += (coefficient_values[point]*
                                            u *v
                                            *weight);
                            }
                            copy_data.cell_vector.

                                    back()(i)

                                    += rhs_values_scalar[point] *
                                       v *weight;
                        }
                    }
                }


                cell->
                                face(face)
                        ->
                                get_dof_indices (dofs_on_face_vector);
                // for each dof on the cell, have a flag whether it is on
                // the face
                copy_data.dof_is_on_face.
                        emplace_back(copy_data
                                             .dofs_per_cell);
                // check for each of the dofs on this cell whether it is
                // on the face
                for (
                        unsigned int i = 0;
                        i<copy_data.
                                dofs_per_cell;
                        ++i)
                    copy_data.dof_is_on_face.

                            back()[i] = (std::find(dofs_on_face_vector.begin(),
                                                   dofs_on_face_vector.end(),
                                                   copy_data.dofs[i])
                                         !=
                                         dofs_on_face_vector.end());

            }
    }


    template<int dim, int spacedim, typename number>
    void copy_boundary_mass_matrix_1(CopyData<DoFHandler < dim,
            spacedim>, number

    > const &copy_data,
                                     std::map<types::boundary_id, const Function <spacedim, number> *> const &boundary_functions,
                                     std::vector<types::global_dof_index> const
                                     &dof_to_boundary_mapping,
                                     SparseMatrix <number> &matrix,
                                     Vector<number>
                                     &rhs_vector)
    {
        // now transfer cell matrix and vector to the whole boundary matrix
        //
        // in the following: dof[i] holds the global index of the i-th degree of
        // freedom on the present cell. If it is also a dof on the boundary, it
        // must be a nonzero entry in the dof_to_boundary_mapping and then
        // the boundary index of this dof is dof_to_boundary_mapping[dof[i]].
        //
        // if dof[i] is not on the boundary, it should be zero on the boundary
        // therefore on all quadrature points and finally all of its
        // entries in the cell matrix and vector should be zero. If not, we
        // throw an error (note: because of the evaluation of the shape
        // functions only up to machine precision, the term "must be zero"
        // really should mean: "should be very small". since this is only an
        // assertion and not part of the code, we may choose "very small"
        // quite arbitrarily)
        //
        // the main problem here is that the matrix or vector entry should also
        // be zero if the degree of freedom dof[i] is on the boundary, but not
        // on the present face, i.e. on another face of the same cell also
        // on the boundary. We can therefore not rely on the
        // dof_to_boundary_mapping[dof[i]] being !=-1, we really have to
        // determine whether dof[i] is a dof on the present face. We do so
        // by getting the dofs on the face into @p{dofs_on_face_vector},
        // a vector as always. Usually, searching in a vector is
        // inefficient, so we copy the dofs into a set, which enables binary
        // searches.
        unsigned int pos(0);
        for (
                unsigned int face = 0;
                face<GeometryInfo<dim>::faces_per_cell;
                ++face)
        {
            // check if this face is on that part of
            // the boundary we are interested in
            if (boundary_functions.
                    find(copy_data
                                 .cell->
                            face(face)
                                 ->

                                         boundary_id()

            ) !=
                boundary_functions.

                        end()

                    )
            {
                for (
                        unsigned int i = 0;
                        i<copy_data.
                                dofs_per_cell;
                        ++i)
                {
                    if (copy_data.dof_is_on_face[pos][i] &&
                        dof_to_boundary_mapping[copy_data.dofs[i]] != numbers::invalid_dof_index)
                    {
                        for (
                                unsigned int j = 0;
                                j<copy_data.
                                        dofs_per_cell;
                                ++j)
                            if (copy_data.dof_is_on_face[pos][j] &&
                                dof_to_boundary_mapping[copy_data.dofs[j]] != numbers::invalid_dof_index)
                            {
                                AssertIsFinite(copy_data
                                                       .cell_matrix[pos](i,j));
                                matrix.
                                        add(dof_to_boundary_mapping[copy_data.dofs[i]],
                                            dof_to_boundary_mapping[copy_data.dofs[j]],
                                            copy_data
                                                    .cell_matrix[pos](i,j));
                            }
                        AssertIsFinite(copy_data
                                               .cell_vector[pos](i));
                        rhs_vector(dof_to_boundary_mapping[copy_data.dofs[i]])
                                += copy_data.cell_vector[pos](i);
                    }
                }
                ++
                        pos;
            }
        }
    }

    template<int dim, int spacedim, typename number>
    void
    create_boundary_mass_matrix(const Mapping <dim, spacedim> &mapping,
                                const DoFHandler <dim, spacedim> &dof,
                                const Quadrature<dim - 1> &q,
                                SparseMatrix <number> &matrix,
                                const std::map<types::boundary_id, const Function <spacedim, number> *> &boundary_functions,
                                Vector <number> &rhs_vector,
                                std::vector <types::global_dof_index> &dof_to_boundary_mapping,
                                const Function <spacedim, number> *const coefficient,
                                std::vector<unsigned int> component_mapping,
                                const bool nomatrix) {
        // what would that be in 1d? the
        // identity matrix on the boundary
        // dofs?
        if (dim == 1) {
            Assert(false, ExcNotImplemented());
            return;
        }

        const FiniteElement<dim, spacedim> &fe = dof.get_fe();
        const unsigned int n_components = fe.n_components();

        Assert(matrix.n() == dof.n_boundary_dofs(boundary_functions),
               ExcInternalError());
        Assert(matrix.n() == matrix.m(), ExcInternalError());
        Assert(matrix.n() == rhs_vector.size(), ExcInternalError());
        Assert(boundary_functions.size() != 0, ExcInternalError());
        Assert(dof_to_boundary_mapping.size() == dof.n_dofs(),
               ExcInternalError());
        Assert(coefficient == nullptr ||
               coefficient->n_components == 1 ||
               coefficient->n_components == n_components, ExcNotImplemented());

        if (component_mapping.size() == 0) {
            AssertDimension(n_components, boundary_functions.begin()->second->n_components);
            for (unsigned int i = 0; i < n_components; ++i)
                component_mapping.push_back(i);
        } else
        AssertDimension(n_components, component_mapping.size());

        Scratch scratch;
        CopyData<DoFHandler < dim, spacedim>, number > copy_data;

        WorkStream::run(dof.begin_active(), dof.end(),
                        static_cast<std::function<void(typename DoFHandler<dim, spacedim>::active_cell_iterator
                                                       const &, Scratch const &,
                                                       CopyData<DoFHandler < dim, spacedim>, number> &) > >
                        (std::bind(&create_boundary_mass_matrix_1<dim, spacedim, number>, std::placeholders::_1,
                                   std::placeholders::_2,
                                   std::placeholders::_3,
                                   std::cref(mapping), std::cref(fe), std::cref(q),
                                   std::cref(boundary_functions), coefficient,
                                   std::cref(component_mapping),
                                   std::cref(nomatrix))),
                        static_cast<std::function<void(CopyData<DoFHandler < dim, spacedim>, number> const &) > >
                        (std::bind(
                                &copy_boundary_mass_matrix_1<dim, spacedim, number>,
                                std::placeholders::_1,
                                std::cref(boundary_functions),
                                std::cref(dof_to_boundary_mapping),
                                std::ref(matrix),
                                std::ref(rhs_vector))),
                        scratch,
                        copy_data);
    }

    template <int dim>
    class Projector
    {
    public:
        Projector(unsigned int s) : state(s) {}

        template <int spacedim, typename number>
        void
        project_boundary_values
                (const DoFHandler<dim,spacedim>                                       &dof,
                 const std::map<types::boundary_id, const Function<spacedim,number>*> &boundary_functions,
                 const Quadrature<dim-1>                                              &q,
                 ConstraintMatrix                                                     &constraints,
                 std::vector<unsigned int>                                             component_mapping = std::vector<unsigned int>());

        void reset()
        {
            state = false;
            mass_matrix.clear();
            selected_boundary_components.clear();
            dof_to_boundary_mapping.clear();
        }

        //friend class MixedBiotProblemDD<dim>;

    private:
        unsigned int state;
        SparseDirectUMFPACK factorization;
        std::vector<types::global_dof_index> dof_to_boundary_mapping;
        std::set<types::boundary_id> selected_boundary_components;
        SparsityPattern sparsity;
        SparseMatrix<double> mass_matrix;

        template <typename number>
        void invert_mass_matrix(const SparseMatrix<number> &mass_matrix,
                                const Vector<number>       &rhs,
                                Vector<number>             &solution);

        template <int spacedim, typename number>
        void
        do_project_boundary_values
                (const DoFHandler<dim,spacedim>                                   &dof,
                 const std::map<types::boundary_id, const Function<spacedim,number>*> &boundary_functions,
                 const Quadrature<dim-1>                                                 &q,
                 std::map<types::global_dof_index,number>                             &boundary_values,
                 std::vector<unsigned int>                                             component_mapping);


    };


    template <int dim>
    template <typename number>
    void Projector<dim>::invert_mass_matrix(const SparseMatrix<number> &mass_matrix,
                                            const Vector<number>       &rhs,
                                            Vector<number>             &solution)
    {
//      ReductionControl      control(5*rhs.size(), 0., 1e-12, false, false);
//      GrowingVectorMemory<Vector<number> > memory;
//      SolverCG<Vector<number> >            cg(control,memory);
//
//      PreconditionSSOR<SparseMatrix<number> > prec;
//      prec.initialize(mass_matrix, 1.2);
//
//      cg.solve (mass_matrix, solution, rhs, prec);

        if (state == true)
            factorization.vmult(solution, rhs);
        else
        {
            factorization.initialize(mass_matrix);
            factorization.vmult(solution, rhs);
        }
    }

    template <int dim>
    template <int spacedim, typename number>
    void
    Projector<dim>::do_project_boundary_values
            (const DoFHandler<dim,spacedim>                                   &dof,
             const std::map<types::boundary_id, const Function<spacedim,number>*> &boundary_functions,
             const Quadrature<dim-1>                                                 &q,
             std::map<types::global_dof_index,number>                             &boundary_values,
             std::vector<unsigned int>                                             component_mapping)
    {
        Assert(dim >= 2, ExcNotImplemented());

        MappingQGeneric<dim> mapping(StaticMappingQ1<dim>::mapping);

        if (component_mapping.size() == 0)
        {
            //AssertDimension (dof.get_fe(0).n_components(), boundary_functions.begin()->second->n_components);
            // I still do not see why i
            // should create another copy
            // here
            component_mapping.resize(dof.get_fe(0).n_components());
            for (unsigned int i = 0; i < component_mapping.size(); ++i)
                component_mapping[i] = i;
        } else
        Assert (dof.get_fe(0).n_components() == component_mapping.size(), ExcInternalError());

        if (state == false)
        {
            for (typename std::map<types::boundary_id, const Function<dim, double> *>::const_iterator
                         i = boundary_functions.begin();
                 i != boundary_functions.end(); ++i)
                selected_boundary_components.insert(i->first);

            DoFTools::map_dof_to_boundary_indices(dof, selected_boundary_components,
                                                  dof_to_boundary_mapping);
        }

        // Done if no degrees of freedom on the boundary
        if (dof.n_boundary_dofs(boundary_functions) == 0)
            return;

        // set up sparsity structure
        if (state == false)
        {
            DynamicSparsityPattern dsp(dof.n_boundary_dofs(boundary_functions),
                                       dof.n_boundary_dofs(boundary_functions));
            DoFTools::make_boundary_sparsity_pattern(dof,
                                                     boundary_functions,
                                                     dof_to_boundary_mapping,
                                                     dsp);

            sparsity.copy_from(dsp);
            sparsity.compress();

            mass_matrix.reinit(sparsity);
        }

        Vector<double>       rhs(sparsity.n_rows());

        create_boundary_mass_matrix (mapping, dof, q,
                                     mass_matrix, boundary_functions,
                                     rhs, dof_to_boundary_mapping,
                                     (const Function<dim,double> *) nullptr,
                                     component_mapping,
                                     state);

        Vector<double> boundary_projection (rhs.size());

        if (rhs.norm_sqr() < 1e28 * std::numeric_limits<double>::min())
            boundary_projection = 0;
        else
        {
            invert_mass_matrix(mass_matrix,
                               rhs,
                               boundary_projection);
        }
        // fill in boundary values
        for (unsigned int i=0; i<dof_to_boundary_mapping.size(); ++i)
            if (dof_to_boundary_mapping[i] != numbers::invalid_dof_index)
            {
                AssertIsFinite(boundary_projection(dof_to_boundary_mapping[i]));

                boundary_values[i] = boundary_projection(dof_to_boundary_mapping[i]);
            }
    }


    template <int dim>
    template <int spacedim, typename number>
    void
    Projector<dim>::project_boundary_values
            (const DoFHandler<dim,spacedim>                                       &dof,
             const std::map<types::boundary_id, const Function<spacedim,number>*> &boundary_functions,
             const Quadrature<dim-1>                                              &q,
             ConstraintMatrix                                                     &constraints,
             std::vector<unsigned int>                                             component_mapping)
    {
        std::map<types::global_dof_index,number> boundary_values;


        do_project_boundary_values(dof, boundary_functions, q,
                                   boundary_values, component_mapping);

        typename std::map<types::global_dof_index,number>::const_iterator boundary_value =
                boundary_values.begin();
        for ( ; boundary_value !=boundary_values.end(); ++boundary_value)
        {
            if (!constraints.is_constrained(boundary_value->first))
            {
                constraints.add_line (boundary_value->first);
                constraints.set_inhomogeneity (boundary_value->first,
                                               boundary_value->second);
            }
        }
    }
}

#endif //ELASTICITY_MFEDD_PROJECTOR_H
