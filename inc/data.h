/* ---------------------------------------------------------------------
 * Functions representing RHS, physical parameters, boundary conditions and
 * the true solution.
 * ---------------------------------------------------------------------
 *
 * Author: Eldar Khattatov, University of Pittsburgh, 2018
 */

#ifndef ELASTICITY_MFEDD_DATA_H
#define ELASTICITY_MFEDD_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

namespace dd_biot
{
    using namespace dealii;

    // Lame parameters (lambda and mu)
    template <int dim>
    class LameParameters : public Function<dim>
    {
    public:
        LameParameters ()  : Function<dim>() {}
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const;

        virtual void vector_value_list (const std::vector<Point<dim> > &points,
        std::vector<Vector<double> > &value_list) const;
    };

    template <int dim>
    void LameParameters<dim>::vector_value (const Point<dim> &p,
                                            Vector<double>   &values) const
    {
        Assert(values.size() == 2,
               ExcDimensionMismatch(values.size(),2));
        Assert(dim != 1, ExcNotImplemented());

        double x,y,z;
        x = p[0];
        y = p[1];

        if (dim == 3)
            z = p[2];

        switch (dim)
        {
            case 2:
                values(0) = 100.0;
                values(1) = 100.0;
                break;
            case 3:
                values(0) = 100.0;
                values(1) = 100.0;
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    template <int dim>
    void LameParameters<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                                std::vector<Vector<double> > &value_list) const
    {
        Assert(value_list.size() == points.size(),
               ExcDimensionMismatch(value_list.size(), points.size()));

        const unsigned int n_points = points.size();

        for (unsigned int p=0; p<n_points; ++p)
            LameParameters<dim>::vector_value(points[p], value_list[p]);
    }

    // Inverse of permeability tensor
    template <int dim>
    class KInverse : public TensorFunction<2,dim>
    {
    public:
      KInverse () : TensorFunction<2,dim>() {}

      virtual void value_list (const std::vector<Point<dim> > &points,
                               std::vector<Tensor<2,dim> >    &values) const;
    };

    template <int dim>
    void
    KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                               std::vector<Tensor<2,dim> >    &values) const
    {
      Assert (points.size() == values.size(),
              ExcDimensionMismatch (points.size(), values.size()));

      for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();

        const double x = points[p][0];
        const double y = points[p][1];

        switch (dim)
        {
          case 2:
            values[p][0][0] = 1.0;
            values[p][0][1] = 0.0;
            values[p][1][0] = 0.0;
            values[p][1][1] = 1.0;
            break;
          default:
          Assert(false, ExcMessage("The inverse of permeability tensor for dim != 2 is not provided"));
        }
      }
    }


    // Right hand side values, boundary conditions and exact solution
    template <int dim>
    class RightHandSideElasticity : public Function<dim>
    {
        public:
            RightHandSideElasticity () : Function<dim>(dim) {}

            virtual void vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const;

            virtual void vector_value_list (const std::vector<Point<dim> >   &points,
            std::vector<Vector<double> > &value_list) const;
    };

    template <int dim>
    inline
    void RightHandSideElasticity<dim>::vector_value(const Point<dim> &p,
                                          Vector<double>   &values) const
    {
        Assert(values.size() == dim,
               ExcDimensionMismatch(values.size(),dim));
        Assert(dim != 1, ExcNotImplemented());

        double x = p[0];
        double y = p[1];
        double z;

        if (dim == 3)
            z = p[2];

        const LameParameters<dim> lame_function;
        Vector<double> vec(2);
        lame_function.vector_value(p,vec);

        const double lmbda = vec[0];
        const double mu = vec[1];
        double t = FunctionTime<double>::get_time();

        switch (dim)
        {
            case 2:
                values(0) = mu*(exp(t)*((x*x*x)*(y*y)*-1.2E1+cos(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(x-1.0,2.0)+sin(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0)*2.0)*(1.0/2.0)+exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x))*(1.0/2.0))*2.0-lmbda*(exp(t)*(x*(y*y*y*y)*6.0-cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(y-1.0,2.0)+2.0)-exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x)))-mu*exp(t)*(x*(y*y*y*y)*6.0-cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(y-1.0,2.0)+2.0)*2.0+3.141592653589793*exp(t)*cos(x*3.141592653589793)*cos(y*3.141592653589793);
                values(1) = mu*(exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,2.0)*pow(y-1.0,3.0)*1.2E1+cos(x*y)*sin(x)+y*sin(x*y)*cos(x)*2.0+(y*y)*cos(x*y)*sin(x))*(1.0/2.0))*-2.0+lmbda*(exp(t)*((y*2.0-2.0)*pow(x-1.0,4.0)*3.0+(x*x)*cos(x*y)*sin(x)-2.0)-exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0)))+mu*exp(t)*((y*2.0-2.0)*pow(x-1.0,4.0)*3.0+(x*x)*cos(x*y)*sin(x)-2.0)*2.0-3.141592653589793*exp(t)*sin(x*3.141592653589793)*sin(y*3.141592653589793);
                break;
            case 3:
                values(0) = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-4.0+lmbda*exp(x)*(cos(M_PI/12.0)*-2.0E1+sin(M_PI*y)*sin(M_PI*z)+2.0E1)*(1.0/1.0E1)+mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
                values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*2.0+M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1);
                values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*-2.0+M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/1.0E1);
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    template <int dim>
    void RightHandSideElasticity<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                               std::vector<Vector<double> >   &value_list) const
    {
        Assert(value_list.size() == points.size(),
               ExcDimensionMismatch(value_list.size(), points.size()));

        const unsigned int n_points = points.size();

        for (unsigned int p=0; p<n_points; ++p)
            RightHandSideElasticity<dim>::vector_value(points[p], value_list[p]);
    }

    // Right hand side for momentum equation
    template <int dim>
    class RightHandSidePressure : public Function<dim>
    {
    public:
      RightHandSidePressure (const double c0=1.0, const double alpha=1.0); //{}

      virtual double value (const Point<dim>   &p,
                            const unsigned int component = 0 ) const;

    private:
      double c0; //=1.0;
      double alpha ;//=1.0;
    };

    template <int dim>
      RightHandSidePressure<dim>::RightHandSidePressure (const double c0, const double alpha) :
	  Function<dim>(1),
	  c0(c0),
    alpha(alpha)
	  {}

    template <int dim>
    double RightHandSidePressure<dim>::value (const Point<dim>  &p,
                                      const unsigned int /*component*/) const
//									  ,const double c0,
//									  const double alpha) const
    {
      const double x = p[0];
      const double y = p[1];
      double t = FunctionTime<double>::get_time();
//      double c0=0.0001;
//      double alpha=1.0;

      switch (dim)
      {
        case 2:
          return c0*exp(t)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1)+alpha*exp(t)*(x*2.0+(x*x)*(y*y*y*y)*3.0+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0))-exp(t)*(y*-2.0+pow(x-1.0,4.0)*pow(y-1.0,2.0)*3.0+x*sin(x*y)*sin(x)+2.0)+(3.141592653589793*3.141592653589793)*exp(t)*cos(y*3.141592653589793)*sin(x*3.141592653589793)*2.0;

        default:
        Assert(false, ExcMessage("The RHS data for dim != 2 is not provided"));
      }
    }


  // Boundary conditions for displacement (natural)
    template <int dim>
    class DisplacementBoundaryValues : public Function<dim>
    {
    public:
        DisplacementBoundaryValues() : Function<dim>(dim) {}

        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const;
        virtual void vector_value_list (const std::vector<Point<dim> >   &points,
        std::vector<Vector<double> > &value_list) const;
    };

    template <int dim>
    void DisplacementBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                                        Vector<double>   &values) const
    {
        double x = p[0];
        double y = p[1];
        double z;

        if (dim == 3)
            z = p[2];

        const LameParameters<dim> lame_function;
        Vector<double> vec(2);
        lame_function.vector_value(p,vec);

        const double lmbda = vec[0];
        const double mu = vec[1];
        double t = FunctionTime<double>::get_time();

        switch (dim)
        {
            case 2:
                values(0) = exp(t)*((x*x*x)*(y*y*y*y)+cos(y-1.0)*sin((x-1.0)*(y-1.0))+x*x);
                values(1) = exp(t)*(pow(y-1.0,2.0)-pow(x-1.0,4.0)*pow(y-1.0,3.0)+cos(x*y)*sin(x));
                break;
            case 3:
                values(0) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
                values(1) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
                values(2) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    template <int dim>
    void DisplacementBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim> > &points,
    std::vector<Vector<double> >   &value_list) const
    {
        Assert(value_list.size() == points.size(),
               ExcDimensionMismatch(value_list.size(), points.size()));

        const unsigned int n_points = points.size();

        for (unsigned int p=0; p<n_points; ++p)
            DisplacementBoundaryValues<dim>::vector_value(points[p], value_list[p]);
    }

    template <int dim>
    class PressureBoundaryValues : public Function<dim>
    {
    public:
      PressureBoundaryValues () : Function<dim>(1) {}

      virtual double value (const Point<dim>   &p,
                            const unsigned int component = 0) const;
    };

    template <int dim>
    double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                               const unsigned int /*component*/) const
    {
      const double x = p[0];
      const double y = p[1];
      double t = FunctionTime<double>::get_time();

      switch (dim)
      {
        case 2:
          return exp(t)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1);
        default:
        Assert(false, ExcMessage("The BC data for dim != 2 is not provided"));
      }
    }

    // Exact solution
    template <int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution() : Function<dim>(static_cast<unsigned int>(dim*dim + dim + 0.5*dim*(dim-1) + dim + 1)) {}

        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const;
        virtual void vector_gradient (const Point<dim> &p,
                                      std::vector<Tensor<1,dim,double> > &grads) const;
    };

    template <int dim>
    void
    ExactSolution<dim>::vector_value (const Point<dim> &p,
                                      Vector<double>   &values) const
    {
        double x = p[0];
        double y = p[1];
        double z;

        if (dim == 3)
            z = p[2];

        const LameParameters<dim> lame_function;
        Vector<double> vec(2);
        lame_function.vector_value(p,vec);

        const double lmbda = vec[0];
        const double mu = vec[1];
        double t = FunctionTime<double>::get_time();

        switch (dim)
        {
            case 2:
                values(0) = -exp(t)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1)+lmbda*(exp(t)*(x*2.0+(x*x)*(y*y*y*y)*3.0+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0))-exp(t)*(y*-2.0+pow(x-1.0,4.0)*pow(y-1.0,2.0)*3.0+x*sin(x*y)*sin(x)+2.0))+mu*exp(t)*(x*2.0+(x*x)*(y*y*y*y)*3.0+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0))*2.0;
                values(1) = mu*(exp(t)*((x*x*x)*(y*y*y)*4.0-sin(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,3.0)*pow(y-1.0,3.0)*4.0-cos(x*y)*cos(x)+y*sin(x*y)*sin(x))*(1.0/2.0))*2.0;
                values(2) = mu*(exp(t)*((x*x*x)*(y*y*y)*4.0-sin(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,3.0)*pow(y-1.0,3.0)*4.0-cos(x*y)*cos(x)+y*sin(x*y)*sin(x))*(1.0/2.0))*2.0;
                values(3) = -exp(t)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1)+lmbda*(exp(t)*(x*2.0+(x*x)*(y*y*y*y)*3.0+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0))-exp(t)*(y*-2.0+pow(x-1.0,4.0)*pow(y-1.0,2.0)*3.0+x*sin(x*y)*sin(x)+2.0))-mu*exp(t)*(y*-2.0+pow(x-1.0,4.0)*pow(y-1.0,2.0)*3.0+x*sin(x*y)*sin(x)+2.0)*2.0;
                values(4) = exp(t)*((x*x*x)*(y*y*y*y)+cos(y-1.0)*sin((x-1.0)*(y-1.0))+x*x);
                values(5) = exp(t)*(pow(y-1.0,2.0)-pow(x-1.0,4.0)*pow(y-1.0,3.0)+cos(x*y)*sin(x));
                values(6) = exp(t)*((x*x*x)*(y*y*y)*4.0-sin(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0))*(1.0/2.0)+exp(t)*(pow(x-1.0,3.0)*pow(y-1.0,3.0)*4.0-cos(x*y)*cos(x)+y*sin(x*y)*sin(x))*(1.0/2.0);
                values(7) = -3.141592653589793*exp(t)*cos(x*3.141592653589793)*cos(y*3.141592653589793);
                values(8) = 3.141592653589793*exp(t)*sin(x*3.141592653589793)*sin(y*3.141592653589793);
                values(9) = exp(t)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1);
                break;
            case 3:
                values(0) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
                values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
                values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
                values(3) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
                values(4) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;
                values(5) = 0;
                values(6) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
                values(7) = 0;
                values(8) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;

                values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
                values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
                values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);

                values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
                values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
                values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
                break;
            default:
                Assert(false, ExcNotImplemented());
        }
    }

    template <int dim>
    void
    ExactSolution<dim>::vector_gradient (const Point<dim> &p,
                                         std::vector<Tensor<1,dim,double> > &grads) const
    {
        double x = p[0];
        double y = p[1];
        double z;

        if (dim == 3)
        z = p[2];

        const LameParameters<dim> lame_function;
        Vector<double> vec(2);
        lame_function.vector_value(p,vec);

        const double lambda = vec[0];
        const double mu = vec[1];
        double t = FunctionTime<double>::get_time();

        int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));
        Tensor<1,dim> tmp;
        switch (dim)
        {
        case 2:
            grads[0][0] = lambda*(exp(t)*(x*(y*y*y*y)*6.0-cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(y-1.0,2.0)+2.0)-exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x)))+mu*exp(t)*(x*(y*y*y*y)*6.0-cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(y-1.0,2.0)+2.0)*2.0-3.141592653589793*exp(t)*cos(x*3.141592653589793)*cos(y*3.141592653589793);
            grads[0][1] = -lambda*(exp(t)*((y*2.0-2.0)*pow(x-1.0,4.0)*3.0+(x*x)*cos(x*y)*sin(x)-2.0)-exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0)))+mu*exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0))*2.0+3.141592653589793*exp(t)*sin(x*3.141592653589793)*sin(y*3.141592653589793);

            grads[1][0] = mu*(exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,2.0)*pow(y-1.0,3.0)*1.2E1+cos(x*y)*sin(x)+y*sin(x*y)*cos(x)*2.0+(y*y)*cos(x*y)*sin(x))*(1.0/2.0))*2.0;
            grads[1][1] = mu*(exp(t)*((x*x*x)*(y*y)*-1.2E1+cos(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(x-1.0,2.0)+sin(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0)*2.0)*(1.0/2.0)+exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x))*(1.0/2.0))*-2.0;

            grads[2][0] = mu*(exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,2.0)*pow(y-1.0,3.0)*1.2E1+cos(x*y)*sin(x)+y*sin(x*y)*cos(x)*2.0+(y*y)*cos(x*y)*sin(x))*(1.0/2.0))*2.0;
            grads[2][1] = mu*(exp(t)*((x*x*x)*(y*y)*-1.2E1+cos(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(x-1.0,2.0)+sin(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0)*2.0)*(1.0/2.0)+exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x))*(1.0/2.0))*-2.0;

            grads[3][0] = lambda*(exp(t)*(x*(y*y*y*y)*6.0-cos(y-1.0)*sin((x-1.0)*(y-1.0))*pow(y-1.0,2.0)+2.0)-exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x)))-mu*exp(t)*(sin(x*y)*sin(x)+pow(x-1.0,3.0)*pow(y-1.0,2.0)*1.2E1+x*sin(x*y)*cos(x)+x*y*cos(x*y)*sin(x))*2.0-3.141592653589793*exp(t)*cos(x*3.141592653589793)*cos(y*3.141592653589793);
            grads[3][1] = -lambda*(exp(t)*((y*2.0-2.0)*pow(x-1.0,4.0)*3.0+(x*x)*cos(x*y)*sin(x)-2.0)-exp(t)*(cos(y-1.0)*cos((x-1.0)*(y-1.0))+(x*x)*(y*y*y)*1.2E1-sin(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0)-cos(y-1.0)*sin((x-1.0)*(y-1.0))*(x-1.0)*(y-1.0)))-mu*exp(t)*((y*2.0-2.0)*pow(x-1.0,4.0)*3.0+(x*x)*cos(x*y)*sin(x)-2.0)*2.0+3.141592653589793*exp(t)*sin(x*3.141592653589793)*sin(y*3.141592653589793);

            grads[7][0] = (3.141592653589793*3.141592653589793)*exp(t)*cos(y*3.141592653589793)*sin(x*3.141592653589793);
            grads[7][1] = (3.141592653589793*3.141592653589793)*exp(t)*cos(x*3.141592653589793)*sin(y*3.141592653589793);

            grads[8][0] = (3.141592653589793*3.141592653589793)*exp(t)*cos(x*3.141592653589793)*sin(y*3.141592653589793);
            grads[8][1] = (3.141592653589793*3.141592653589793)*exp(t)*cos(y*3.141592653589793)*sin(x*3.141592653589793);
            break;
        case 3:
            for (int k=0;k<total_dim;++k)
                grads[k] = tmp;

            break;
        default:
            Assert(false, ExcNotImplemented());
        }
    }

  // Exact solution
  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    InitialCondition() : Function<dim>(static_cast<unsigned int>(dim*dim + dim + 0.5*dim*(dim-1) + dim + 1)) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
  };

  template <int dim>
  void
  InitialCondition<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
      double x = p[0];
      double y = p[1];
      double z;

      if (dim == 3)
          z = p[2];

      const LameParameters<dim> lame_function;
      Vector<double> vec(2);
      lame_function.vector_value(p,vec);

      const double lmbda = vec[0];
      const double mu = vec[1];
      double t = 0;

      switch (dim)
      {
          case 2:
              values(0) = -exp(t)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1)+lmbda*(exp(t)*(x*2.0+(x*x)*(y*y*y*y)*3.0+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0))-exp(t)*(y*-2.0+pow(x-1.0,4.0)*pow(y-1.0,2.0)*3.0+x*sin(x*y)*sin(x)+2.0))+mu*exp(t)*(x*2.0+(x*x)*(y*y*y*y)*3.0+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0))*2.0;
              values(1) = mu*(exp(t)*((x*x*x)*(y*y*y)*4.0-sin(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,3.0)*pow(y-1.0,3.0)*4.0-cos(x*y)*cos(x)+y*sin(x*y)*sin(x))*(1.0/2.0))*2.0;
              values(2) = mu*(exp(t)*((x*x*x)*(y*y*y)*4.0-sin(y-1.0)*sin((x-1.0)*(y-1.0))+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(x-1.0))*(1.0/2.0)-exp(t)*(pow(x-1.0,3.0)*pow(y-1.0,3.0)*4.0-cos(x*y)*cos(x)+y*sin(x*y)*sin(x))*(1.0/2.0))*2.0;
              values(3) = -exp(t)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1)+lmbda*(exp(t)*(x*2.0+(x*x)*(y*y*y*y)*3.0+cos(y-1.0)*cos((x-1.0)*(y-1.0))*(y-1.0))-exp(t)*(y*-2.0+pow(x-1.0,4.0)*pow(y-1.0,2.0)*3.0+x*sin(x*y)*sin(x)+2.0))-mu*exp(t)*(y*-2.0+pow(x-1.0,4.0)*pow(y-1.0,2.0)*3.0+x*sin(x*y)*sin(x)+2.0)*2.0;
              values(4) = 0;
              values(5) = 0;
              values(6) = 0;
              values(7) = 0;
              values(8) = 0;
              values(9) = (cos(y*M_PI)*sin(x*M_PI)+1.0E1);
            //values = 0;
          break;
          case 3:
              values(0) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
              values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
              values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
              values(3) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
              values(4) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;
              values(5) = 0;
              values(6) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
              values(7) = 0;
              values(8) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;

              values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
              values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
              values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);

              values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
              values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
              values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
              break;
          default:
            Assert(false, ExcNotImplemented());
      }
  }

//     // First and second Lame parameters
//     template <int dim>
//     class LameFirstParameter : public Function<dim>
//     {
//     public:
//         LameFirstParameter ()  : Function<dim>() {}
//         virtual double value (const Point<dim>   &p,
//                               const unsigned int  component = 0) const;
//         virtual void value_list (const std::vector<Point<dim> > &points,
//                                  std::vector<double>            &values,
//                                  const unsigned int              component = 0) const;
//     };

//     template <int dim>
//     double LameFirstParameter<dim>::value (const Point<dim> &p,
//                                            const unsigned int /* component */) const
//     {
//         const double nu = 0.2;
//         double x,y,z;
//         x = p[0];
//         y = p[1];

//         if (dim == 3)
//             z = p[2];

//         switch (dim)
//         {
//             case 2:
//                 return (sin(3.0*M_PI*x)*sin(3.0*M_PI*y)+5.0)*nu/((1.0-nu)*(1.0-2.0*nu));
//                 break;
//             case 3:
//                 //return exp(4*x)*nu/((1.0-nu)*(1.0-2.0*nu));
//                 return 100.0;
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }

//     }

//     template <int dim>
//     void LameFirstParameter<dim>::value_list(const std::vector<Point<dim> > &points,
//                                              std::vector<double> &values,
//                                              const unsigned int component) const
//     {
//         Assert(values.size() == points.size(),
//                ExcDimensionMismatch(values.size(), points.size()));

//         const unsigned int n_points = points.size();

//         for (unsigned int p=0; p<n_points; ++p)
//             values[p] = LameFirstParameter<dim>::value(points[p]);
//     }

//     template <int dim>
//     class LameSecondParameter : public Function<dim>
//     {
//     public:
//         LameSecondParameter ()  : Function<dim>() {}
//         virtual double value (const Point<dim>   &p,
//                               const unsigned int  component = 0) const;
//         virtual void value_list (const std::vector<Point<dim> > &points,
//                                  std::vector<double>            &values,
//                                  const unsigned int              component = 0) const;
//     };

//     template <int dim>
//     double LameSecondParameter<dim>::value (const Point<dim> &p,
//                                             const unsigned int /* component */) const
//     {
//         const double nu = 0.2;
//         double x,y,z;
//         x = p[0];
//         y = p[1];

//         if (dim == 3)
//             z = p[2];

//         switch (dim)
//         {
//             case 2:
//                 return (sin(3.0*M_PI*x)*sin(3.0*M_PI*y)+5.0)/(2.0*(1.0+nu));
//                 break;
//             case 3:
//                 //return exp(4*x)/(2.0*(1.0+nu));
//                 return 100.0;
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }

//     template <int dim>
//     void LameSecondParameter<dim>::value_list(const std::vector<Point<dim> > &points,
//                                               std::vector<double> &values,
//                                               const unsigned int component) const
//     {
//         Assert(values.size() == points.size(),
//                ExcDimensionMismatch(values.size(), points.size()));

//         const unsigned int n_points = points.size();

//         for (unsigned int p=0; p<n_points; ++p)
//             values[p] = LameSecondParameter<dim>::value(points[p]);
//     }

//     // Right hand side values, boundary conditions and exact solution
//     template <int dim>
//     class RightHandSideElasticity : public Function<dim>
//     {
//     public:
//         RightHandSideElasticity () : Function<dim>(dim) {}

//         virtual void vector_value (const Point<dim> &p,
//                                    Vector<double>   &values) const;
//         virtual void vector_value_list (const std::vector<Point<dim> >   &points,
//                                         std::vector<Vector<double> > &value_list) const;
//     };

//     template <int dim>
//     inline
//     void RightHandSideElasticity<dim>::vector_value(const Point<dim> &p,
//                                           Vector<double>   &values) const
//     {
//         Assert(values.size() == dim,
//                ExcDimensionMismatch(values.size(),dim));
//         Assert(dim != 1, ExcNotImplemented());

//         double x = p[0];
//         double y = p[1];
//         double z;

//         if (dim == 3)
//             z = p[2];

//         const LameFirstParameter<dim> lmbda_function;
//         const LameSecondParameter<dim> mu_function;

//         const double lmbda = lmbda_function.value(p);
//         const double mu = mu_function.value(p);

//         switch (dim)
//         {
//             case 2:
//                 values(0) = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*(x*(y*y*y*y)*6.0-(y*y)*sin(x*y)*cos(y)+2.0)+(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*(sin(x*y)*sin(x)*(1.0/2.0)-(x*x*x)*(y*y)*1.2E1+sin(x*y)*cos(y)*(1.0/2.0)+x*sin(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*sin(y)+(x*x)*sin(x*y)*cos(y)*(1.0/2.0)+x*y*cos(x*y)*sin(x)*(1.0/2.0))+(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/1.2E1)+2.5E1/1.2E1)*(sin(x*y)*sin(x)-(x*x*x)*(y*y)*1.2E1-x*(y*y*y*y)*6.0+x*sin(x*y)*cos(x)+(y*y)*sin(x*y)*cos(y)+x*y*cos(x*y)*sin(x)-2.0)-M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0)-M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*(x*2.0+(x*x)*(y*y*y*y)*3.0+y*cos(x*y)*cos(y))*(5.0/2.0)-M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*(x*2.0+y*2.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x))*(5.0/4.0);
//                 values(1) = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x*x*x)*y*6.0-(x*x)*cos(x*y)*sin(x)+2.0)+(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x)*(y*y*y)*-1.2E1-cos(x*y)*cos(y)*(1.0/2.0)+cos(x*y)*sin(x)*(1.0/2.0)+y*sin(x*y)*cos(x)+y*cos(x*y)*sin(y)*(1.0/2.0)+(y*y)*cos(x*y)*sin(x)*(1.0/2.0)+x*y*sin(x*y)*cos(y)*(1.0/2.0))-(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/1.2E1)+2.5E1/1.2E1)*((x*x)*(y*y*y)*1.2E1+(x*x*x*x)*y*6.0+cos(x*y)*cos(y)-y*cos(x*y)*sin(y)-(x*x)*cos(x*y)*sin(x)-x*y*sin(x*y)*cos(y)+2.0)-M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0)-M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*(y*2.0+(x*x*x*x)*(y*y)*3.0-x*sin(x*y)*sin(x))*(5.0/2.0)-M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*(x*2.0+y*2.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x))*(5.0/4.0);
//                 break;
//             case 3:
// //        values(0) = exp(x*4.0)*(cos(M_PI/12.0)*8.0E1+exp(x)*1.3E2-cos(M_PI/12.0)*exp(x)*1.3E2+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*3.0+exp(x)*sin(M_PI*y)*sin(M_PI*z)*2.0E1-(M_PI*M_PI)*exp(x)*sin(M_PI*y)*sin(M_PI*z)*3.0-8.0E1)*(1.0/3.6E1);
// //        values(1) = exp(x*4.0)*(exp(x)*7.5E1-cos(M_PI/12.0)*exp(x)*7.5E1+sin(M_PI/12.0)*exp(x)*7.5E1-y*exp(x)*1.5E2+cos(M_PI/12.0)*y*exp(x)*1.5E2-sin(M_PI/12.0)*z*exp(x)*1.5E2+M_PI*cos(M_PI*y)*sin(M_PI*z)*1.2E1-M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*1.7E1)*(-1.0/7.2E1);
// //        values(2) = exp(x*4.0)*(exp(x)*7.5E1-cos(M_PI/12.0)*exp(x)*7.5E1-sin(M_PI/12.0)*exp(x)*7.5E1-z*exp(x)*1.5E2+cos(M_PI/12.0)*z*exp(x)*1.5E2+sin(M_PI/12.0)*y*exp(x)*1.5E2+M_PI*cos(M_PI*z)*sin(M_PI*y)*1.2E1-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*1.7E1)*(-1.0/7.2E1);
//                 values(0) = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-4.0+lmbda*exp(x)*(cos(M_PI/12.0)*-2.0E1+sin(M_PI*y)*sin(M_PI*z)+2.0E1)*(1.0/1.0E1)+mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                 values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*2.0+M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1);
//                 values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*-2.0+M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/1.0E1);
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }

//     template <int dim>
//     void RightHandSideElasticity<dim>::vector_value_list(const std::vector<Point<dim> > &points,
//                                                std::vector<Vector<double> >   &value_list) const
//     {
//         Assert(value_list.size() == points.size(),
//                ExcDimensionMismatch(value_list.size(), points.size()));

//         const unsigned int n_points = points.size();

//         for (unsigned int p=0; p<n_points; ++p)
//             RightHandSideElasticity<dim>::vector_value(points[p], value_list[p]);
//     }

//     // Boundary conditions (natural)
//     template <int dim>
//     class DisplacementBoundaryValues : public Function<dim>
//     {
//     public:
//         DisplacementBoundaryValues() : Function<dim>(dim) {}

//         virtual void vector_value (const Point<dim> &p,
//                                    Vector<double>   &values) const;
//         virtual void vector_value_list (const std::vector<Point<dim> >   &points,
//                                         std::vector<Vector<double> > &value_list) const;
//     };

//     template <int dim>
//     void DisplacementBoundaryValues<dim>::vector_value (const Point<dim> &p,
//                                                         Vector<double>   &values) const
//     {
//         double x = p[0];
//         double y = p[1];
//         double z;

//         if (dim == 3)
//             z = p[2];

//         const LameFirstParameter<dim> lmbda_function;
//         const LameSecondParameter<dim> mu_function;

//         const double lmbda = lmbda_function.value(p);
//         const double mu = mu_function.value(p);

//         switch (dim)
//         {
//             case 2:
//                 values(0) = (x*x*x)*(y*y*y*y)+x*x+sin(x*y)*cos(y);
//                 values(1) = (x*x*x*x)*(y*y*y)+y*y+cos(x*y)*sin(x);
//                 break;
//             case 3:
// //        values(0) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
// //        values(1) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
// //        values(2) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
//                 values(0) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
//                 values(1) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
//                 values(2) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }

//     template <int dim>
//     void DisplacementBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim> > &points,
//                                                             std::vector<Vector<double> >   &value_list) const
//     {
//         Assert(value_list.size() == points.size(),
//                ExcDimensionMismatch(value_list.size(), points.size()));

//         const unsigned int n_points = points.size();

//         for (unsigned int p=0; p<n_points; ++p)
//             DisplacementBoundaryValues<dim>::vector_value(points[p], value_list[p]);
//     }

//     // Exact solution
//     template <int dim>
//     class ExactSolution : public Function<dim>
//     {
//     public:
//         ExactSolution() : Function<dim>(dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim) {}

//         virtual void vector_value (const Point<dim> &p,
//                                    Vector<double>   &value) const;
//         virtual void vector_gradient (const Point<dim> &p,
//                                       std::vector<Tensor<1,dim,double> > &grads) const;
//     };

//     template <int dim>
//     void
//     ExactSolution<dim>::vector_value (const Point<dim> &p,
//                                       Vector<double>   &values) const
//     {
//         double x = p[0];
//         double y = p[1];
//         double z;

//         if (dim == 3)
//             z = p[2];

//         const LameFirstParameter<dim> lmbda_function;
//         const LameSecondParameter<dim> mu_function;

//         const double lmbda = lmbda_function.value(p);
//         const double mu = mu_function.value(p);

//         switch (dim)
//         {
//             case 2:
//                 // Stress Test 1
//                 values(0) = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*(x*6.0+y*2.0+(x*x)*(y*y*y*y)*9.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)*3.0-x*sin(x*y)*sin(x))*(5.0/1.2E1);
//                 values(1) = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0));
//                 values(2) = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0));
//                 values(3) = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*(x*2.0+y*6.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*9.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x)*3.0)*(5.0/1.2E1);
//                 // Displacement
//                 values(4) = (x*x*x)*(y*y*y*y)+x*x+sin(x*y)*cos(y);
//                 values(5) = (x*x*x*x)*(y*y*y)+y*y+cos(x*y)*sin(x);
//                 // Rotation
//                 values(6) = sin(x*y)*sin(y)*(-1.0/2.0)-cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)+y*sin(x*y)*sin(x)*(1.0/2.0);
//                 break;
//             case 3:
// //        // Stress
// //        values(0) = exp(x*4.0)*(cos(M_PI/12.0)*5.0+exp(x)*5.0-cos(M_PI/12.0)*exp(x)*5.0+exp(x)*sin(M_PI*y)*sin(M_PI*z)-5.0)*(-1.0/9.0);
// //        values(1) = exp(x*4.0)*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*(-5.0/6.0);
// //        values(2) = exp(x*4.0)*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*(5.0/6.0);
// //        values(3) = exp(x*4.0)*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*(-5.0/6.0);
// //        values(4) = exp(x*4.0)*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))*(5.0/1.8E1)+exp(x*4.0)*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*(5.0/6.0);
// //        values(5) = 0;
// //        values(6) = exp(x*4.0)*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*(5.0/6.0);
// //        values(7) = 0;
// //        values(8) = exp(x*4.0)*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))*(5.0/1.8E1)+exp(x*4.0)*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*(5.0/6.0);
// //        // Displacement
// //        values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
// //        values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
// //        values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
// //        // Rotation
// //        values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
// //        values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
// //        values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
//                 values(0) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                 values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                 values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 values(3) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                 values(4) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;
//                 values(5) = 0;
//                 values(6) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 values(7) = 0;
//                 values(8) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;

//                 values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
//                 values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
//                 values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);

//                 values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
//                 values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
//                 values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }

//     template <int dim>
//     void
//     ExactSolution<dim>::vector_gradient (const Point<dim> &p,
//                                          std::vector<Tensor<1,dim,double> > &grads) const
//     {
//         double x = p[0];
//         double y = p[1];
//         double z;

//         if (dim == 3)
//             z = p[2];

//         const LameFirstParameter<dim> lmbda_function;
//         const LameSecondParameter<dim> mu_function;

//         const double lmbda = lmbda_function.value(p);
//         const double mu = mu_function.value(p);

//         int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));
//         Tensor<1,dim> tmp;
//         switch (dim)
//         {
//             case 2:
//                 // sigma_11 Test1
//                 tmp[0] = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*(sin(x*y)*sin(x)-(x*x*x)*(y*y)*1.2E1-x*(y*y*y*y)*1.8E1+x*sin(x*y)*cos(x)+(y*y)*sin(x*y)*cos(y)*3.0+x*y*cos(x*y)*sin(x)-6.0)*(-5.0/1.2E1)+M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*(x*6.0+y*2.0+(x*x)*(y*y*y*y)*9.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)*3.0-x*sin(x*y)*sin(x))*(5.0/4.0);
//                 tmp[1] = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*((x*x)*(y*y*y)*3.6E1+(x*x*x*x)*y*6.0+cos(x*y)*cos(y)*3.0-y*cos(x*y)*sin(y)*3.0-(x*x)*cos(x*y)*sin(x)-x*y*sin(x*y)*cos(y)*3.0+2.0)*(5.0/1.2E1)+M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*(x*6.0+y*2.0+(x*x)*(y*y*y*y)*9.0+(x*x*x*x)*(y*y)*3.0+y*cos(x*y)*cos(y)*3.0-x*sin(x*y)*sin(x))*(5.0/4.0);
//                 grads[0] = tmp;
//                 // sigma_12
//                 tmp[0] = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x)*(y*y*y)*-1.2E1-cos(x*y)*cos(y)*(1.0/2.0)+cos(x*y)*sin(x)*(1.0/2.0)+y*sin(x*y)*cos(x)+y*cos(x*y)*sin(y)*(1.0/2.0)+(y*y)*cos(x*y)*sin(x)*(1.0/2.0)+x*y*sin(x*y)*cos(y)*(1.0/2.0))+M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0);
//                 tmp[1] = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*(sin(x*y)*sin(x)*(1.0/2.0)-(x*x*x)*(y*y)*1.2E1+sin(x*y)*cos(y)*(1.0/2.0)+x*sin(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*sin(y)+(x*x)*sin(x*y)*cos(y)*(1.0/2.0)+x*y*cos(x*y)*sin(x)*(1.0/2.0))+M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0);
//                 grads[1] = tmp;
//                 // sigma_21
//                 tmp[0] = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*((x*x)*(y*y*y)*-1.2E1-cos(x*y)*cos(y)*(1.0/2.0)+cos(x*y)*sin(x)*(1.0/2.0)+y*sin(x*y)*cos(x)+y*cos(x*y)*sin(y)*(1.0/2.0)+(y*y)*cos(x*y)*sin(x)*(1.0/2.0)+x*y*sin(x*y)*cos(y)*(1.0/2.0))+M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0);
//                 tmp[1] = -(sin(M_PI*x*3.0)*sin(M_PI*y*3.0)*(5.0/6.0)+2.5E1/6.0)*(sin(x*y)*sin(x)*(1.0/2.0)-(x*x*x)*(y*y)*1.2E1+sin(x*y)*cos(y)*(1.0/2.0)+x*sin(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*sin(y)+(x*x)*sin(x*y)*cos(y)*(1.0/2.0)+x*y*cos(x*y)*sin(x)*(1.0/2.0))+M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*((x*x*x)*(y*y*y)*4.0-sin(x*y)*sin(y)*(1.0/2.0)+cos(x*y)*cos(x)*(1.0/2.0)+x*cos(x*y)*cos(y)*(1.0/2.0)-y*sin(x*y)*sin(x)*(1.0/2.0))*(5.0/2.0);
//                 grads[2] = tmp;
//                 // sigma_22
//                 tmp[0] = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*(sin(x*y)*sin(x)*3.0-(x*x*x)*(y*y)*3.6E1-x*(y*y*y*y)*6.0+x*sin(x*y)*cos(x)*3.0+(y*y)*sin(x*y)*cos(y)+x*y*cos(x*y)*sin(x)*3.0-2.0)*(-5.0/1.2E1)+M_PI*cos(M_PI*x*3.0)*sin(M_PI*y*3.0)*(x*2.0+y*6.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*9.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x)*3.0)*(5.0/4.0);
//                 tmp[1] = (sin(M_PI*x*3.0)*sin(M_PI*y*3.0)+5.0)*((x*x)*(y*y*y)*1.2E1+(x*x*x*x)*y*1.8E1+cos(x*y)*cos(y)-y*cos(x*y)*sin(y)-(x*x)*cos(x*y)*sin(x)*3.0-x*y*sin(x*y)*cos(y)+6.0)*(5.0/1.2E1)+M_PI*cos(M_PI*y*3.0)*sin(M_PI*x*3.0)*(x*2.0+y*6.0+(x*x)*(y*y*y*y)*3.0+(x*x*x*x)*(y*y)*9.0+y*cos(x*y)*cos(y)-x*sin(x*y)*sin(x)*3.0)*(5.0/4.0);
//                 grads[3] = tmp;
//                 // The gradient for the rest is meaningless
//                 tmp[0] = 0.0;
//                 tmp[1] = 0.0;
//                 for (int k=dim*dim;k<total_dim;++k)
//                     grads[k] = tmp;

//                 break;
//             case 3:
//                 tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                 tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1)-M_PI*mu*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                 tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1)-M_PI*mu*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/5.0);
//                 grads[0] = tmp;

//                 tmp[0] = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*-2.0;
//                 tmp[1] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 tmp[2] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)+(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                 grads[1] = tmp;

//                 tmp[0] = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*2.0;
//                 tmp[1] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)-(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 tmp[2] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 grads[2] = tmp;

//                 tmp[0] = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(1.0/2.0E1))*-2.0;
//                 tmp[1] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 tmp[2] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)+(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                 grads[3] = tmp;

//                 tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*exp(x)*(cos(M_PI/12.0)-1.0)*2.0;
//                 tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1);
//                 tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1);
//                 grads[4] = tmp;

//                 tmp[0] = 0;
//                 tmp[1] = 0;
//                 tmp[2] = 0;
//                 grads[5] = tmp;

//                 tmp[0] = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(1.0/2.0E1))*2.0;
//                 tmp[1] = mu*(sin(M_PI/12.0)*exp(x)*(1.0/2.0)-(M_PI*M_PI)*cos(M_PI*y)*cos(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 tmp[2] = mu*(exp(x)*(cos(M_PI/12.0)-1.0)*(1.0/2.0)+(M_PI*M_PI)*sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                 grads[6] = tmp;

//                 tmp[0] = 0;
//                 tmp[1] = 0;
//                 tmp[2] = 0;
//                 grads[7] = tmp;

//                 tmp[0] = lmbda*(exp(x)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*exp(x)*(cos(M_PI/12.0)-1.0)*2.0;
//                 tmp[1] = M_PI*lmbda*exp(x)*cos(M_PI*y)*sin(M_PI*z)*(-1.0/1.0E1);
//                 tmp[2] = M_PI*lmbda*exp(x)*cos(M_PI*z)*sin(M_PI*y)*(-1.0/1.0E1);
//                 grads[8] = tmp;

//                 // The gradient for the rest is meaningless
//                 tmp[0] = 0.0;
//                 tmp[1] = 0.0;
//                 tmp[2] = 0.0;
//                 for (int k=dim*dim;k<total_dim;++k)
//                     grads[k] = tmp;

//                 break;
//             default:
//             Assert(false, ExcNotImplemented());
//         }
//     }
}

#endif //ELASTICITY_MFEDD_DATA_H
