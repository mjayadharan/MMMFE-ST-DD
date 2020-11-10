/* ---------------------------------------------------------------------
 * Functions representing RHS, physical parameters, boundary conditions and
 * the true solution for the space-time DD for time-dependent parabolic equations.
 * All conditions and RHS are derived using p(x,y,t)= sin(8t)sin(11x)cos(11y-pi/4)
 * and permeability tensor K = I_2x2.
 * ---------------------------------------------------------------------
 *
 * Author: Manu Jayadharan, Eldar Khattatov, University of Pittsburgh, 2020
 */

#ifndef ELASTICITY_MFEDD_DATA_H
#define ELASTICITY_MFEDD_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

namespace vt_darcy
{
    using namespace dealii;





    const double t_scale = 1; //time scaling to adjust the evolution of velocity and pressure with time.

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

    {
      const double x = p[0];
      const double y = p[1];
      double t = FunctionTime<double>::get_time();

      switch (dim)
      {
        case 2:
        	return 8*cos(8*t)*sin(11*x)*cos(11*y-(3.1415/4))+sin(8*t)*242*sin(11*x)*cos(11*y-(3.1415/4));
        default:
        Assert(false, ExcMessage("The RHS data for dim != 2 is not provided"));
      }
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
      double z;
      if (dim == 3)
          z = p[2];
      double t = FunctionTime<double>::get_time();

      switch (dim)
      {
        case 2:

        	return sin(8*t)*sin(11*x)*cos(11*y-(3.1415/4));
        case 3:
        	return sin(8*z)*sin(11*x)*cos(11*y-(3.1415/4));
        default:
        Assert(false, ExcMessage("The BC data for dim != 2 is not provided"));
      }
    }

    // Exact solution
    template <int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution() : Function<dim>(static_cast<unsigned int>(dim + 1)) {}

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

        double t = FunctionTime<double>::get_time();

        switch (dim)
        {
            case 2:
            	  values(0) = -sin(8*t)*11*cos(11*x)*cos(11*y-(3.1415/4)) ;
            	  values(1) =  sin(8*t)*11*sin(11*x)*sin(11*y-(3.1415/4)) ;
            	  values(2) = sin(8*t)*sin(11*x)*cos(11*y-(3.1415/4));
                break;
            case 3:

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

        double t = FunctionTime<double>::get_time();

        int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));
        Tensor<1,dim> tmp;
        switch (dim)
        {
        case 2:



        	 grads[0][0] = sin(8*t)*36*sin(11*x)*cos(11*y);
           	 grads[0][1] = -sin(8*t)*36.0*cos(6*x)*cos(6*y);

           	 grads[1][0] = -sin(8*t)*36.0*cos(6*x)*cos(6*y);
           	 grads[1][1] = sin(8*t)*36*sin(11*x)*cos(11*y);

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
    InitialCondition() : Function<dim>(static_cast<unsigned int>(dim + 1)) {}

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

      double t = 0;

      switch (dim)
      {
          case 2:
              values(0) = 0;
              values(1) = 0;

              values(2) = sin(8*t)*sin(11*x)*cos(11*y-(3.1415/4));;
          break;
          case 3:

              break;
          default:
            Assert(false, ExcNotImplemented());
      }
  }


}

#endif //ELASTICITY_MFEDD_DATA_H
