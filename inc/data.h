/* ---------------------------------------------------------------------
 * Functions representing RHS, physical parameters, boundary conditions and
 * the true solution for the space-time DD for time-dependent parabolic equations.
 * All conditions and RHS are derived using p(x,y,t)= 1000xyt*exp( -10(x^2 + y^2 + (at)^2) )
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
      RightHandSidePressure (const double c0=1.0, const double alpha=1.0, double coe_a_in=1.0);

      virtual double value (const Point<dim>   &p,
                            const unsigned int component = 0 ) const;

    private:
      double c0; //=1.0;
      double alpha ;//=1.0;
      double coe_a; // coefficient for controlling variation in t.
    };

    template <int dim>
      RightHandSidePressure<dim>::RightHandSidePressure (const double c0, const double alpha, double coe_a_in) :
	  Function<dim>(1),
	  c0(c0),
    alpha(alpha),
	coe_a(coe_a_in)
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
        	return 1000.0*(x*y*exp(-10*(x*x + y*y))) * (exp(-10*coe_a*coe_a*t*t)*(1-20*coe_a*coe_a*t*t))
        			-1000.0*(x*y*t*exp(-10*(x*x + y*y + coe_a*coe_a*t*t )))
					*(-120.0 + 400*(x*x + y*y));
        default:
        Assert(false, ExcMessage("The RHS data for dim != 2 is not provided"));
      }
    }



    template <int dim>
    class PressureBoundaryValues : public Function<dim>
    {
    public:
      PressureBoundaryValues () :
    	  Function<dim>(1),
		  coe_a(1.0)
		  {}
      PressureBoundaryValues(double coe_a);

      double coe_a; //coefficient for controlling variation in t.
      virtual double value (const Point<dim>   &p,
                            const unsigned int component = 0) const;
    };

    template<int dim>
    PressureBoundaryValues<dim>::PressureBoundaryValues(double coe_a_in):
		Function<dim>(1),
		coe_a(coe_a_in)
	{}


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

        	return 1000 * (t*exp(-10*coe_a*coe_a*t*t)) * (x*y*exp( -10*(x*x + y*y) )); //1000*t*exp(-10t^2)*exp(-10(x^2+y^2))
        case 3:
        	return 1000 * (z*exp(-10*coe_a*coe_a*z*z)) * (x*y*exp( -10*(x*x + y*y) ));
        default:
        Assert(false, ExcMessage("The BC data for dim != 2 is not provided"));
      }
    }

    // Exact solution
    template <int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution(double coe_a_in) :
        	Function<dim>(static_cast<unsigned int>(dim + 1)),
			coe_a(coe_a_in)
			{}
        ExactSolution() :
                	Function<dim>(static_cast<unsigned int>(dim + 1)),
        			coe_a(1.0)
        			{}
        double coe_a; //coefficient for controlling variation in t.
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
            	  values(0) = -1000 * (t*exp(-10*coe_a*coe_a*t*t)) * (y*exp( -10*(x*x + y*y) ) * (1 - 20*x*x));
            	  values(1) =  -1000 * (t*exp(-10*coe_a*coe_a*t*t)) * (x*exp( -10*(x*x + y*y) ) * (1 - 20*y*y));
            	  values(2) = 1000 * (t*exp(-10*coe_a*coe_a*t*t)) * (x*y*exp( -10*(x*x + y*y) )); //1000*t*exp(-10t^2)*exp(-10(x^2+y^2))

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
    InitialCondition(double coe_a_in) :
    	Function<dim>(static_cast<unsigned int>(dim + 1)),
		coe_a(coe_a_in)
		{}

    double coe_a;
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

              values(2) = 1000 * (t*exp(-10*coe_a*coe_a*t*t)) * (x*y*exp( -10*(x*x + y*y) )); //1000*t*exp(-10t^2)*exp(-10(x^2+y^2))
          break;
          case 3:

              break;
          default:
            Assert(false, ExcNotImplemented());
      }
  }


}

#endif //ELASTICITY_MFEDD_DATA_H
