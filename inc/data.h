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
//          return c0*t_scale*exp(t*t_scale)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1)-exp(t*t_scale)*(y*-2.0+pow(x-1.0,4.0)*pow(y-1.0,2.0)*3.0+x*sin(x*y)*sin(x)+2.0)+(3.141592653589793*3.141592653589793)*exp(t*t_scale)*cos(y*3.141592653589793)*sin(x*3.141592653589793)*2.0;
//        	return c0*t_scale*exp(t*t_scale)*sin(p[0])*sin(2.0*p[1]) + exp(t*t_scale)*5.0*sin(p[0])*sin(2.0*p[1]);
//        	return 8*cos(8*t)*sin(3*x)*sin(4*y)+sin(8*t)*25*sin(3*x)*sin(4*y);
        	return 8*cos(8*t)*sin(11*x)*cos(11*y)+sin(8*t)*242*sin(11*x)*cos(11*y);
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
//          return exp(t*t_scale)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1);
//        	return exp(t*t_scale)*sin(x)*sin(2.0*y);
//        	return sin(8*t)*sin(3*x)*sin(4*y);
        	return sin(8*t)*sin(11*x)*cos(11*y);
        case 3:
//        	return sin(8*z)*sin(3*x)*sin(4.0*y);
        	return sin(8*z)*sin(11*x)*cos(11*y);
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
//                values(0) = -3.141592653589793*exp(t*t_scale)*cos(x*3.141592653589793)*cos(y*3.141592653589793);
//                values(1) = 3.141592653589793*exp(t*t_scale)*sin(x*3.141592653589793)*sin(y*3.141592653589793);
//                values(2) = exp(t*t_scale)*(cos(y*3.141592653589793)*sin(x*3.141592653589793)+1.0E1);

//            	  values(0) = -sin(8*t)*3*cos(3*x)*sin(4*y) ;
//            	  values(1) = -sin(8*t)*4*sin(3*x)*cos(4*y) ;
//            	  values(2) = sin(8*t)*sin(3*x)*sin(4*y);
          	  values(0) = -sin(8*t)*11*cos(11*x)*cos(11*y) ;
          	  values(1) = sin(8*t)*11*sin(11*x)*sin(11*y) ;
          	  values(2) = sin(8*t)*sin(11*x)*cos(11*y);


                break;
            case 3:
//                values(0) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))-mu*exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/5.0);
//                values(1) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                values(2) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                values(3) = mu*(exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*-2.0;
//                values(4) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;
//                values(5) = 0;
//                values(6) = mu*(exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0))*2.0;
//                values(7) = 0;
//                values(8) = lmbda*((exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0-exp(x)*sin(M_PI*y)*sin(M_PI*z)*(1.0/1.0E1))+mu*(exp(x)-1.0)*(cos(M_PI/12.0)-1.0)*2.0;
//
//                values(9) = -sin(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1);
//                values(10) = -(exp(x)-1.0)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0);
//                values(11) = (exp(x)-1.0)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0);
//
//                values(12) = sin(M_PI/12.0)*(exp(x)-1.0);
//                values(13) = exp(x)*(-z+cos(M_PI/12.0)*(z-1.0/2.0)+sin(M_PI/12.0)*(y-1.0/2.0)+1.0/2.0)*(-1.0/2.0)-M_PI*cos(M_PI*z)*sin(M_PI*y)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
//                values(14) = exp(x)*(y-cos(M_PI/12.0)*(y-1.0/2.0)+sin(M_PI/12.0)*(z-1.0/2.0)-1.0/2.0)*(-1.0/2.0)+M_PI*cos(M_PI*y)*sin(M_PI*z)*(exp(x)*(1.0/1.0E1)-1.0/1.0E1)*(1.0/2.0);
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

        Vector<double> vec(2);
        const double lambda = vec[0];
        const double mu = vec[1];
        double t = FunctionTime<double>::get_time();

        int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));
        Tensor<1,dim> tmp;
        switch (dim)
        {
        case 2:


//            grads[0][0] = (3.141592653589793*3.141592653589793)*exp(t*t_scale)*cos(y*3.141592653589793)*sin(x*3.141592653589793);
//            grads[0][1] = (3.141592653589793*3.141592653589793)*exp(t*t_scale)*cos(x*3.141592653589793)*sin(y*3.141592653589793);
//
//            grads[1][0] = (3.141592653589793*3.141592653589793)*exp(t*t_scale)*cos(x*3.141592653589793)*sin(y*3.141592653589793);
//            grads[1][1] = (3.141592653589793*3.141592653589793)*exp(t*t_scale)*cos(y*3.141592653589793)*sin(x*3.141592653589793);

//        	 grads[0][0] = sin(8*t)*9*sin(3*x)*sin(4*y);
//        	 grads[0][1] = -sin(8*t)*12.0*cos(3*x)*cos(4*y);
//
//        	 grads[1][0] = -sin(8*t)*12.0*cos(3*x)*cos(4*y);
//        	 grads[1][1] = sin(8*t)*16*sin(3*x)*sin(4*y);

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
//              values(2) = (cos(y*M_PI)*sin(x*M_PI)+1.0E1);

//              values(2) = sin(8*t)*sin(3*x)*sin(4*y);
              values(2) = sin(8*t)*sin(11*x)*cos(11*y);
          break;
          case 3:

              break;
          default:
            Assert(false, ExcNotImplemented());
      }
  }


}

#endif //ELASTICITY_MFEDD_DATA_H
