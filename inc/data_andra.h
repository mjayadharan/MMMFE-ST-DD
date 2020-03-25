/* ---------------------------------------------------------------------
 * Functions representing RHS, physical parameters, boundary conditions 
 * for the space-time DD for time-dependent parabolic equations.
 * All conditions and RHS are for the Andra test-case from paper by
 * Ali Hassan et al., ETNA , 49 (2018),  doi:10.1553/etna_vol49s151
 * 
 * ---------------------------------------------------------------------
 *
 * Author: Michel Kern (Inria), Manu Jayadharan, Eldar Khattatov, University of Pittsburgh, 2020
 */

#ifndef DARCY_MFEDD_DATA_H
#define DARCY_MFEDD_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

namespace vt_darcy
{
    using namespace dealii;

    // Some useful variables to describe the geometry, and the physical properties

    const double Lx = 3950, Ly = 140; // real dimension of domain
    const double Lxr = 2950, Lyr = 10; // dimensions of repository
    const double sx = 14, sy = 1; // scaling factor in each direction
    const double st = 3.1536e7; // scaling in time (# seconds in one year)
    const double Krepo = 2e-9, Khost = 5e-12; // permeability values
    const double pororepo = 0.2, porohost = 0.05; // porosity;
    const double tsource = 1e5, fsource = 1e-5; // source time and intensity

    // utility function
   template <int dim>
   bool isInRepo(const Point<dim> &p)
   {
       const double x = p[0];
       const double y = p[1];

       return(
	   (Lx/sx - Lxr/sx)/2. < x && x < (Lx/sx + Lxr/sx)/2  &&
	   (Ly/sy - Lyr/sy)/2. < y && y < (Ly/sy + Lyr/sy)/2
	   );
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
	    values[p][0][0] = isInRepo(points[p]) ? Krepo / pororepo*(st /sx*sx) : Khost / porohost *(st/sx*sx);
            values[p][0][1] = 0.0;
            values[p][1][0] = 0.0;
            values[p][1][1] = isInRepo(points[p]) ? Krepo / pororepo*(st /sy*sy) : Khost / porohost *(st/sy*sy);
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
	    return (isInRepo(p) && t < tsource) ? fsource *st / pororepo : 0; 
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

	    return 0;
        case 3:
	    return 0;
        default:
        Assert(false, ExcMessage("The BC data for dim != 2 is not provided"));
      }
    }


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

              values(2) = 0;
          break;
          case 3:

              break;
          default:
            Assert(false, ExcNotImplemented());
      }
  }


}

#endif //DARCY_MFEDD_DATA_H
