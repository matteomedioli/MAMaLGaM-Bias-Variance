/*
AMaLGaM

Implementation by S.C. Maree, 2017
s.c.maree[at]amc.uva.nl
smaree.com

*/

#include "../domination_based_MO_optimization/mohillvallea/fitness.h"
#include <unistd.h>
double x[50] = {-2.13876,-0.843971,-2.77733,0.969594,0.948479,0.0596248,-2.81538,-2.68752,-2.66449,1.43964,-1.82261,0.75355,
                    1.77127,1.50504,0.563664,0.934696,-0.964971,-0.0580217,-1.03172,1.74479,0.949619,-0.235184,-1.03222,
                    -2.34456,-1.96341,1.3474,-1.99459,-0.616371,1.23675,-0.115709,-0.620406,-0.313958,-2.91901,-0.51485,
                    -1.23434,1.8188,-2.15125,1.94123,-1.34461,1.81213,-2.54684,-2.63072,0.402378,-2.42504,0.166079,-1.07568,
                    -1.61749,-2.91959,-1.71093,1.08362};
double y[50]= {-10.4758,-4.80274,-23.3352,-1.80149,5.61443,1.73721,-33.2348,-32.3649,-27.0149,14.9916,-9.4299,-12.8493,12.8735,
                  2.71406,3.78351,2.43841,-6.41437,-10.7886,-0.705705,3.52402,10.0806,-9.5326,-0.447625,-14.892,-11.4165,8.8522,-7.94916,
                  0.822361,2.71519,3.67672,0.177861,-2.70773,-32.1672,-7.76417,-7.15321,6.14037,-21.8601,-1.02493,-2.55413,16.949,-23.3274,
                  -29.4496,12.4751,-18.4906,1.72832,1.86347,-2.40768,-27.5401,-1.73747,-1.90976};
namespace hicam
{

  class RBF_t : public fitness_t
  {

  public:

    RBF_t()
    {
      number_of_objectives = 2; // fixed
      number_of_parameters = 2; // default, can be adapted
      number_of_radial;  // HARDCODED 3 Radial Basis Functions, TODO
      hypervolume_max_f0 = 400;
      hypervolume_max_f1 = 400;

    }
    ~RBF_t() {}

    // number of objectives
    void set_number_of_objectives(size_t & number_of_objectives)
    {
      this->number_of_objectives = 2;
      number_of_objectives = this->number_of_objectives;
    }

    void set_number_of_radial(size_t & number_of_radial)
    {

      this->number_of_radial = number_of_radial;
    }

    // any positive value
    void set_number_of_parameters(size_t & number_of_parameters)
    {

      this->number_of_parameters = number_of_parameters;
    }


    void get_param_bounds(vec_t & lower, vec_t & upper) const
    {
      lower.clear();
      upper.clear();
      lower.resize(number_of_parameters, -100.0);
      upper.resize(number_of_parameters, 100.0);
      
      for(int i = 0; i<this->number_of_parameters; i+=3)
      {
        lower[i] = -10.0;
        lower[i+1] = -10.0;
        lower[i+2] = 0.1;

        upper[i] = 10.0;
        upper[i+1] = 10.0;
        upper[i+2] = 100.0;
      }
      /*
      for(int i = 0; i<this->number_of_parameters; i+=3)
        std::cout<<lower[i]<<" "<<lower[i+1]<<" "<<lower[i+2];
      std::cout<<std::endl;
      for(int i = 0; i<this->number_of_parameters; i+=3)
        std::cout<<upper[i]<<" "<<upper[i+1]<<" "<<upper[i+2];
      std::cout<<std::endl;
      */

    }

    void define_problem_evaluation(solution_t & sol)
    {

        // f1
        //3 fuzioni gaussine con muk sigmak omega k
        // 9 parametri


        int K = this->number_of_radial;
        int num_data = 50;
        double rad, result;
        result = 0.0;
        rad = 0.0;
        for(int n = 0; n < num_data; n++)
        {
          /*
          for(int k = 0; k < this->number_of_parameters; k++)
            std::cout<<" "<<sol.param[k];
          std::cout<<std::endl;
          */
            for(int k = 0; k < this->number_of_parameters; k+=3)
            {
                    //std::cout<<"W: "<<sol.param[k]<<std::endl;
                    //std::cout<<"MU: "<<sol.param[k+1]<<std::endl;
                    //std::cout<<"SIGMA: "<<sol.param[k+2]<<std::endl;
                    //std::cout<<std::endl;
                rad += sol.param[k] * exp ((-(pow(x[n] - sol.param[k+1],2))) / (2*pow(sol.param[k+2], 2)));
            }
            result += pow(fabs(y[n] - rad), 2);
        }
        sol.obj[0] = result/num_data;
        // f2
        result = 0.0;
        for(int k = 0; k < this->number_of_parameters; k+=3)
        {
            //std::cout<<"W"<<k<<": "<<sol.param[k]<<std::endl;
            result += pow(fabs(sol.param[k]), 2);
        }
      sol.obj[1] = result/K;
      sol.constraint = 0;
    }

    std::string name() const
    {
      return "RBF";
    }

    // compute VTR in terms of the D_{\mathcal{P}_F}\rightarrow\mathcal{S}
    bool get_pareto_set()
    {

      size_t pareto_set_size = 5000;

      // generate default front
      if (pareto_set.size() != pareto_set_size)
      {

        pareto_set.sols.clear();
        pareto_set.sols.reserve(pareto_set_size);

        // the front
        for (size_t i = 0; i < pareto_set_size; ++i)
        {
          solution_pt sol = std::make_shared<solution_t>(number_of_parameters, number_of_objectives);

          sol->param.fill(0.0);
          sol->param[0] = (i / ((double)pareto_set_size - 1.0));

          define_problem_evaluation(*sol); // runs a feval without registering it.

          pareto_set.sols.push_back(sol);
        }

        igdx_available = true;
        igd_available = true;

      }

      return true;
    }

  };
}
