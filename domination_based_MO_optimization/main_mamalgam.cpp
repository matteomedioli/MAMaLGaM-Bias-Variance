/*

HillVallEA

Real-valued Multi-Modal Evolutionary Optimization

By S.C. Maree
s.c.maree[at]amc.uva.nl
github.com/SCMaree/HillVallEA

Example script to demonstrate the usage of HillVallEA
 on the well-known 2D Six Hump Camel Back function

*/

// for MO problems
#include "./mohillvallea/hicam_external.h"
#include "./gomea/MO-RV-GOMEA.h"
#include "../benchmark_functions/mo_benchmarks.h"


// all setting variables
int problem_index;
size_t mo_number_of_parameters;
double lower_init;
double upper_init;
size_t popsize;
int random_seed;
size_t elitist_archive_size_target;
size_t approximation_set_size;
int maximum_number_of_mo_evaluations;
int maximum_number_of_seconds;
int use_vtr;
double value_to_reach;
bool write_generational_statistics;
bool write_generational_solutions;
std::string write_directory;
bool enable_niching;
std::string file_appendix;
hicam::fitness_pt mo_fitness_function; 
hicam::vec_t mo_lower_init_ranges, mo_upper_init_ranges;
int variance_RBF_multiplier;
size_t rbf_number_of_radials;
double HL_tol; // no longer used

// IMS population sizing scheme
unsigned int number_of_subgenerations_per_population_factor ;
unsigned int maximum_number_of_populations;
bool print_verbose_overview;
bool print_generational_statistics;

bool optimize_whitebox;
    

void printUsage(void)
{      
  printf("Usage: mamalgam [-?] [-P] [-s] [-n] [-w] [-v] [-r] [-w] pro dim low upp pop ela apprs eva sec vtr varm rnd wrp\n"); // [-e] [-f]
  printf(" -?: Prints out this usage information.\n");
  printf(" -P: Prints out a list of all installed optimization problems.\n");
  printf(" -s: Enables computing and writing of statistics every generation.\n");
  printf(" -w: Enable writing of solutions and their fitnesses every generation.\n");
  printf(" -v: Verbose mode. Prints the settings before starting the run + ouput of generational statistics to terminal.\n");
  printf(" -n: Enable niching \n");
  // printf(" -e: Enable collecting all solutions in an elitist archive\n");
  // printf(" -f: Enforce use of finite differences for gradient-based methods\n");
  printf(" -r: Enables use of vtr (value-to-reach) termination condition based on the hypervolume.\n");
  printf("\n");
  printf("  pro: Multi-objective optimization problem index (minimization).\n");
  printf("  dim: Number of parameters (if the problem is configurable).\n");
  printf("  low: Overall initialization lower bound.\n");
  printf("  upp: Overall initialization upper bound.\n");
  printf("  pop: Population size.\n");
  printf("  ela: Max Elitist archive size target.\n");
  printf("apprs: Approximation set size (reduces size of ela after optimization using gHSS.\n");
  printf("  eva: Maximum number of evaluations of the multi-objective problem allowed.\n");
  printf("  sec: Time limit in seconds.\n");
  printf("  vtr: The value to reach. If the hypervolume of the best feasible solution reaches this value, termination is enforced (if -r is specified).\n");
  printf("  varm: Variance multiplier for RBF whitebox-optimization.\n");
  printf("  rnd: Random seed.\n");
  printf("  wrp: write path.\n");
  exit(0);
  
}

/**
 * Returns the problems installed.
 */
void printAllInstalledProblems(void)
{
  int i = 0;
  
  hicam::fitness_pt objective = getObjectivePointer(i);
  
  std::cout << "Installed optimization problems:\n";
  
  while(objective != nullptr)
  {
    std::cout << std::setw(3) << i << ": " << objective->name() << std::endl;
    
    i++;
    objective = getObjectivePointer(i);
  }
  
  exit(0);
}

/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError(char **argv, int index)
{
  printf("Illegal option: %s\n\n", argv[index]);
  
  printUsage();
}

void parseOptions(int argc, char **argv, int *index)
{
  double dummy;
  
  write_generational_statistics = 0;
  write_generational_solutions = 0;
  print_verbose_overview = 0;
  use_vtr = 0;
  enable_niching = 0;
  
  for (; (*index) < argc; (*index)++)
  {
    if (argv[*index][0] == '-')
    {
      /* If it is a negative number, the option part is over */
      if (sscanf(argv[*index], "%lf", &dummy) && argv[*index][1] != '\0')
        break;
      
      if (argv[*index][1] == '\0')
        optionError(argv, *index);
      else if (argv[*index][2] != '\0')
        optionError(argv, *index);
      else
      {
        switch (argv[*index][1])
        {
          case '?': printUsage(); break;
          case 'P': printAllInstalledProblems(); break;
          case 's': write_generational_statistics = 1; break;
          case 'w': write_generational_solutions = 1; break;
          case 'v': print_verbose_overview = 1; break;
          case 'n': enable_niching = true; break;
          case 'W': optimize_whitebox = true; break;
          // case 'e': collect_all_mo_sols_in_archive = true; break;
          // case 'f': use_finite_differences = true; break;
          case 'r': use_vtr = 1; break; // HV-based vtr (note, use_vtr = 2 is an IGD-based VTR)
          default: optionError(argv, *index);
        }
      }
    }
    else /* Argument is not an option, so option part is over */
      break;
  }
  
}

void parseParameters(int argc, char **argv, int *index)
{
  int noError;
  
  int n_params = 14;
  if ((argc - *index) != n_params)
  {
    printf("Number of parameters is incorrect, require %d parameters (you provided %d).\n\n", n_params, (argc - *index));
    
    printUsage();
  }
  
  noError = 1;
  noError = noError && sscanf(argv[*index + 0], "%d", &problem_index);
  noError = noError && sscanf(argv[*index + 1], "%ld", &rbf_number_of_radials);
  noError = noError && sscanf(argv[*index + 2], "%zd", &mo_number_of_parameters);
  noError = noError && sscanf(argv[*index + 3], "%lf", &lower_init);
  noError = noError && sscanf(argv[*index + 4], "%lf", &upper_init);
  noError = noError && sscanf(argv[*index + 5], "%zd", &popsize);
  noError = noError && sscanf(argv[*index + 6], "%zu", &elitist_archive_size_target);
  noError = noError && sscanf(argv[*index + 7], "%zd", &approximation_set_size);
  noError = noError && sscanf(argv[*index + 8], "%d", &maximum_number_of_mo_evaluations);
  noError = noError && sscanf(argv[*index + 9], "%d", &maximum_number_of_seconds);
  noError = noError && sscanf(argv[*index + 10], "%lf", &value_to_reach);
  noError = noError && sscanf(argv[*index + 11], "%d", &variance_RBF_multiplier);
  noError = noError && sscanf(argv[*index + 12], "%d", &random_seed);
  write_directory = argv[*index + 13];
  if (!noError)
  {
    printf("Error parsing parameters.\n\n");
    printUsage();
  }
  
} 
 
void checkOptions(void)
{
  mo_fitness_function = getObjectivePointer(problem_index);
  if (mo_fitness_function == nullptr)
  {
    printf("\n");
    printf("Error: unknown index for problem (read index %d).", problem_index);
    printf("\n\n");
    
    exit(0);
  }
  
  if (problem_index!=60)
    mo_fitness_function->set_number_of_parameters(mo_number_of_parameters);
  else
  {
    mo_fitness_function->set_number_of_radial(rbf_number_of_radials);
    if (mo_number_of_parameters <= 0)
    {
      printf("\n");
      printf("Error: number of MO parameters <= 0 (read: %d). Require MO number of parameters >= 1.", (int) mo_number_of_parameters);
      printf("\n\n");
      exit(0);
    }
    size_t rbf_param = 3*rbf_number_of_radials;    
    mo_fitness_function->set_number_of_parameters(rbf_param);
    std::cout<<mo_fitness_function->number_of_parameters<<std::endl;
  }

  

  if (mo_number_of_parameters <= 0)
  {
    printf("\n");
    printf("Error: number of MO parameters <= 0 (read: %d). Require MO number of parameters >= 1.", (int) mo_number_of_parameters);
    printf("\n\n");
    
    exit(0);
  }

  if (elitist_archive_size_target <= 0)
  {
    printf("\n");
    printf("Error: elitist archive target size <= 0 (read: %d) ", (int) elitist_archive_size_target);
    printf("\n\n");
    
    exit(0);
  }
  
  if(approximation_set_size <= 0) {
    approximation_set_size = elitist_archive_size_target * 10; // so that its never filtered!
  }
  
  // initialize the init ranges
  mo_lower_init_ranges.resize(mo_fitness_function->number_of_parameters, lower_init);
  mo_upper_init_ranges.resize(mo_fitness_function->number_of_parameters, upper_init);
  
  if(!mo_fitness_function->redefine_random_initialization)
  {
    if(lower_init >= upper_init) {
      printf("\n");
      printf("Error: init range empty (read lower %f, upper, %f)", lower_init, upper_init);
      printf("\n\n");
      
      exit(0);
    }
  }
  
  
  // prepares the fitness function
  mo_fitness_function->get_pareto_set();
  
  HL_tol = 0; // this is no longer used.
  
  // Interleaved multi-start scheme
  // set number of pops larger than 0 to increase popsize over time
  number_of_subgenerations_per_population_factor = 2;
  maximum_number_of_populations = 1;

  
  // File appendix for writing
  std::stringstream ss;
  ss << "_problem" << problem_index << "_run" << std::setw(3) << std::setfill('0') << random_seed;
  file_appendix = ss.str();
  

}



void parseCommandLine(int argc, char **argv)
{
  int index;
  
  index = 1;
  
  parseOptions(argc, argv, &index);
  
  parseParameters(argc, argv, &index);
}

void interpretCommandLine(int argc, char **argv)
{
  
  parseCommandLine(argc, argv);
  checkOptions();
}


// Main: Run the CEC2013 niching benchmark
//--------------------------------------------------------
int main(int argc, char **argv)
{
  
  interpretCommandLine(argc, argv);

  
  // create optimizer
  if(print_verbose_overview)
  {
    std::cout << "Problem settings:\n\tfunction_name = " << mo_fitness_function->name() << "\n\tproblem_index = " << problem_index << "\n\tmo_number_of_parameters = " << mo_fitness_function->number_of_parameters << "\n\tinit_range = [" << lower_init << ", " << upper_init << "]\n\tNumber of radials = " <<mo_fitness_function->number_of_radial<<"\n\tHV reference point = " << mo_fitness_function->hypervolume_max_f0 << ", " << mo_fitness_function->hypervolume_max_f1 << "\n";
    std::cout << "Run settings:\n\tmax_number_of_MO_evaluations = " << maximum_number_of_mo_evaluations <<"\n\tmaximum_number_of_seconds = " << maximum_number_of_seconds << "\n\tuse_vtr = " << use_vtr << "\n\tvtr = " << value_to_reach << "\n";
    std::cout << "Archive settings:\n\tElitist_archive_target_size = " << elitist_archive_size_target << "\n\tApproximation_set_size = " <<  approximation_set_size << "\n";
  }

// 0 = AMaLGaM-full, 1 = AMaLGaM-univariate. For more details, see cluster.cpp
  int local_optimizer_index  = 0; 
  
  // int MO_popsize = (int) (number_of_reference_points * popsize);
    std::stringstream ss;
     
    // mamalgam with niching is MOHillVallEA
    if(enable_niching == 0) {
      std::cout << "Optimizer settings: \n\tOptimizer = MAMaLGaM\n\t";
      ss << "_MAMaLGaM" << file_appendix;   
    } else {
      std::cout << "Optimizer settings: \n\tOptimizer = MO-HillVallEA\n\t";
      ss << "_moHillVallEA" << file_appendix;
    }
       
    file_appendix = ss.str();
    
    print_generational_statistics = write_generational_statistics && print_verbose_overview;
    std::cout << "local_optimizer_index = " << local_optimizer_index <<"\n\tPopulation size = " << popsize << "\n\tenable_niching = " << (enable_niching ? "yes" : "no") << "\n\tRBF optimization(WBO) = " << (optimize_whitebox ? "yes" : "no") << "\n\tRBF variance multiplier = " << variance_RBF_multiplier <<"\n\trandom_seed = " << random_seed << "\n";
    std::cout << "Problem settings:\n\tfunction_name = " << mo_fitness_function->name() << "\n\tproblem_index = " << problem_index << "\n\tmo_number_of_parameters = " << mo_fitness_function->number_of_parameters << "\n\tinit_range = [" << lower_init << ", " << upper_init << "]\n\tNumber of radials = " <<mo_fitness_function->number_of_radial<<"\n\tHV reference point = " << mo_fitness_function->hypervolume_max_f0 << ", " << mo_fitness_function->hypervolume_max_f1 << "\n";
    std::cout << "Run settings:\n\tmax_number_of_MO_evaluations = " << maximum_number_of_mo_evaluations <<"\n\tmaximum_number_of_seconds = " << maximum_number_of_seconds << "\n\tuse_vtr = " << use_vtr << "\n\tvtr = " << value_to_reach << "\n";
    std::cout << "Archive settings:\n\tElitist_archive_target_size = " << elitist_archive_size_target << "\n\tApproximation_set_size = " <<  approximation_set_size << "\n";
                    
    hicam::recursion_scheme_t opt(
      mo_fitness_function,  
      mo_lower_init_ranges, 
      mo_upper_init_ranges,
      value_to_reach,
      use_vtr,
      enable_niching, // 0 = MAMaLGaM, else, MO-HillVallEA
      local_optimizer_index, // 0 = AMaLGaM-full, 1 = AMaLGaM-univariate. For more details, see cluster.cpp
      HL_tol,
      elitist_archive_size_target,
      approximation_set_size,
      maximum_number_of_populations, 
      (int) popsize,
      number_of_subgenerations_per_population_factor,
      maximum_number_of_mo_evaluations,
      maximum_number_of_seconds,
      random_seed,
      write_generational_solutions,
      write_generational_statistics,
      print_generational_statistics,
      write_directory,
      file_appendix,
      print_verbose_overview,
      optimize_whitebox,
      variance_RBF_multiplier
      );
    opt.run();

 
  
  if(print_verbose_overview)
  {
    std::cout << "Best: \n\tHV = " << std::fixed << std::setprecision(14) << opt.approximation_set->compute2DHyperVolume(mo_fitness_function->hypervolume_max_f0, mo_fitness_function->hypervolume_max_f1) << "\n\tMO-fevals = " << opt.number_of_evaluations  << "\n\truntime = " << opt.run_time << " sec" << std::endl;
    
        
    std::cout << "pareto_front" << " = [";
    for(size_t k = 0; k < opt.approximation_set->sols.size(); ++k) {
      std::cout << "\n\t" << std::fixed << std::setw(10) << std::setprecision(4) << opt.approximation_set->sols[k]->obj;
    }
    std::cout << " ];\n";
    
       
    std::cout << "pareto_set" << " = [";
    for(size_t k = 0; k < opt.approximation_set->sols.size(); ++k)
    {
      std::cout << "\n\t";
      for(size_t i = 0; i < opt.approximation_set->sols[k]->param.size(); ++i) {
        std::cout << std::fixed << std::setw(10) << std::setprecision(4) << opt.approximation_set->sols[k]->param[i] << " ";
      }
    }
    std::cout << " ];\n";
  }
  
}
