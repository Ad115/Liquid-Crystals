#ifndef CONFIGURATION_HEADER
#define CONFIGURATION_HEADER

/*  
    DECLARATIONS (INTERFACE)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

typedef struct Config Config;

struct Config { /*
    * Parameters for a program run.
    */
    int n_particles;
    int sampling_frequency;
    double cutoff_radius;
    double time_step;
    double numeric_density;

    double initial_temperature;
    unsigned long int random_seed;
};


Config parse_user_configuration(int argc, char *argv[], Config defaults); /* 
    * Here we parse the command line arguments;  If
    * you add an option, document it in the print_usage() function! 
    */

void print_usage (char *argv[]); /*
    * Print the help of the program's command line interface.
    */


#endif