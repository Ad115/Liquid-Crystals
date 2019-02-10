/*  
    configuration: DEFINITIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include "string_utils.h"
# include "args_parse.h"
# include "configuration.h"



Config parse_user_configuration(int argc, char *argv[], Config defaults) {
    /* Here we parse the command line arguments;  If
       you add an option, document it in the usage() function! */
    Config config = defaults;

    Args *args = parse_arguments(argc, argv);
    char *val;
    if (val=find_argument(args,"N|n_particles")) 
        config.n_particles = atoi(val);

    if (val=find_argument(args,"sf|sampling|sampling_frequency")) 
        config.sampling_frequency = atoi(val);

    if (val=find_argument(args,"rc|cutoff|cutoff_radius")) 
        config.cutoff_radius = atof(val);

    if (val=find_argument(args,"dt|step|time_step")) 
        config.time_step = atof(val);

    if (val=find_argument(args,"n|rho|numeric_density")) 
        config.numeric_density = atof(val);

    if (val=find_argument(args,"T|T0|initial_temperature")) 
        config.initial_temperature = atof(val);

    if (val=find_argument(args,"seed|random_seed")) 
        config.random_seed = (unsigned long)atoi(val);

    if (find_argument(args,"h|help")) {
        print_usage(argv); exit(0);
    }
    delete_arguments(args);
    return config;
}



void print_usage (char *argv[]) {
    printf(
        "%s usage:\n"
        "%s [options]\n\n"
        "Options:\n"
        "     -N [integer]      Number of particles\n"
        "     -rho [real]       Number density\n"
        "     -dt [real]        Time step\n"
        "     -rc [real]        Cutoff radius\n"
        "     -T0 [real]        Initial temperature\n"
        "     -fs [integer]     Sample frequency\n"
        "     -sf [a|w]         Append or write config output file\n"
        "     -seed [integer]   Random number generator seed\n"
        "     -h                Print this info\n",
      argv[0], argv[0]);
}