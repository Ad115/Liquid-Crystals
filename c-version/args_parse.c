/*  
    args_parse: DEFINITIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "string_utils.h"
#include "args_parse.h"

#define DEFAULT_BUFFER_SIZE 10


char *parse_key(char *str) { /*
    * From the string `str` of the form "--key=value" returns "key".
    * The key is returned as a dinamically allocated string.
    */
    // Search the first non-dash (-) character
    str = str_skip_all(str, '-');
    
    // Save the characters until the equals (=) sign
    return str_copy_until(str, '=');
    
} // --- parse_key



char *parse_value(char *str) { /*
    * From the string `str` of the form "--key=value" obtains "value".
    * The string returned points to the original source.
    */
    // Search for the equals (=) sign
    str = str_seek_after(str, '=');
    
    // Save the value string
    return str;
    
} // --- parse_value



char *get_argument_value(char ***args, char *arg_name) { /* 
    * Searches if the argument is present in `args`.
    * Returns the value associated with the argument, or an empty string if 
    * there is no value. If the argument is not found, returns NULL.
    */
   for (int i=0; args[i] != NULL; i++) {

       // If the key was found...
       if( strcmp(args[i][0], arg_name)==0 ) {

           return (args[i][1]) ? args[i][1] : NULL; 
           // Return a reference to it's value if there is a value
       }
   }
   // The key was not found
   return NULL;
   
} // --- get_argument_value


bool argument_present(char ***args, char *arg_name) { /* 
 * Searches if the argument is present in `args`.
 * Returns a boolean representing prescence or abscence of the argument.
 */
   for (int i=0; args[i] != NULL; i++) {

       // If the key was found...
       if( strcmp(args[i][0], arg_name)==0 ) {

           return true; // Return it's value
       }
   }
   // The key was not found
   return false;
   
} // --- searchArg



char *find_argument(char ***args, char *pattern) { /*
 * Search if any of the arguments in `pattern` is in `args`.
 * Pattern is a pipe('|')-separated argument list "arg1|arg2|arg3...".
 * The value associated with the argument is returned (owned by `args`).
 */
    char *value;
    
    do {
        // Fetch the argument
        char *arg = str_copy_until(pattern, '|');
        // Move to the next argument
        pattern = str_seek_after(pattern, '|');
        
        if (*arg == '\0') {
            // The pattern was exhausted and no argument was read
            free(arg);
            // Return a NULL pointer
            value = NULL;
            break;
        }
        
        // Search the argument
        if ( argument_present(args, arg) ) {
            // The argument was found, fetch the value and break the loop
            value = get_argument_value(args, arg);
            free(arg);
            break;
            
        } 
        
        // The argument was not found, free the arg used
        free(arg);
            
    } while (1);
    
    return value;
}



 char ***parse_arguments(const int argc, char *argv[]) {/*
    * From the string list `argv` with `argc` entries, get a list of key-value 
    * pairs, one per argument.
    * The arguments are in the form `-[-]key`, `-[-]key=value` or `-[-]key value`.
    */

    // Returns a list `args` of pairs of strings with the format: 
    //args[i][0] = "key", args[i][1] = "value"(empty string if there is no value)
    // args[i] = NULL marks the end of the list.

    // Create space for the arguments list (a list of pairs of strings)
    int nargs = 0;
    char ***args = malloc( (nargs+1) * sizeof(*args) ); // Count the NULL at the end
    
    // Parse the arguments
    for (int i=0; i<argc && strcmp(argv[i],"--")!=0; i++) {
        // New argument
        // Allocate space for the key-value pair
        nargs++;
        args = realloc(args, (nargs+1) * sizeof(*args)); // Count the NULL at the end 
        args[nargs-1]= malloc(2 * sizeof(**args));
        
        // Parse first part of the argument
        args[nargs-1][0] = parse_key(argv[i]);
        
        // Check if the string contains an equals sign
        if ( strchr(argv[i], '=') ) { 
            // Assume argument in the format: "-[-]key=value"
            args[nargs-1][1] = parse_value(argv[i]);

        } else {
            // Assume argument in the format: "-[-]key value"
            // Check if there is a corresponding value
            if ( (i+1 < argc) && !(strchr(argv[i+1], '-')) ) {
                // Save the value and skim index to the next argument
                args[nargs-1][1] = str_copy_from(argv[i+1]);
                i++;
            } else {
                // The argument is a loner
                // Set the value as empty
                args[nargs-1][1] = str_copy_from("");
            }
        }
    }
    // Finally, add an empty entry marking the end of the list
    args[nargs] = NULL;

    return args;
    
} // --- parse_arguments


void delete_arguments(char ***args) {/*
    * Free the space allocated for the arguments structure.
    */
    for(int i=0; args[i] != NULL; i++) {
        // Free the contents of the key
        if (args[i][0]) { free(args[i][0]); }
        // Free the value
        //if (args[i][1]) { free(args[i][1]); }
        // Free the key-value pair space
        free(args[i]);
    }
    // Free the array
    free(args);
    
} // --- delete_arguments


void print_arguments(char ***args) { 

    printf("Arguments parsed:\n");
    
    if(args) {
        // Print the pairs
        for(int i=0; args[i] != NULL; i++) {
            char *key = args[i][0];
            char *value = args[i][1];
            
            // Print the contents of the key
            if (key) { 
                printf("%s", key);
            }
            printf("  :  ");
            
            // Print the value
            if (value && strcmp(value, "") != 0) { 
                printf("%s", value); 
            } else {
                printf("(No value)"); 
            }
            printf("\n");
        }
    } else {
        // There are no arguments
        printf("None");
    }
    printf("\n");
    
} // --- print_arguments


#undef DEFAULT_BUFFER_SIZE
