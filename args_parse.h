#ifndef ARGS_PARSE_HEADER
#define ARGS_PARSE_HEADER

/*
Argument parsing utilities
--------------------------

Functions for almost pain-free parsing of command line arguments.
Command line arguments are of the form `--key=value`, `-key value` 
or simply `-key`.

Usage example
.............

```
    #include "args_parse.h"

    int main(int argc, char *argv[]) {

        Args *args = parse_arguments(argc, argv);

        char *value = find_argument(args, "input|i");
        if (value) {
            printf("Input file specified: %s", value);
        }
    }
```
If this code is compiled and run as: `$ ./a.out --input=thisfile.txt`, the 
output will be:
```
    Input file specified: thisfile.txt
```

The same output is obtained for these various calls:
```
    $ /a.out --input thisfile.txt

    $ /a.out -i thisfile.txt

    $ /a.out --i thisfile.txt

    $ /a.out -i=thisfile.txt
```
*/



/*  
     DECLARATIONS (INTERFACE)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

typedef char** Args;

Args *parse_arguments(const int argc, char *argv[]); /*
    * From the string list `argv` with `argc` entries, get a list of key-value 
    * pairs, one per argument.
    * The arguments are in the form `-[-]key`, `-[-]key=value` or `-[-]key value`.
    */

char *find_argument(Args *args, char *pattern); /*
 * Search if any of the arguments in `pattern` is in `args`.
 * Pattern is a pipe('|')-separated argument list "arg1|arg2|arg3...".
 * The value associated with the argument is returned.
 */

void delete_arguments(Args *args); /*
    * Free the space allocated for the arguments structure.
    */
    
void print_arguments(Args *args);



#endif