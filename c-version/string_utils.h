# ifndef STRING_UTILS_H
# define STRING_UTILS_H

/*
Dynamic string utilities
------------------------

Functions and structures for almost pain-free handling of dynamic char strings.
*/


/*  
    DECLARATIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


// Functions to handle c-style strings (null-terminated character buffers)

char *str_copy_from(char *buffer); /*
    * Utility function that copies the contents of `buffer` to another dynamically
    * allocated one. Adjusts the space allocated to the minimum needed and adds 
    * a null terminator at the end.
    */

char *str_copy_until(char *buffer, char c); /*
    * Similar to `str_from` copies the input string to a new buffer but 
    * stops copying before the first 'c' or '\0'. Adjusts the space allocated to
    * the minimum needed and adds a null terminator at the end.
    */

char *str_find(char *buffer, char c); /*
    * Search for the first occurrece of the given character and return a pointer
    * to it. If the character is not found, return a pointer to the string 
    * terminator.
    */

char *str_seek_after(char *buffer, char c); /*
    * Search for the given character in the buffer and returns a pointer to the 
    * next position. If the character is not in the string, return a pointer to 
    * the string terminator.
    */

char *str_skip_all(char *buffer, char c); /*
    * Search for the given character in the buffer and returns a pointer to the 
    * next position after the last contiguous occurrence of it. If the character 
    * is not in the string, return a pointer to the string terminator.
    */

/* NOTE:
Functions that search for a substring return a pointer to the string terminator 
so that the functions are composable and that calling such function on an 
already exhausted string (or an empty string) can be done safely.
*/


// Dynamic string structure and methods


    typedef struct String_struct String;


String *String__new( void ) ; /* 
    Create a new dynamically allocated empty String.
    */
    
void String__delete( String *str ); /* 
    * Free the space occupied by the string
    */

String *String__copy_until(char *buffer, char c); /* 
    * Create a new dynamically allocated String initialized from the contents of
    * the `buffer` before the first occurrence of `c` or '\0'.
    */

String *String__from(char *buffer); /* 
    * Create a new dynamically allocated String initialized from the contents of
    * the argument.
    */

void String__append_char(String *str, char c); /* 
    * Adds the character `c` to the dynamic string `str`.
    * If there is no space,the string is reallocated with more space.
    */

char *String__unwrap( String *str ); /* 
    * Deallocates the String but keeps the inner char buffer.
    * Returns it after adjusting the space it occupies in memory.
    */

void String__print( String *str ); /*
    * Print the content of the dynamic string structure
    */


# endif