/*  
    PART II: DEFINITIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "string_utils.h"



// Functions to handle c-style strings (null-terminated character buffers)
// .............................................................................

char *str_copy_until(char *buffer, char c) {/*
 * Utility function that copies the contents of `buffer` to another dynamically
 * allocated one. Stops copying before the first `c` or '\0'. Adjusts the space
 * allocated to the minimal necessary and adds a null terminator at the end.
 */
    // Create a new dynamical string with the contents of the buffer up to
    // before the first appearance of `c`
    String *str = String__copy_until(buffer, c);
    
    // Return the structure contents and free the unneeded memory
    return String__unwrap( str );
    
} // --- str_copy_until



char *str_copy_from(char *buffer) {/*
 * Utility function that copies the contents of `buffer` to another dynamically
 * allocated one. Adjusts the space allocated to the minimal necessary and adds 
 * a null terminator at the end.
 */
    // Create a new dynamical string with the contents of the buffer up to
    // before the first appearance of `c`
    String *str = String__from(buffer);
    
    // Return the structure contents and free the unneeded memory
    return String__unwrap( str );
    
} // --- str_copy_from



char *str_find(char *buffer, char c) {/*
    * Search for the first occurrece of the given character and return a pointer
    * to it. If the character is not found, return a pointer to the string 
    * terminator.
    */
    // Search for `c` or the string terminator
    int i=0;
    while (1) {
        // Get next character
        char next = buffer[i];
        
        if (next == '\0') {
            break; // Character not found
            
        } else if (next == c) {
            break; // Character found.
        }

        i++;
    }
    
    // Return a pointer to the character
    return (buffer + i);
} // --- str_find



char *str_seek_after(char *buffer, char c) {/*
 * Utility function that searches `c` in the buffer and returns a pointer to 
 * the next position. If the character is not in the string, return a pointer
 * to the string terminator.
 */
    char *here_is_c = str_find(buffer, c);

    int offset = 1;
    if (*here_is_c != c) {
        // If the character was not found, keep the pointer invariant
        offset = 0;
    }

    return (here_is_c + offset);
} // --- str_seek_after



char *str_skip_all(char *buffer, char c) {
/*
 * Utility function that searches contiguous occurrences of `c` in the buffer
 * and returns a pointer to the next position.
 */    
    // Search for a char different from `c` or the string terminator
    int i=0;
    while (1) {
        // Get next character
        char next = buffer[i];
        
        if ( (next != c) || (next == '\0') ) {
            // A different character was found.
            // Exit the loop
            break;
            
        }
        
        i++;
    }
    
    // Return a pointer to the character
    return (buffer+i);
    
} // --- str_skip_all




// Dynamic string structure and methods
// .............................................................................



#define DEFAULT_BUFFER_SIZE 100



struct String_struct {/* 
    * A structure intended for use as a dynamic string.
    */
    char *content;   // The string per se
    int capacity;// The capacity of the string
    int ocupancy;// The number of characters already in the string
};


String *String__new( void ) {/* 
    Create a new dynamically allocated empty String.
    */
    int str_capacity = DEFAULT_BUFFER_SIZE;
    
    // Create the structure
    String *str = malloc( sizeof(*str) );
    
    // Create the inner string
    str->content = malloc( str_capacity * sizeof(*(str->content)) );
    str->content[0] = '\0';
    
    // Initialize the structure
    str->capacity = str_capacity;
    str->ocupancy = 0;
    
    return str;

} // --- String__new


void String__delete( String *str ) {/* 
    * Free the space occupied by the string
    */
    free(str->content);
    free(str);
    return;

} // --- String__delete



String *String__copy_until(char *buffer, char c) {/* 
    * Create a new dynamically allocated String initialized from the contents of
    * the `buffer` before the first occurrence of `c` or '\0'.
    */
    // Create a new dynamical string
    String *str = String__new();
    
    // Copy char by char the contents of the buffer
    int i=0;
    while (1) {
        // Get next character
        char next = buffer[i];

        if ((next == c) || (next == '\0')) {
            // Character found, exit the loop
            break;

        } else {
            // Add the character to the string
            String__append_char(str, next);
            i++;
        }
    }

    // Return the created object
    return str;

} // --- String__copy_until



String *String__from(char *buffer) {/* 
    * Create a new dynamically allocated String initialized from the contents of
    * the argument.
    */
    // Read the buffer until the string terminator
    return String__copy_until(buffer, '\0');

} // --- String__from



void String__append_char(String *str, char c) {/* 
    * Adds the character `c` to the dynamic string `str`.
    * If there is no space,the string is reallocated with more space.
    */
    int chars_in_str = str->ocupancy;
    int str_capacity = str->capacity;

    // Check if reallocation is needed
    int needed_space = (chars_in_str + 1); // Don't count the '\0'
    
    if ( !(needed_space < str_capacity) ) {
        // A reallocation is needed
        int new_size = str_capacity + DEFAULT_BUFFER_SIZE/2;
        // Reallocate inner string
        str->content = realloc(str->content, new_size);
        str->capacity = new_size;
    }

    // Add the character and the terminator
    str->content[ chars_in_str ] = c;
    str->content[chars_in_str + 1] = '\0';
    str->ocupancy += 1;
        
    return;
    
} // --- String__append_char



char *String__unwrap( String *str ) {/* 
    * Deallocates the String but keeps the inner char buffer.
    * Returns it after adjusting the space it occupies in memory.
    */
    // Save the contents and get rid of the wrapper structure
    char *buffer = str->content;
    free(str);
    
    return buffer;
    
} // --- String__unwrap



void String__print( String *str ) {/*
    * Print the content of the dynamic string structure
    */
    printf("%s", str->content);
    return;
    
} // --- String__print


# undef DEFAULT_BUFFER_SIZE