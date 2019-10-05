#ifndef TRANSFORMATIONS_HEADER
#define TRANSFORMATIONS_HEADER

class Transformation {
public:

    template<typename SystemT>
    void operator()(SystemT& s); /*
    * Apply the given transformation.
    */
};

#endif