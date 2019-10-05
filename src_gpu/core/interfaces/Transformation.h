#ifndef TRANSFORMATION_INTERFACE_HEADER
#define TRANSFORMATION_INTERFACE_HEADER

class Transformation {
public:

    template<typename SystemT>
    void operator()(SystemT& s); /*
    * Apply the given transformation.
    */
};

#endif