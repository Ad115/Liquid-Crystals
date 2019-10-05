#ifndef INTEGRATOR_HEADER
#define INTEGRATOR_HEADER

#include "Transformations.cu"


class Integrator: Transformation {
public:
    double time_step;

    template<typename SystemT>
    void operator()(SystemT& s); /*
    * Do the integration step
    */
};

#endif