#ifndef INTEGRATOR_INTERFACE_HEADER
#define INTEGRATOR_INTERFACE_HEADER

#include "TransformationInterface.h"


class Integrator: Transformation {
public:
    double time_step;

    template<typename SystemT>
    void operator()(SystemT& s); /*
    * Do the integration step
    */
};

#endif