#pragma once

#include "pcuditas/gpu/gpu_object.cu"

class EmptySpace {
public:

    template <class VectorT>
    __host__ __device__
    VectorT apply_boundary_conditions(const VectorT& position) const {
        return position;
    }

    template <class VectorT>
    __host__ __device__
    VectorT distance_vector(const VectorT& p1, const VectorT& p2) const {
        return (p2 - p1);
    }
};