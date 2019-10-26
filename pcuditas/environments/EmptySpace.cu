#pragma once

#include "pcuditas/gpu/gpu_object.cu"

class GPUEmptySpace {
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


class EmptySpace {

    gpu_object<GPUEmptySpace> _gpu_self;

public:

    GPUEmptySpace *gpu_pointer() {
        return _gpu_self.gpu_pointer();
    }
};