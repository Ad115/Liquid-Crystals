template<typename T>
__global__
void init_array_kernel(T *gpu_array, size_t n) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n; 
        i += blockDim.x * gridDim.x) 
    {
        new (&gpu_array[i]) T();
    }
}

template<class T, class TransformedT, class TransformationT>
__global__
void transform_kernel(
        T *from_array, size_t n, 
        TransformedT *to_array,
        TransformationT transform) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n; 
        i += blockDim.x * gridDim.x) 
    {
        to_array[i] = transform(from_array[i], i);
    }
}

template<typename T, typename Transformation>
__global__
void for_each_kernel(T *gpu_array, size_t n, Transformation fn) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n; 
        i += blockDim.x * gridDim.x) 
    {
        fn(gpu_array[i], i);
    }
}

template<typename T, typename Reduction>
__global__
void reduce_2step_kernel(
        T *gpu_array, size_t n, 
        T *out, 
        Reduction fn, T initial_value=T{}) { /*
    Log-reduction based from the one in the book "The CUDA Handbook" by 
    Nicholas Wilt.
    */

    extern __shared__ T partials[];

    const int tid = threadIdx.x;

    auto reduced = initial_value;
    for (int i = blockIdx.x * blockDim.x + tid; 
         i < n; 
         i += blockDim.x * gridDim.x) {

        reduced = fn(reduced, gpu_array[i]);
    }
    partials[tid] = reduced;
    __syncthreads();


    for (int active_threads = blockDim.x / 2;
         active_threads > 0;
         active_threads /= 2) {
        
        auto is_active_thread = tid < active_threads;
        if (is_active_thread) {
            partials[tid] = fn(partials[tid], partials[tid + active_threads]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = partials[0];
    }
}

/* -----------------------------------------------------------------------

 These kernels are used and tested on the gpu_array and gpu_object classes.
*/

