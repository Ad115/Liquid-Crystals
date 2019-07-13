#include "src/Vector.cu"

/* Init the vector object in the device */
template <int Size, typename T>
__global__ void init_vector_seq(Vector<Size,T> *ptr) {
    for (int i=0; i<Size; i++) {
        (*ptr)[i] = i;
    }
}

template <int Size, typename T>
__global__ 
void operate_on_vectors_device(Vector<Size,T> *v1, Vector<Size,T> *v2) {
    auto sum = 2*(3+*v1) + (1+(*v2));
    print_vector(&sum);
}


int main( int argc, char **argv )
{
    // Test vector device
    Vector<10> *vector_ptr;
    cuda_alloc_vector(&vector_ptr);
    print_vector_kernel<<<1,1>>>(vector_ptr);
    cudaFree(vector_ptr);

    // Test vector device
    cuda_alloc_vector(&vector_ptr);
    init_vector_seq<<<1,1>>>(vector_ptr);
    print_vector_kernel<<<1,1>>>(vector_ptr);
    cudaFree(vector_ptr);

    // Test vector operations
    Vector<> *vector_1;
    cuda_alloc_vector(&vector_1);
    Vector<> *vector_2;
    cuda_alloc_vector(&vector_2);


    operate_on_vectors_device<<<1,1>>>(vector_1, vector_2);
    cudaFree(vector_1);
    cudaFree(vector_2);
}