/* 
## Clase `device_obj`

Una clase que emula el comportamiento de un *smart pointer* para objetos en el 
GPU. La contraparte del `thrust::device_vector` pero para un sólo objeto.

La clase abstrae la alocación y liberación de memoria además de las operaciones 
de copia entre Host y Device. Esta es una abstracción del Host, por lo que no 
se puede utilizar en un kernel. 
*/

template<typename T, typename ... Args>
__global__
void init_object_kernel_(T *device_pointer, Args ... args) {
         
  new (device_pointer) T(std::forward<Args>(args) ...);
}


template< typename T >
class device_obj {
  
    T *device_pointer;
    
    public:
    
     template<typename ... Args>
     device_obj(Args&& ... args) {
         
         cudaMallocManaged(&device_pointer, sizeof(T));      
         
         init_object_kernel_<<<1,1>>>(
             device_pointer, 
             std::forward<Args>(args) ...
         );
     }
    
    T *device_ptr() const {
        return device_pointer;
    }
    
    T *raw_ptr() const {
        return device_pointer;
    }
    
    T get() {
        cudaDeviceSynchronize();
        
        return *device_pointer;
    }
    
    T operator*() {
        return get();
    }
    
    ~device_obj() {
        
        cudaFree(device_pointer);
    }
};
