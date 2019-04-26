  
thrust::device_vector<Vector> applyFunctionVector( thrust::device_vector<Vector> vect )
{
    thrust::device_vector<Vector> dev_out( vect.size() );
    auto ff = [=]  __device__ (Vector v)
    {
        v.set_X(1);
        v.set_Y(2);
        v.set_Z(3);


        return v;
    };
    thrust::transform(vect.begin(), vect.end(), dev_out.begin(),ff);

    return dev_out;
}


thrust::device_vector<Vector> applyFunctionResta( thrust::device_vector<Vector> vect )
{
    thrust::device_vector<Vector> dev_out( vect.size() );
    
    auto ff = [=]  __device__ (Vector v)
    {
        Vector w(4,5,6);
        return 10.0*(w-v);
    };
    thrust::transform(vect.begin(), vect.end(), dev_out.begin(),ff);

    return dev_out;
}

void printVectors(thrust::device_vector<Vector> vecs_device){
    thrust::host_vector<Vector> host = vecs_device;
    for(int i=0; i<host.size(); ++i ){
        cout << host[i].X() << " " 
             << host[i].Y() << " " 
             << host[i].Z() << endl;
    }
    return;
}