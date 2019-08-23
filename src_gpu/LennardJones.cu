template< typename VectorT=Vector<> >
class LennardJones: public Particle<VectorT>{

public:

        template< typename ContainerT >
        __host__ __device__
        VectorT force_law(LennardJones *other, ContainerT *box ){ /*
            * The force law: Lennard Jones
            * ============================
            * 
            *  f_ij = 48*e*[(s / r_ij)^12 - 1/2(s / r_ij)^6] * dr/r^2
            * 
            * See: http://www.pages.drexel.edu/~cfa22/msim/node26.html
            */
            VectorT dr = box->distance_vector((*this).position, other->position);
            double r2=1./(dr*dr);
            double rcut=3.5;

            double f;
            if (r2<rcut*rcut){
                double r6=r2*r2*r2;
                f=48*(r6*r6-0.5*r6);

            } else { f = 0; }

            return f/r2 * dr;
        }
};