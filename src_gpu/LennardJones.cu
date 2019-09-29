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

            double r2 = (dr*dr);
            double r6 = 1./(r2 * r2 * r2);

            return ( 48*(r6*r6 - 0.5*r6)/r2 )* dr;
        }
};