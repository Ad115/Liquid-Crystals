#ifndef MD_H
#define MD_H

#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

class particle{
public:
	double pos[3]; //posicion
	double vel[3]; //velocidad
	double rad;    //radio
	void init(int ,int, double , gsl_rng* , double , double*); // inicializador
};


#endif
