// aqui van las funciones
#include "md.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;

// asigna pociciones en una red cubica y velocidades aleatorias
void particle::init (int npart,int partnum, double L, gsl_rng * r, double T0, double * KE){

  int n3=ceil(pow(npart,1./3));
  double d=L/n3;
  // asignar posiciones en red cubica
  pos[0]=(partnum%n3)*d; //numero del 0 al n3-1
  pos[1]=((partnum/n3)%n3)*d; 
  pos[2]=((partnum/(n3*n3))%n3)*d;

  // asignar velocidades en dist exp
  for (int i=0;i<3;i++){
  vel[i]=gsl_ran_exponential(r,1.0);
  }
 
}
