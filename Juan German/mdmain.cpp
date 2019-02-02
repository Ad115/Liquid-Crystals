// se compila con g++ mdmain.cpp md.cpp -o mdmain -lgsl -lgslcblas 
#include "md.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;

int main(){
// variables 
int npart =9;
particle part[npart];
double L=10;
double* K;
gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);
unsigned long int Seed = 23410981;
gsl_rng_set(r,Seed);

// inicializar las particulas
for (int i=0;i<npart;i++){ part[i].init(npart,i,L,r,L,K);}



return 0;
}
