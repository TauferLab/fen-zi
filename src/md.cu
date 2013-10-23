/*****************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
//            Joseph E. Davis
//            Michela Taufer (taufer@udel.edu)
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/*****************************************************************************/
#include "globals.h"

#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <cufft.h>

#ifndef _WIN32
#include <ctype.h>
#endif

void genrandfloat2(float2 &e){
	int randval;
	do{
		randval = rand();
		e.x = ((float)randval) / RAND_MAX;
	}while(randval == 0);

	do{
		randval = rand();
		e.y = ((float)randval) / RAND_MAX;
	}while(randval == 0);
}

double Dmod(double a, double b){
	int n;
	n = (int)(a / b);
	return (a - b * n);
}

double RandR(double* seed){
	*seed = Dmod(*seed * DMUL, D2P31M);
	return (*seed / D2P31M);
}

void RandVec3(double* p, double* seed){
	double x;
	double y;
	double s = 2.0;

	while(s > 1.0){
		x = 2.0 * RandR(seed) - 1.0;
		y = 2.0 * RandR(seed) - 1.0;
		s = x * x + y * y;
	}

	p[2] = 1.0 - 2.0 * s;
	s = 2.0 * sqrt(1.0 - s);
	p[0] = s * x;
	p[1] = s * y;
}

void print_to_stdout(const char* format, ... ){
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
}

void print_to_file(const char* format, ... ){
	va_list args;
	va_start(args, format);
	vfprintf(outfile, format, args);
	va_end(args);
}
