/* -*- mode: C++; tab-width: 2; indent-tabs-mode: t; c-basic-offset: 2 -*- */
// vim:sts=2:sw=2:ts=2:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
/*****************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
//            Joseph E. Davis
//            Michela Taufer 
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/*****************************************************************************/


/******************************************************************************
File md.h is an include file for program md.c.
******************************************************************************/
// #include "globals.h"

// #include <stdio.h>
// #include <stdarg.h>
// #include <math.h>
// #include <time.h>
// #include <string.h>
// #include <cuda.h>
// #include <cufft.h>

// #ifndef _WIN32
// #include <ctype.h>
// #endif

#ifndef _MD_H_
#define _MD_H_

/* Functions & function prototypes *******************************************/

void genrandfloat2(float2 &e);

double Dmod(double a, double b);

double RandR(double* seed);

void RandVec3(double* p, double* seed);

void checkCUDAError(const char*);

void (*LoadCoord)();
void (*LoadCheckpoint)(char *);
void (*SaveCheckpoint)(char *);

void (*NBBuild_Kernel)(float4*, float4*, unsigned int*, unsigned int*, int*,
                       unsigned long long*, char*, int*
#ifdef PCONSTANT
                       ,float4*
#endif
                      );

//------------------------------------------------------------------------
void print_to_stdout(const char* format, ... );

void print_to_file(const char* format, ... );

void (*printout)(const char* format, ...);

//------------------------------------------------------------------------

#include "md.cu"

#endif //_MD_H_
