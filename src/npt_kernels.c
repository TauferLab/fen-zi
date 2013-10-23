/* -*- mode: C++; tab-width: 2; indent-tabs-mode: t; c-basic-offset: 2 -*- */
// vim:sts=2:sw=2:ts=2:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s

#ifndef _NPT_KERNELS_H_
#define _NPT_KERNELS_H_

#include "globals.h"
#include "cucomplexops.h"

// sp values to initialize 
#define GKB 1
#define R0 1
#define PREF 1
#define TREF 1
#define Ms 1
#define Mxi 1

// sp values to initialize 
//------------------------------------------------------------------------------
void SingleStep_npt(){
//------------------------------------------------------------------------------
//      r & rv are propagated by DeltaT in time using the
//------------------------------------------------------------------------------       

	ScanVirial();

	return;
}

#endif //_NPT_KERNELS_H_
