/* -*- mode: C++; tab-width: 2; indent-tabs-mode: t; c-basic-offset: 2 -*- */
// vim:sts=2:sw=2:ts=2:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
/******************************************************************************/
//
// Md code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan
//            Joseph E. Davis
//            Michela Taufer
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/******************************************************************************/

#ifndef _NONBONDED_FORCES_H_
#define _NONBONDED_FORCES_H_

#include "globals.h"
#include "cucomplexops.h"

#define USE_NEW_NONBOND
#ifdef USE_NEW_NONBOND
// -----------------------------------------------------------------------------
// New version of nonbondforces from mt - Use nbfix parameters if available
// Precompute extended eps and sigma - Save values in texture memory
//
//------------------------------------------------------------------------------
__global__ void nonbondforce(float4* f4d_nonbond
#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
                           , float4* f4d_nonbond0,
                             float4* f4d_nonbond1
#endif
#endif
                           , unsigned int* nblistd,
                             float* evdwd,
                             float* eelecd
#ifdef PCONSTANT
                           , float4* viriald
#endif
#ifdef DEBUG_NONB
                           , float* debug_nonbd
#endif
                                               );

#else

// -----------------------------------------------------------------------------
// Old nonbond forces of ng - this version computes eps and sigma at runtime
// Use of shared memory is activated with this definition
// #define USE_SHMEM

//------------------------------------------------------------------------------
__global__ void nonbondforce(float4* f4d_nonbond
#ifdef USE_CONSFIX
                           , float4* f4d_nonbond0,
                             float4* f4d_nonbond1
#endif
                           , unsigned int* nblistd,
                             float* evdwd,
                             float* eelecd
#ifdef PCONSTANT
                           , float4* viriald
#endif
#ifdef DEBUG_NONB
                           , float* debug_nonbd
#endif
                                               );

#endif

#include "nonbonded_forces.cu"

#endif //_NONBONDED_FORCES_H_
