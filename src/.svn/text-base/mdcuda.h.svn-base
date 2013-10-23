/* -*- mode: C++; tab-width: 2; indent-tabs-mode: t; c-basic-offset: 2 -*- */
// vim:sts=2:sw=2:ts=2:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
/******************************************************************************/
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
/******************************************************************************/

#ifndef _MDCUDA_H_
#define _MDCUDA_H_

#include "globals.h"
#include "cucomplexops.h"
#include "timer.h"

/*----------------------------------------------------------------------------*/
__global__ void HalfKickGPU(float4* v4d,
                            float4* f4d,
                            float4* f4d_nonbond,
                            float4* f4d_bonded
#ifdef PCONSTANT
                          , float4* boxVelocd,
                            float4* boxAcceld
#endif
                                             );

//------------------------------------------------------------------------------
__device__ float4 compute_nearest_image(float4 r0, float4 r1);

//------------------------------------------------------------------------------
__global__ void ConjugatedGradient(float4* r4d,
                                   float4* roldd,
                                   int* nupdated,
                                   float4* v4d,
                                   float4* f4d,
                                   float4* f4d_nonbond,
                                   float4* f4d_bonded,
                                   float4* prev_f4d,
                                   float3* Htd,
                                   float* cgfacd
#ifdef PME_CALC
                                 , float4* r4d_scaled,
                                   float4* prev_r4d_scaled,
                                   int4* disp
#endif
                                             );

//------------------------------------------------------------------------------
__global__ void SteepestDescent(float4* r4d,
                                float4* roldd,
                                int* nupdated,
                                float4* v4d,
                                float4* f4d,
                                float4* f4d_nonbond,
                                float4* f4d_bonded,
                                float* sdfacd
#ifdef PME_CALC
                              , float4* r4d_scaled,
                                float4* prev_r4d_scaled,
                                int4* disp
#endif
                                          );

//------------------------------------------------------------------------------
__global__ void UpdateCoords(float4* r4d,
                             float4* v4d
#ifdef PCONSTANT
                           , float4* viriald
#endif
                                            );

//------------------------------------------------------------------------------
__global__ void restraint_force(float3* com0d,
                                float3* com1d,
                                char* segidd,
                                float4* f4d,
                                float4* r4d);

#ifdef PME_CALC
//------------------------------------------------------------------------------
__device__ float4 ewaldCorrection(float4 r, float& eEwexcl);
#endif //PME_CALC

//------------------------------------------------------------------------------
void ComputeAccelGPU();

//==============================================================================
#include "bonded_forces.h"
#include "nonbonded_forces.h"
#include "cell_based_kernels.h"
#include "pme_kernels.h"
//==============================================================================

#include "mdcuda.cu"

#endif //_MDCUDA_H_
