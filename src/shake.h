/*****************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/*****************************************************************************/

/* -*- mode: C++; tab-width: 2; indent-tabs-mode: t; c-basic-offset: 2 -*- */
// vim:sts=2:sw=2:ts=2:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
#ifndef _SHAKE_H_
#define _SHAKE_H_

#include "globals.h"
#include "cucomplexops.h"

//Shake implementation in FENZI using constraint algorithm...
// solve_bond_constraints
// solve_velocity_constraints

//------------------------------------------------------------------------------
__global__ void solve_bond_constraints(int2* constraintsd,
                                       unsigned char* constraints_by_atomd,
                                       float2* constraintsprmd,
                                       int* atoms_in_clusterd,
                                       float4* r4shaked,
                                       float4* r4d,
                                       float4* v4d
#ifdef PCONSTANT
                                     , float4* viriald
#endif
                                                      );

//------------------------------------------------------------------------------
__global__ void solve_velocity_constraints(int2* constraintsd,
                                           unsigned char* constraints_by_atomd,
                                           float2* constraintsprmd,
                                           int* atoms_in_clusterd,
                                           float4* v4d //, float4 *r4d
#ifdef PCONSTANT
                                         , float4* viriald
#endif
                                                          );

#include "shake.cu"

#endif //_SHAKE_H_
