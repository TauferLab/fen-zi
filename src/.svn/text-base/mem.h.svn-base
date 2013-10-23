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
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/******************************************************************************/

#ifndef _MEM_H_
#define _MEM_H_

#include "globals.h"


void CudaAllocateCopy(void **globalVarPointer,
                      void *hostPointer,
                      size_t size,
                      const char* var_name);

void CudaAllocate(void **globalVarPointer,
                  size_t size,
                  const char* Variable_Name);

void InitMem();

#include "mem.cu"

#endif //_MEM_H_