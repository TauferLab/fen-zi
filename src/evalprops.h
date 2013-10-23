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
#ifndef _EVALPROPS_H_
#define _EVALPROPS_H_

#include "globals.h"

/*----------------------------------------------------------------------------*/
void EvalProps();

//------------------------------------------------------------------------------
void print_output_headers();
//------------------------------------------------------------------------------

#include "evalprops.cu"

#endif //_EVALPROPS_H_
