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

#ifndef _DYNAMICS_H_
#define _DYNAMICS_H_

#include "globals.h"
#include "shake.h"

//----------------------------------------------------------------------------
void CallCellBuild();

//----------------------------------------------------------------------------
void Check_Static_Parameters(char success);

//----------------------------------------------------------------------------
void BuildNBGPU();

//////////////////////////////////////////
#ifdef PME_CALC
//////////////////////////////////////////
//==============================================================================
void CallLatticeBuild();
//////////////////////////////////////////
#endif
//////////////////////////////////////////

#ifdef PCONSTANT
//----------------------------------------------------------------------------
void ScanVirial();
#endif

//----------------------------------------------------------------------------
void Scan_Energy();

//----------------------------------------------------------------------------
void MinimizeStep();

//------------------------------------------------------------------------------
void SingleStep();

//------------------------------------------------------------------------------
void write_trj_header(FILE* xyzfile, int stepCount);

/*----------------------------------------------------------------------------*/
void checkCUDAError(const char* msg);


/*----------------------------------------------------------------------------*/
void validate_dihedral();

//----------------------------------------------------------------------------
void validate_angle();

#include "dynamics.cu"

#endif //_DYNAMICS_H_
