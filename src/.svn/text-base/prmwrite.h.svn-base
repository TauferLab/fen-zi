/******************************************************************************/
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
/******************************************************************************/
#ifndef _PRMWRITE_H_
#define _PRMWRITE_H_

#include "globals.h"


//----------------------------------------------------------------------------
int accessfile(char filename[512]);

//----------------------------------------------------------------------------
void SaveCheckpointBinary(char restartfilename[512]);

//----------------------------------------------------------------------------
void LoadCheckpointBinary(char restartfilename[512]);

//----------------------------------------------------------------------------
void SaveCheckpointAscii(char restartfilename[512]);

//----------------------------------------------------------------------------
void LoadCheckpointAscii(char restartfilename[512]);

float mag_float3(float4 a, float4 b);

//----------------------------------------------------------------------------
void Write_xyz(char* tempfilename, char pbc);

//----------------------------------------------------------------------------
void write_forces(char* tempfilename, int countstep);

//----------------------------------------------------------------------------
void print_mass_charge_type();

#include "prmwrite.cu"

#endif //_PRMWRITE_H_
