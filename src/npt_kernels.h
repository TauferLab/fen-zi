/*****************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
//            Michela Taufer (taufer@udel.edu)
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/*****************************************************************************/


#ifndef _NPT_KERNELS_H_
#define _NPT_KERNELS_H_

#include "globals.h"
#include "cucomplexops.h"
#include "shake.h"

//------------------------------------------------------------------------------
void SingleStep_npt();

#include "npt_kernels.cu"

#endif //_NPT_KERNELS_H_
