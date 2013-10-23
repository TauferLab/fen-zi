/******************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/******************************************************************************/

#ifndef _CELL_BASED_KERNELS_H_
#define _CELL_BASED_KERNELS_H_

#include "globals.h"
#include "cucomplexops.h"

//------------------------------------------------------------------------------
__global__ void CheckCellOccupancy(float4* rd, int* num_nonbond);

//------------------------------------------------------------------------------
__global__ void CheckNonbondNum(float4* rd,
                                unsigned int* nblistd,
                                unsigned int* cell_nonbond,
                                int* num_nonbond
#ifdef SEARCH_EXCLUSION_LIST
                              , int* excllistd
#endif
                                              );


//------------------------------------------------------------------------------
__global__ void CellBuild(float4* rd,
                          float4* roldd,
                          unsigned int* cell_nonbond,
                          int* num_nonbond,
                          unsigned int* nblistd
#ifdef NUM_DEBUG_CELL
                        , float* debug_celld
#endif
                                            );

//------------------------------------------------------------------------------
//after updating the migrant atoms from the new cells,
//remove them from the current cell list...
__global__ void CellClean(float4* rd,
                          unsigned int* cell_nonbond,
                          int* num_nonbond);

//------------------------------------------------------------------------------
__global__ void CellUpdate(float4* rd,
                           float4* roldd,
                           unsigned int* cell_nonbond,
                           int* num_nonbond
#ifdef PCONSTANT
                         , float4* prev_boxLengthd
#endif
                                                  );

#ifdef PCONSTANT
//------------------------------------------------------------------------------
__global__ void reduce_virial(float4* v4d,
                              float4* boxVelocd,
                              float4* boxAcceld,
                              float4* kineticd,
                              float4* viriald
#ifdef USE_NPT
                            , float4* boxLengthd,
                              float4* propertiesd
#endif
                                                 );
#endif

//------------------------------------------------------------------------------
__global__ void reduce_PE(float4* f4d, float4* f4d_nonbond, float4* f4d_bonded);

/////////////////////Reduce center of mass for restraints/////////////////
//------------------------------------------------------------------------------
__global__ void reduce_COM(char* segidd,
                           float3* com0d,
                           float3* com1d,
                           float* mass_segid0d,
                           float* mass_segid1d);
/////////////////////End Reduce center of mass for restraints/////////////////

//------------------------------------------------------------------------------
__global__ void reduce_PE_partial_sum(float4* f4d);

//------------------------------------------------------------------------------
__global__ void reduce_PE_final_sum(float4* f4d);

//------------------------------------------------------------------------------
__device__ char Search_excl_binary(int* excl,
                                   int x,
                                   int y,
                                   char start,
                                   char end);

/*
//----------------------------------------------------------------------------
__device__ char Search_exclGPU_shmem(int *excl, int y,
 unsigned char first, unsigned char last) {
//----------------------------------------------------------------------------
//Searches exclusion list on GPU for a given atom x and return the position of
//the first occurrence of a given value y
//----------------------------------------------------------------------------
	char j;

	//already checked for first and last atoms in the exclusion list in nbbuild_*
	for(j=first; j<=last; j++)
	  if(excl[MAX_NBCELL*j + threadIdx.x] == y) return 1;

	return 0;
}

//----------------------------------------------------------------------------
__device__ char exclGPU(int *excl, int x, int y, unsigned char first,
 unsigned char last) {
//----------------------------------------------------------------------------
//Searches exclusion list on GPU for a given atom x and return the position of
//the first occurrence of a given value y
//----------------------------------------------------------------------------
	char j;

	//already checked for first and last atoms in the exclusion list in nbbuild_*
	for(j=first; j<=last; j++)
	  if(excl[WorkgroupSized*j + x] == y) return 1;

	return 0;
}

*/

//------------------------------------------------------------------------------
__global__ void nbbuild_exclbitvec(float4* rd,
                                   float4* roldd,
                                   unsigned int* nblistd,
                                   unsigned int* cell_nonbond,
                                   int* num_nonbond,
                                   unsigned long long* excl_bitvecd,
                                   char* excl_bitvec_offsetd,
                                   int* excllistd
#ifdef PCONSTANT
                                 , float4* prev_boxLengthd
#endif
#ifdef DEBUG_NBLIST
                                 , float* debug_nblistd
#endif
                                                       );

//------------------------------------------------------------------------------
__global__ void nbbuild_excllist(float4* rd,
                                 float4* roldd,
                                 unsigned int* nblistd,
                                 unsigned int* cell_nonbond,
                                 int* num_nonbond,
                                 unsigned long long* excl_bitvecd,
                                 char* excl_bitvec_offsetd,
                                 int* excllistd
#ifdef PCONSTANT
                               , float4* prev_boxLengthd
#endif
#ifdef DEBUG_NBLIST
                               , float* debug_nblistd
#endif
                                                     );

//------------------------------------------------------------------------------

#ifdef PME_CALC
//------------------------------------------------------------------------------
__global__ void CheckLatticeNum(float4* r4d, int* numL);

//------------------------------------------------------------------------------
//////////////////////////updating the cell list/////////////////////////////
//Iterates through the global list of atoms for each cell..
//Called only at the start of the simulation to build the list.
__global__ void LatticeBuild(float4* r4d,
                             float4* r4d_scaled,
                             int* cellL,
                             int* numL
#ifdef DEBUG_PME
                           , float* pme_debug_d
#endif
                                               );

//------------------------------------------------------------------------------
__global__ void LatticeUpdate(float4* r4d_scaled,
                              int4* disp,
                              int* cellL,
                              int* numL
#ifdef DEBUG_PME
                            , float* pme_debug_d
#endif
                                                );

//------------------------------------------------------------------------------
__global__ void BCMultiply(cufftComplex* Qd
#ifdef DEBUG_PME
                         , float* pme_debug_d
#endif
                                             );
#endif //#ifdef PME_CALC

//------------------------------------------------------------------------------
//initializes B and C for PME and VanderWaals and Electostatic force tables...
void InitDeviceConstants();

#include "cell_based_kernels.cu"

#endif //_CELL_BASED_KERNELS_H_
