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

#ifndef _BONDED_FORCES_H_
#define _BONDED_FORCES_H_

#include "globals.h"
#include "cucomplexops.h"

/*------------------------------------------------------------------------------
Updates acceleration of one atom from a single bonded pair interaction (on GPU),
4th coord of acceleration used for potential energy
------------------------------------------------------------------------------*/
__device__ float4 bondInteraction(float4 r, float2 bprm, float& eBond);

//------------------------------------------------------------------------------
//Updates acceleration of one atom from a single angle interaction (on GPU),
//4th coord of acceleration used for potential energy
//------------------------------------------------------------------------------
__device__ float4 angleInteraction(float3 v1,
                                   float3 v2,
                                   unsigned char id,
                                   float2 aprm,
                                   float& eAngle
#ifdef PCONSTANT
                                 , float3& VirSum
#endif
                                                 );


#ifdef IMPROPER

//------------------------------------------------------------------------------
__device__ float4 improperInteraction(float3 v1,
                                      float3 v2,
                                      float3 v3,
                                      char id,
                                      float2 prm,
                                      float& eImprop
#ifdef PCONSTANT
                                    , float3& VirSum
#endif
                                                    );

#endif

//------------------------------------------------------------------------------
__device__ float4 torsionInteraction(float3 v1,
                                     float3 v2,
                                     float3 v3,
                                     char id,
                                     dihedralParameter* dihedprm,
                                     char dcount,
                                     char h,
                                     float& eDihed
#ifdef PCONSTANT
                                   , float3& VirSum
#endif
                                                   );

//------------------------------------------------------------------------------
__device__ float4 torsionInteraction_c36(float3 v1,
                                         float3 v2,
                                         float3 v3,
                                         char id,
                                         dihedralParameter* dihedprm,
                                         char dcount,
                                         char h,
                                         float& eDihed
#ifdef PCONSTANT
                                       , float3& VirSum
#endif
                                                        );



//---------------------------------------------------------------------------------------------------------------------------------
void __global__ bondedforce(float4* f4d_bonded,
                            //bond Data...
                            int* bonds_indexd,
                            float2* bondprmd,
                            float* ebndd,
                            //angle Data...
                            int* angles_indexd,
                            int* angles_setd,
                            float2* angleprmd,
                            float* eangd
#ifdef UREY_BRADLEY
                          , int* ureyb_indexd,
                            float2* ureybprmd,
                            float* eureybd
#endif

                            //dihedral Data...
                          , int* dihedrals_indexd,
                            int* dihedrals_setd,
                            dihedralParameter* dihedral_prmd,
                            unsigned char* dihedral_type_countd,
                            float* edihedd,
                            float* evdwd,
                            float* eelecd

#ifdef IMPROPER
                          , int* impropers_indexd,
                            int* impropers_setd,
                            float2* improper_prmd,
                            float* eimpropd
#endif

                          , int* ewlistd,
                            float* eEwexcld

#ifdef PCONSTANT
                          , float4* viriald
#endif
#ifdef BOND_DEBUG
                          , float* bonds_debugd
#endif
#ifdef ANGLE_DEBUG
                          , float* angles_debugd
#endif
#ifdef DIHED_DEBUG
                          , float* dihedrals_debugd
#endif
                                                   );

//---------------------------------------------------------------------------------------------------------------------------------
void __global__ bondedforce_c36(float4* f4d_bonded,
                            //bond Data...
                            int* bonds_indexd,
                            float2* bondprmd,
                            float* ebndd,
                            //angle Data...
                            int* angles_indexd,
                            int* angles_setd,
                            float2* angleprmd,
                            float* eangd
#ifdef UREY_BRADLEY
                          , int* ureyb_indexd,
                            float2* ureybprmd,
                            float* eureybd
#endif

                            //dihedral Data...
                          , int* dihedrals_indexd,
                            int* dihedrals_setd,
                            dihedralParameter* dihedral_prmd,
                            unsigned char* dihedral_type_countd,
                            float* edihedd,
                            float* evdwd,
                            float* eelecd

#ifdef IMPROPER
                          , int* impropers_indexd,
                            int* impropers_setd,
                            float2* improper_prmd,
                            float* eimpropd
#endif

                          , int* ewlistd,
                            float* eEwexcld

#ifdef PCONSTANT
                          , float4* viriald
#endif
#ifdef BOND_DEBUG
                          , float* bonds_debugd
#endif
#ifdef ANGLE_DEBUG
                          , float* angles_debugd
#endif
#ifdef DIHED_DEBUG
                          , float* dihedrals_debugd
#endif
                                                   );

#include "bonded_forces.cu"

#endif //_BONDED_FORCES_H_
