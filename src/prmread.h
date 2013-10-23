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
//            Michela Taufer (taufer@udel.edu)
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/******************************************************************************/

#ifndef _PRMREAD_H_
#define _PRMREAD_H_

#include "globals.h"

//----------------------------------------------------------------------------
//function to perform case insensitive string comparison...
int cicompare(char* input, char* compstr);

//----------------------------------------------------------------------------
int addtype(char* curr_type);

//----------------------------------------------------------------------------
int typetonum_present(char* curr_type);

//----------------------------------------------------------------------------
void InitHostMem();

//----------------------------------------------------------------------------
void ReadXYZ();

//----------------------------------------------------------------------------
void ReadCrd();

//----------------------------------------------------------------------------
void ReadPdb();

//------------------------------------------------------------------------------
void ReadInput(char* input_filename);

//------------------------------------------------------------------------------
int present_in_dihedrals_set(int w, int x, int y, int z);

//------------------------------------------------------------------------------
int add_retrieve_dihedrals_set_type(int w,
                                    int x,
                                    int y,
                                    int z,
                                    char add_dihedral);

#ifdef IMPROPER
//------------------------------------------------------------------------------
int add_retrieve_impropers_set_type(int w,
                                    int x,
                                    int y,
                                    int z,
                                    char add_improper);
#endif
//------------------------------------------------------------------------------
int add_retrieve_angles_set_type(int x, int y, int z, char add_angle);

#ifdef UREY_BRADLEY
//----------------------------------------------------------------------------
int add_retrieve_ureyb_set_type(int x, int y, int z, char add_ureyb);
#endif

//------------------------------------------------------------------------------
int add_retrieve_bonds_set_type(int x, int y, char add_bond);

//------------------------------------------------------------------------------
int add_retrieve_nbfix_set_type(int x, int y, char add_nbfix);

//----------------------------------------------------------------------------
void add_to_constraints(int x, int y);

//----------------------------------------------------------------------------
int get_localid(int* local_atoms, int atomid);

//----------------------------------------------------------------------------
void add_atom_to_cluster(int clusterid, int key, int total_clusters);

//----------------------------------------------------------------------------
void build_constraints();

/*----------------------------------------------------------------------------*/
void ReadPsf_CHARMM();

//-----------------------------------------------------------------------------------------------
int add_specific_dihedrals_set_type_to_prm(int w,
                                           int x,
                                           int y,
                                           int z,
                                           float a,
                                           int n,
                                           float c);

//------------------------------------------------------------------------------
int add_wildcard_dihedrals_set_type_to_prm(int x,
                                           int y,
                                           float a,
                                           int n,
                                           float c);

#ifdef IMPROPER
//------------------------------------------------------------------------------
int add_specific_impropers_set_type_to_prm(int w,
                                           int x,
                                           int y,
                                           int z,
                                           float a,
                                           float b,
                                           float c);

//------------------------------------------------------------------------------
int add_wildcard_impropers_set_type_to_prm(int w,
                                           int z,
                                           float a,
                                           float b,
                                           float c);
#endif

/*----------------------------------------------------------------------------*/
void ReadPrm_CHARMM();

//----------------------------------------------------------------------------
char Search_excl_binary_host(int* excl, int x, int y, char start, char end);

//----------------------------------------------------------------------------
int SearchExcl(int x, int y);

//----------------------------------------------------------------------------
void SearchInsertExcl(int x, int y);

//----------------------------------------------------------------------------
void checkexcllistorder();

//----------------------------------------------------------------------------
int Generate_ExclBitVector();

//----------------------------------------------------------------------------
void GenExcl_minimal_new();

#include "prmread.cu"

#endif //_PRMREAD_H_
