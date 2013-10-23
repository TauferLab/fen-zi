/* -*- mode: C++; tab-width: 2; indent-tabs-mode: t; c-basic-offset: 2 -*- */
// vim:sts=2:sw=2:ts=2:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
#ifndef _GLOBALS_H_
#define _GLOBALS_H_

#include <stdio.h>
#include <cufft.h>
#include "defs.h"

//----------------    GLOBAL VARIABLES    ---    md.h     ---------------------

// consfix
// #define USE_CONSFIX
// #define DEBUG_NPT
#define EXPLICIT_EWALD_LIST

extern int printtostdout;

/* Input parameters (read from an input file in this order) ******************/

extern char FileName[512]; //File name to read coordinates from
extern char topFileName[512]; //Topology file containing the atom type information...
extern char psfFileName[512]; //File name to read psf (topology) from
extern char prmFileName[512]; //File name to read parameters from
extern char trajFileName[512]; //File name to write trajectories to
extern char filename_prefix[512];

//char restartfile_format[512] = "";
//char restartfile_prefix[512] = "", restartfile_suffix[512] = "";

extern char restartfilename[512];
extern int keepcheckpoint;
extern int c36;

extern float InitTemp; //Starting temperature
//float CurrentTemp; //Current temperature
extern float DeltaT; //Size of a time step
extern float Tau; //Coupling constant for Berendsen thermostat
extern int VAssign; //frequeny for reassigning velocities
extern int StartCount; //Starting step number - 0 is first run, n otherwise
extern int CheckpointTimestep;
extern int StepLimit; //Number of time steps to be simulated
extern int StepAvg; //Reporting interval for statistical data
extern int StepStore; //Reporting interval for checkpoing
extern int MinimizeStepLimit; //Number of minimization steps...
extern int BlockSize; //Thread block size = p, number of particles per tile for
                      //GPU force calculation
extern int TrjFreq; // Frequency of trajectory frames

extern float KAPPa; //KAPPa - it can be read in .in
extern float KAPPaSqr;
extern int INBFRQ; // INBFRQ - it can be read in .in^M

extern int PrintLevel; //determines how much output is printed
extern double seed; //seed for random number generator

extern float4 Region; //box lengths
extern float4 CellSize;
extern float4 RegionH; //Half the box lengths
extern float DeltaTH; //Half the time step
extern float CutCheck; //0.5*(CutMax-Cutoff), for checking nonbond list
extern int4 NumCells;

#ifdef PCONSTANT
//Maximum number of cells that can be expanded into due to volume fluctuations
extern int4 MaxNumCells;
#endif

//int TotalCells;

extern int nAtom; //Number of atoms
extern int nAtomwoseg; //Number of atoms

extern dim3 BondeddimGrid;
extern dim3 NBdimGrid;
extern dim3 BondeddimBlock;
extern dim3 NBdimBlock;

extern dim3 BCMuldimGrid;
extern dim3 BCMuldimBlock;

extern dim3 LatUpdimGrid;
extern dim3 LatUpdimBlock;

extern dim3 PMEForceBlock;
extern dim3 PMEForceGrid;

extern dim3 ChargeSpreadBlock;
extern dim3 ChargeSpreadGrid;

extern dim3 NBBuild_dimGrid; //(TOTAL_CELLS, 1);
extern dim3 NBBuild_dimBlock; //(MAX_NBCELL, 1);

extern dim3 CellClean_dimGrid;
extern dim3 CellClean_dimBlock;

extern dim3 DynadimGrid;
extern dim3 DynadimBlock;

//////////////////////////////////////////
#ifdef PME_CALC
//////////////////////////////////////////

extern int fftx;
extern int ffty;
extern int fftz;

extern float* ePME32;
extern double* ePME;
extern float* ePMEd;

extern float* eEwexcl32;
extern double* eEwexcl;
extern float* eEwexcld;

//int* ew_correction_list;
//int* ew_correction_listd;

//CUFFT for PME
extern cufftHandle plan;

extern float* M4d; //Spline M4...
extern float* dM4d; //derivative of M4...

extern int* cellL;
extern int* numL;

extern float4* r4d_scaled; // Pointer to GPU device memory for Scaled Coordinates
extern float4* prev_r4d_scaled;
extern int4* disp;//integer displacements in scaled coordinates
                  //used to update lattice list..

extern int Total_PME_Lattices;

//B and C for PME
//__constant__ __device__ cufftComplex b1d[FFTX], b2d[FFTY], b3d[FFTZ];

__constant__ __device__ float sqr_b1d[MAXFFT_X];
__constant__ __device__ float sqr_b2d[MAXFFT_Y];
__constant__ __device__ float sqr_b3d[MAXFFT_Z];

__constant__ __device__ float c1d[MAXFFT_X];
__constant__ __device__ float c2d[MAXFFT_Y];
__constant__ __device__ float c3d[MAXFFT_Z];

//__constant__ __device__ float erfc_table[NUM_ERFC_SAMPLES + 1];
//__constant__ __device__ float exp_table[NUM_ERFC_SAMPLES + 1];

//Q Matrix in the device..
extern cufftComplex* Qd;
//cufftComplex *Qd1;

#ifdef PCONSTANT
extern float4* fracd;
#else
__constant__ __device__ float4 frac;
#endif

__constant__ __device__ int FFTX;
__constant__ __device__ int FFTY;
__constant__ __device__ int FFTZ;

__constant__ __device__ float FFTXH;
__constant__ __device__ float FFTYH;
__constant__ __device__ float FFTZH;

__constant__ __device__ int TOTAL_PME_LATTICES;

//////////////////////////////////////////
#endif//#ifdef PME_CALC
//////////////////////////////////////////

// Host arrays for holding npt properties

// Device arrays for holding npt properties
extern float4* propertiesd;

// Host arrays for holding atomic contributions to bond, angle, etc. energies
extern float* ebnd;
extern float* eang;
extern float* evdw;
extern float* eelec;
extern float* edihed;
extern float4* virial;

// Device arrays for holding atomic contributions to bond, angle, etc. energies
extern float* ebndd;
extern float* eangd;
extern float* evdwd;
extern float* eelecd;
extern float* edihedd;

//////////////////RESTRAINTS////////////////////////////
extern char seg_name[NMAX][128]; //segment name of all the atoms in system...

#ifdef USE_CONSFIX
extern char consfix; //logical variable 0: no constrains, 1: constrains
extern int consfix_nseg; //number of segments given in input
extern char consfix_segname[128][16]; //segements name
extern int consfix_segcount[128]; //in [i] length of segment i
extern char* consfix_segidd;
#endif

extern char restraints; //status to indicate if restraints is set..
extern char segname0[128]; //segments to impose restraints on...
extern char segname1[128]; //restraints can act only between 2 segments...

extern char* segidd; //array to indicate segid of atoms...
extern char segid[NMAX];

extern float3* com0d;
extern float3* com1d;
extern float* mass_segid0d;
extern float* mass_segid1d;
extern float mass_segid0;
extern float mass_segid1;

__constant__ __device__ float3 consharmdistd;
__constant__ __device__ float consharmfcd;
extern float3 consharmdist;
extern float consharmfc;
////////////////////// END RESTRAINTS Variables...

//////////////////Shake Parameters and Variables/////////////
#define ATOMS_IN_CLUSTER  4
#define CONSTRAINTS_PER_ATOM  3
#define CLUSTER_SIZE  3
#define MAX_CLUSTERS  100000

extern int2* constraints;
extern int2* constraintsd;

extern unsigned char* constraints_by_atom;
extern unsigned char* constraints_by_atomd;

extern float2* constraintsprm;
extern float2* constraintsprmd;

extern int* atoms_in_cluster;
extern int* atoms_in_clusterd;

extern int num_constraints;

extern dim3 num_cluster_blocks;
extern dim3 cluster_blocksize;

extern float shaketol; //shake tolerance...
__constant__ __device__ float shaketold;

extern int nClusters;
__constant__ __device__ int nClustersd;
extern char shake;
extern float hfac;
//////////////////End Shake Parameters and Variables/////////////

#ifdef PCONSTANT
extern float4* kineticd;
extern float4* viriald;

#ifdef PME_CALC
__constant__ __device__ float eEwselfd;
#endif

#endif //PCONSTANT

#ifdef UREY_BRADLEY
extern float* eureyb;
extern float* eureybd;
#endif

#ifdef IMPROPER
extern float* eimprop;
extern float* eimpropd;
#endif

// Total Values...
extern float kinEnergy; //Kinetic energy
extern float potEnergy; //Potential energy
extern float eBond; //potential energy terms for bonds, angles, etc.
extern float eAngle;
extern float eLJ;
extern float eCoulomb;
extern double eEwself;
extern float totEnergy; //Total energy
extern float temperature; //Current temperature
extern int stepCount; //Current time step
extern int nupdate; //flag whether nonbond list needs updating
extern float vscale; //velocity scaling factor for thermostat

// intermediate values used for calculating shifted force vdw
extern float rc7;
extern float rc13;

//FILE* virfile;

// arrays for performing parallel sum (reduction)
//for velocity scaling (Berendsen thermostat)
//float* sum;
//float* sumd;
//int blocksizeT = 128; //separate block size for reduction

//-----------------Dihed Lists, Indices and Parameters------------------------

extern int* dihedrals_index; //index of all dihedrals per each atom...
extern int* dihedrals_indexd; //index of all dihedrals per atom...

//count of different folds for the a specific dihedral type...
//Range 1-5, for folds 1, 2, 3, 4, 6.
extern unsigned char dihedral_type_count[MAX_DIHED_TYPE];
extern unsigned char* dihedral_type_countd;

extern int* dihedrals_set; //set of dihedrals present in the system.
extern int* dihedrals_setd; //set of dihedrals present in the system.

//#define Dihed_Countd  32000
//#define Dihed_Count  Dihed_Countd

extern int Dihed_Count;
__constant__ __device__ int Dihed_Countd;

//set of dihedrals types (atom types) present in the system.
extern int dihedrals_set_type[4 * MAX_DIHED_TYPE + 1];

struct dihedralParameter{
	float x;
	short int n;
	float d;
};

extern dihedralParameter dihedral_prm[MAX_DIHED_TYPE * MAXDPRM];
extern dihedralParameter* dihedral_prmd; //device memory for dihedrals prms...

extern char wildcardstatus[MAX_DIHED_TYPE];

#ifdef IMPROPER
extern int* impropers_indexd;
extern int* impropers_setd;

extern int* impropers_index;
extern int* impropers_set;

extern int Improper_Count;
__constant__ __device__ int Improper_Countd;

extern int impropers_set_type[4 * MAX_DIHED_TYPE + 1];

extern float2 improper_prm[MAX_DIHED_TYPE];
extern float2* improper_prmd;
extern int Num_ImprTypes; //num_improper=0;
#endif

extern int Num_DihedTypes;

//-----------------Angle Lists, Indices and Parameters-------------------------
extern int* angles_indexd;
extern int* angles_setd;

extern int* angles_index; //index of all angles per each atom...
extern int* angles_set; //set of angles present in the system.

//#define Angle_Countd  32000
//#define Angle_Count  Angle_Countd

extern int Angle_Count;
__constant__ __device__ int Angle_Countd;

//set of angles types (atom types) present in the system.
extern int angles_set_type[3 * MAX_ANGLE_TYPE + 1];

extern float2 angleprm[MAX_ANGLE_TYPE];
extern float2* angleprmd; //device memory for angle prms...

#ifdef UREY_BRADLEY
extern float2 ureybprm[MAX_ANGLE_TYPE];
extern float2* ureybprmd;

extern int* ureyb_index;
extern int* ureyb_indexd;

extern int ureyb_set_type[3 * MAX_ANGLE_TYPE + 1];

extern int Num_UreyBTypes;
#endif

//variable to hold number of angles types from prm file...
extern int Num_AngleTypes;

//variable to hold number of angles types from psf file...
extern int Num_Hash_Angles;

//-----------------Bond Lists, Indices and Parameters--------------------------
extern int* bonds_indexd;
//int* bonds_setd;

extern int* bonds_index; //index of all bonds per each atom...

extern int Bond_Count;

//int bonds_set[3 * MAX_BOND_COUNT + 1]; //set of angles present in the system.

//set of angles types (atom types) present in the system.
extern int bonds_set_type[2 * MAX_BOND_TYPE + 1];

extern float2 bondprm[MAX_BOND_TYPE];
extern float2* bondprmd; //device memory for bond prms...

extern int Num_BondTypes; //variable to hold number of bond types from prm file...
extern int Num_Hash_Bonds; //variable to hold number of bond types from psf file...
//--------------------------------------------

//-----------------NBFIX Lists, Indices and Parameters-------------------------
extern int* nbfix_indexd;
extern int* nbfix_index; //index of all nbfix per each atom...
extern int nbfix_count;

//set of angles types (atom types) present in the system...
extern int nbfix_set_type[2 * MAX_BOND_TYPE + 1];

extern float2 nbfixprm[MAX_NBFIX_TYPE];
extern float2* nbfixprmd; //device memory for nbfix prms...

//variable to hold number of nbfix types from prm file...
extern int Num_nbfixTypes;

//variable to hold number of nbfix types from psf file...
// int Num_Hash_nbfix = 0;
//--------------------------------------------

//-----------------Non-Bond Lists---------------------------
extern unsigned int* nblistd; // Pointer to GPU device memory for storing nonbond list

// Exclusion list = Set Union of Bond Angle and dihedrals for each atom
extern int* excllist;

// Pointer to GPU device memory for storing exclusion list
extern int* excllistd;

extern unsigned long long* excl_bitvec;
extern unsigned long long* excl_bitvecd;

extern char* excl_bitvec_offset;
extern char* excl_bitvec_offsetd;

#ifdef PME_CALC //list for ewald exclusion correction = bond + angle atoms...
extern int* ewlist;
extern int* ewlistd;
#endif

#if (NBXMOD==5)
//int* list1_4d; //Pointer to GPU device memory for storing the 1-4 list
//int* list1_4; //1-4 list with only one other 1-4 atom per dihedral...

// 1-4 VDW parameters by atom type (y = epsilon1-4, z = sigma1-4/2)
extern float2 prm1_4[MAXTYP];

//Pointer to GPU device memory for storing atomic 1-4 vdw parameters
extern float4* prm1_4d;
#endif

//Atomic and Nonbond parameters by atom type
//(x = mass, y = LJ epsilon, z = LJ sigma/2)
extern float4 prm[MAXTYP];

extern float4* prmd; // Pointer to GPU device memory for storing atomic parameters

extern int Num_NBTypes;
//------------------ NBFIX parameters
extern int Num_NBFixTypes;

//-------------------Other Parameters and Constants-------------------------

//Pointer to GPU device memory storing flag whether nonbond list needs updating
extern int* nupdated;

////////Minimization////////////
extern float cgfac; //Conjugated Gradient Factor
extern float* cgfacd; //Conjugated Gradient Factor necessary for minimization...

extern float sdfac; //Steepest Descent Factor
extern float* sdfacd; //Steepest Descent Factor necessary for minimization...

//0 = no(or quit) minimization, 1 = conjugate gradient, 2 = steepest descent...
extern char minimization_method;

extern float cgrate;
extern float sdrate;
///////////////////////////////

//Device variables to store atomic positions, velocity and accleration
extern float4* r4d; //Pointer to GPU device memory to store atomic coordinates

//Pointer to GPU device memory to store previous
//atomic coordinates for constraints...
extern float4* r4shaked;

extern float4* v4d; //Pointer to GPU device memory to store velocities

// npt
extern float4* p1h;
extern float4* p1d; //Pointer to GPU device memory to store P1

extern float4* p2h;
extern float4* p2d; //Pointer to GPU device memory to store P2

extern float4* p3h;
extern float4* p3d; //Pointer to GPU device memory to store P3

extern float4* p4h;
extern float4* p4d; //Pointer to GPU device memory to store P4

extern float4* r1h;
extern float4* r1d; //Pointer to GPU device memory to store temporarly coordinates
// end npt

extern float4* f4d; //Pointer to GPU device memory to hold atomic forces...

extern float4* f4d_bonded;
extern float4* f4d_nonbond;

#ifdef USE_CONSFIX
extern float4* f4d_nonbond0;
extern float4* f4d_nonbond1;
#endif

//Pointer to GPU device memory to store previous forces for energy minimization
extern float4* prev_f4d;

//Pointer to GPU device memory to store search direction for minimzation...
extern float3* Htd;

//Pointer to GPU device memory to store previously
//known good minimized structure...
extern float4* min_r4d;

extern float minEnergy; //Minimized energy...

//Pointer to GPU device memory for old coordinates (last nonbond list update)
extern float4* roldd;

//Host variables to store atomic positions, velocity and accleration
extern float4* r; //Atomic coordinates in x, y, and z; charge in w
extern float4* rv; //Atomic velocities
extern float4* f4h; //Host memory for Atomic Forces..
extern float4* f4h_bonded;
extern float4* f4h_nonbond;

#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
extern float4* f4h_nonbond0;
extern float4* f4h_nonbond1;
#endif
#endif

extern size_t f3size;
extern size_t f4size; //Size of r4d and f4d, equal to WorkgroupSize * sizeof(float4)
extern size_t fsize; //WorkgroupSize * sizeof(float)
extern size_t f2size;

//Size of bond and angle lists, needs to be multiplied by # elements, equals
extern size_t isize;

extern size_t uisize; //WorkgroupSize * sizeof(int)
extern size_t i2size;
extern size_t i4size;

//Size of shared memory to allocate per block,
//equal to BlockSize * sizeof(float4)
extern size_t shSize;

//Smallest Number of threads greater than nAtom,
//that is an integral multiple of BlockSize....
extern int WorkgroupSize;

extern float Cuton; //cut-on for calculation of Switching function(VdW)

//cut-off for calculation of nonbonded forces(elec and VdW)
//and Switching Function(VdW)..
extern float Cutoff;

// cutoff for calculation of nonbond list; list is updated only if an atom has
// moved more than 0.5*(CutMax-Cutoff)
extern float CutMax;

extern float Swcoeff1; //Numerator coefficient for Switching Function...
extern float Swcoeff2; //Numerator coefficient for Switching Function...

//Reciprocal of Denominator coefficient for Switching Function...
extern float Swcoeff3;

__constant__ __device__ float beta; //Constant GPU memory for KAPPa
__constant__ __device__ float betaSqr;
__constant__ __device__ int natomd; //Constant GPU memory for number of atoms

//Constant GPU memory for number of atoms
__constant__ __device__ int natomwosegd;

//Constant GPU memory for number of constaints
__constant__ __device__ int numconstraintsd;

 //Constant GPU memory for number of constaints
__constant__ __device__ int num_types_presentd;

#ifdef PCONSTANT
extern float4* boxVelocd;
extern float4* boxAcceld;

//device memory to store half the box lengths to
//compute minimum image distance..
extern float4* boxHd;
extern float4* boxLengthd;
extern float4* prev_boxLengthd;
extern float4* ReciprocalBoxSquareRatiod;

//ratio of the sides of the boxes to preserve geometry...
__constant__ __device__ float4 boxLengthRatiod;
__constant__ __device__ float pmass_cubicd;
__constant__ __device__ float pRefd;
extern float pmass_cubic; //Andersen Piston Mass for constant pressure simulations..
extern float pRef; //External reference pressure for pconstant simulations...

//host variable to keep track of Region Velocity...
extern float4 RegionVeloc;

#else
//Constant GPU memory for storing box lengths to calculate minimum
__constant__ __device__ float4 boxH;
__constant__ __device__ float4 box;
#endif

__constant__ __device__ float cutoffd; //Constant GPU memory for cutoff
__constant__ __device__ float cutmaxd; //Constant GPU memory for cutmax

//Constant GPU memory for cutcheck = 0.5*(cutmax-cutoff)
__constant__ __device__ float cutcheckd;
__constant__ __device__ float deltaTd; //Constant GPU memory for deltaT

//Constant GPU memory for initial (reference) temperature
//__constant__ __device__ float initTempd;

//Constant GPU memory for initial (reference) temperature
//__constant__ __device__ float currentTempd;

__constant__ __device__ float taud; //Constant GPU memory for tau

// intermediate values for shifted force vdw
__constant__ __device__ float cutoff7; //cutoff^7
__constant__ __device__ float cutoff13; //cutoff^13

#ifdef VSHIFT
__constant__ __device__ float vdwC1d; //Constant Shifting for vdw energy
__constant__ __device__ float vdwC2d; //Constant Shifting for vdw energy
#endif

#ifdef VSWITCH
//Additive numerator coefficient for Switching Function
__constant__ __device__ float Swcoeff1d;

//Additive numerator coefficient for Switching Function
__constant__ __device__ float Swcoeff2d;

//Reciprocal of Denominator of Switching function = (Cuttoff^2 - Cuton^2)^3;
__constant__ __device__ float Swcoeff3d;

__constant__ __device__ float Cutond;
#endif

//device variable for WorkgroupSize..
__constant__ __device__ int WorkgroupSized;

extern FILE* coordfile; //File pointer to md.conf for reading coordinates
extern FILE* trajfile; //File pointer to xyz file to write trajectories
extern FILE* conffile; //File pointer to input file for reading configurations
extern FILE* outfile; //File pointer to xyz file to write trajectories
extern FILE* psffile; //File pointer to psf file
extern FILE* prmfile; //File pointer to prm file

//File pointer to topology file... only used to read atom type information..
extern FILE* topfile;

extern FILE* logfile;

extern unsigned int device_memory_usage;
extern int type[NMAX]; //Atomic type numeric ids...
extern int atom_typeid[NMAX]; //Atomic type numeric ids...

//Atom type in string format..
//CHARMM atom types in psf file different from global atom types..
extern char atom_type[NMAX][10];

//Atom types actually present in the system read from psf file..
extern char types_present[MAXTYP][10];

extern char type_2_name[MAXTYP][10];
extern int num_types_present;

// constfix
//Segment type in string format..
//CHARMM segment types in psf file - second column
extern char seg_type[NMAX][10];

//Segment type id in integer format.. 0: water or segments not considered;
//1 or higher segment id, i.e., SEG1=1, SEG2=2, and so on
extern int seg_typeid[NMAX];

extern int* seg_typeidd; //Pointer to GPU device memory for storing segment id

// npt
extern int molid[NMAX]; //Molecule/group assignments
extern float4 molmass[NMAX]; //Mass of molecule the atom belongs to
extern int* typed; //Pointer to GPU device memory for storing types
extern int* molidd; //Pointer to GPU device memory for storing molecule assignments

//Pointer to GPU device memory for storing molecule mass for the atom's molecule
extern float4* molmassd;

//Pointer to GPU device memory for storing nonbond Cell list
extern unsigned int* cell_nonbond;

//Pointer to GPU device memory for storing nonbond cell neighbor count
extern int* num_nonbond;

__constant__ __device__ float CELL_X;
__constant__ __device__ float CELL_Y;
__constant__ __device__ float CELL_Z;

#ifdef PCONSTANT
//int4 to store number of cells in the x y and z directions,
//w = total number of cells...
extern int4* numcellsd;

__constant__ __device__ int4 max_numcells;

#else	//no PCONSTANT
__constant__ __device__ int4 num_cells;
__constant__ __device__ int total_cells;

//__constant__ __device__ int num_cells_x;
//__constant__ __device__ int num_cells_y;
//__constant__ __device__ int num_cells_z;
#endif

#ifdef USE_CONSFIX
//consfix texture memory
//texture reference for accessing parameter array
texture<int, 1, cudaReadModeElementType> texsegtype;
#endif

//npt
//texture reference for accessing parameter array
texture<float4, 1, cudaReadModeElementType> texmolmass;

//texture reference for accessing atomic coordinates
texture<float4, 1, cudaReadModeElementType> texcrd;

//texture reference for accessing scaled atomic coordinates
texture<float4, 1, cudaReadModeElementType> texsclcrd;

//texture reference for accessing parameter array
texture<float4, 1, cudaReadModeElementType> texprm;

//texture reference for accessing parameter array
texture<float, 1, cudaReadModeElementType> texmolid;

//texture reference for accessing parameter array
texture<float2, 1, cudaReadModeElementType> texprm1_4;

//texture reference for accessing parameter array
texture<float2, 1, cudaReadModeElementType> texnbfixprm;

//texture reference for storing the spline M4...
texture<float, 1, cudaReadModeElementType> texM4;

//texture reference for storing the derivative of M4...
texture<float, 1, cudaReadModeElementType> texdM4;

//texture reference for accessing type array
texture<int, 1, cudaReadModeElementType> textype;

#ifdef PCONSTANT
//texture reference for accessing half box lengths...
texture<float4, 1, cudaReadModeElementType> texboxH;

//texture reference for accessing box lengths...
texture<float4, 1, cudaReadModeElementType> texbox;

//texture reference for accessing fraction of boxlegth/fftsize
texture<float4, 1, cudaReadModeElementType> texfrac;

//texture reference for accessing ReciprocalBoxSquareRatio
texture<float4, 1, cudaReadModeElementType> texRBSRat;

//texture reference for accessing number of cells...
texture<int4, 1, cudaReadModeElementType> texnumcells;
#endif

#ifdef PROFILING

extern double profile_times[];

#define BOND  0
#define ANGLE  1
#define DIHED  2
#define BONDED  0
#define NONBOND  3
#define CELLBUILD  4
#define CELLUPDATE  5
#define CELLCLEAN  6
#define CHARGESPREAD  7
#define BCMULTIPLY  8
#define PMEFORCE  9
#define NBBUILD  10
#define LATTICEBUILD  11
#define HALFKICK  12
#define COORDSUPDATE  13
#define LATTICEUPDATE  14
#define CUDAFFT  15
#define INIT  16
#define REDUCE  17
#define CONSTRAINTS  18
#define EVALPROPS  19

extern int nblist_call_count;
#endif

extern double cpu1;

//===diagnosis======
#ifdef DIAGNOSE
#define NUM_DIAG  7
extern float* diagnose_d;
#endif

#ifdef ANGLE_DEBUG
#define NUM_ANGLE_DEBUG  3
extern float* angles_debugd;
#endif

#ifdef BOND_DEBUG
#define NUM_BOND_DEBUG  3
extern float* bonds_debugd;
#endif

#ifdef DIHED_DEBUG
#define NUM_DIHED_DEBUG  3
extern float* dihedrals_debugd;
#endif

#ifdef DEBUG_NONB
#define NUM_DEBUG_NONB  4
extern float* debug_nonbd;
#endif

#ifdef NUM_DEBUG_CELL
extern float* debug_celld;
#endif

#ifdef DEBUG_PME
#define NUM_DEBUG_PME  1
extern float* pme_debug_d;
#endif

#ifdef DEBUG_NBLIST
#define NUM_DEBUG_NBLIST  4
extern float* debug_nblistd;
#endif

//====end diagnosis===

//--------    GLOBAL VARIABLES    ---    cell_based_kernels.cu     ------

__device__ unsigned int retirementCount = 0;

#endif //_GLOBALS_H_
