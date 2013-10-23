/*****************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
//            Joseph E. Davis
//            Michela Taufer (taufer@udel.edu)
//             
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/*****************************************************************************/
#include "globals.h"

#include <stdio.h>
#include <cufft.h>
#include "defs.h"

//----------------    GLOBAL VARIABLES    ---    md.h     ---------------------

// consfix
// //#define USE_CONSFIX
// //#define DEBUG_NPT
//#define EXPLICIT_EWALD_LIST

int printtostdout = 1;

/* Input parameters (read from an input file in this order) ******************/

char FileName[512]; //File name to read coordinates from
char topFileName[512]; //Topology file containing the atom type information...
char psfFileName[512]; //File name to read psf (topology) from
char prmFileName[512]; //File name to read parameters from
char trajFileName[512]; //File name to write trajectories to
char filename_prefix[512] = "";

//char restartfile_format[512] = "";
//char restartfile_prefix[512] = "", restartfile_suffix[512] = "";

char restartfilename[512] = "";
int keepcheckpoint = 0;
int c36 = 0;

float InitTemp = 298.0f; //Starting temperature
//float CurrentTemp; //Current temperature
float DeltaT = 0.001f; //Size of a time step
float Tau; //Coupling constant for Berendsen thermostat
int VAssign; //frequeny for reassigning velocities
int StartCount = 0; //Starting step number - 0 is first run, n otherwise
int CheckpointTimestep = 0;
int StepLimit = 10000; //Number of time steps to be simulated
int StepAvg = 100; //Reporting interval for statistical data
int StepStore = 100000; //Reporting interval for checkpoing
int MinimizeStepLimit = 0; //Number of minimization steps...
int BlockSize = 128; //Thread block size = p, number of particles per tile for
                     //GPU force calculation
int TrjFreq = StepAvg; // Frequency of trajectory frames

float KAPPa = 0.032f; //KAPPa - it can be read in .in
float KAPPaSqr = KAPPa * KAPPa;
int INBFRQ = -1; // INBFRQ is the frequency to update non-bond list - it can be read in .in^M

int PrintLevel; //determines how much output is printed
double seed; //seed for random number generator

float4 Region; //box lengths
float4 CellSize = {9.5f, 9.5f, 9.5f, 9.5f};
float4 RegionH; //Half the box lengths
float DeltaTH; //Half the time step
float CutCheck; //0.5*(CutMax-Cutoff), for checking nonbond list
int4 NumCells;

#ifdef PCONSTANT
//Maximum number of cells that can be expanded into due to volume fluctuations
int4 MaxNumCells;
#endif

//int TotalCells;

int nAtom; //Number of atoms
int nAtomwoseg; //Number of atoms

dim3 BondeddimGrid;
dim3 NBdimGrid;
dim3 BondeddimBlock;
dim3 NBdimBlock;

dim3 BCMuldimGrid;
dim3 BCMuldimBlock;

dim3 LatUpdimGrid;
dim3 LatUpdimBlock;

dim3 PMEForceBlock;
dim3 PMEForceGrid;

dim3 ChargeSpreadBlock;
dim3 ChargeSpreadGrid;

dim3 NBBuild_dimGrid; //(TOTAL_CELLS, 1);
dim3 NBBuild_dimBlock; //(MAX_NBCELL, 1);

dim3 CellClean_dimGrid;
dim3 CellClean_dimBlock;

dim3 DynadimGrid;
dim3 DynadimBlock;

//////////////////////////////////////////
#ifdef PME_CALC
//////////////////////////////////////////

int fftx;
int ffty;
int fftz;

float* ePME32;
double* ePME;
float* ePMEd;

float* eEwexcl32;
double* eEwexcl;
float* eEwexcld;

//int* ew_correction_list;
//int* ew_correction_listd;

//CUFFT for PME
cufftHandle plan;

float* M4d; //Spline M4...
float* dM4d; //derivative of M4...

int* cellL;
int* numL;

float4* r4d_scaled; // Pointer to GPU device memory for Scaled Coordinates
float4* prev_r4d_scaled;
int4* disp;//integer displacements in scaled coordinates
           //used to update lattice list..

int Total_PME_Lattices;

//B and C for PME
//__constant__ __device__ cufftComplex b1d[FFTX], b2d[FFTY], b3d[FFTZ];

//__constant__ __device__ float sqr_b1d[MAXFFT_X];
//__constant__ __device__ float sqr_b2d[MAXFFT_Y];
//__constant__ __device__ float sqr_b3d[MAXFFT_Z];

//__constant__ __device__ float c1d[MAXFFT_X];
//__constant__ __device__ float c2d[MAXFFT_Y];
//__constant__ __device__ float c3d[MAXFFT_Z];

//__constant__ __device__ float erfc_table[NUM_ERFC_SAMPLES + 1];
//__constant__ __device__ float exp_table[NUM_ERFC_SAMPLES + 1];

//Q Matrix in the device..
cufftComplex* Qd;
//cufftComplex *Qd1;

#ifdef PCONSTANT
float4* fracd;
#else
//__constant__ __device__ float4 frac;
#endif

//__constant__ __device__ int FFTX;
//__constant__ __device__ int FFTY;
//__constant__ __device__ int FFTZ;

//__constant__ __device__ float FFTXH;
//__constant__ __device__ float FFTYH;
//__constant__ __device__ float FFTZH;

//__constant__ __device__ int TOTAL_PME_LATTICES;

//////////////////////////////////////////
#endif//#ifdef PME_CALC
//////////////////////////////////////////

// Host arrays for holding npt properties

// Device arrays for holding npt properties
float4* propertiesd;

// Host arrays for holding atomic contributions to bond, angle, etc. energies
float* ebnd;
float* eang;
float* evdw;
float* eelec;
float* edihed;
float4* virial;

// Device arrays for holding atomic contributions to bond, angle, etc. energies
float* ebndd;
float* eangd;
float* evdwd;
float* eelecd;
float* edihedd;

//////////////////RESTRAINTS////////////////////////////
char seg_name[NMAX][128]; //segment name of all the atoms in system...

#ifdef USE_CONSFIX
char consfix = 0; //logical variable 0: no constrains, 1: constrains
int consfix_nseg = 0; //number of segments given in input
char consfix_segname[128][16]; //segements name
int consfix_segcount[128]; //in [i] length of segment i
char* consfix_segidd;
#endif

char restraints = 0; //status to indicate if restraints is set..
char segname0[128]; //segments to impose restraints on...
char segname1[128]; //restraints can act only between 2 segments...

char* segidd; //array to indicate segid of atoms...
char segid[NMAX];

float3* com0d;
float3* com1d;
float* mass_segid0d;
float* mass_segid1d;
float mass_segid0 = 0.0f;
float mass_segid1 = 0.0f;

//__constant__ __device__ float3 consharmdistd;
//__constant__ __device__ float consharmfcd;
float3 consharmdist = {10.0f, 10.0f, 20.0f};
float consharmfc = 0.01f;
////////////////////// END RESTRAINTS Variables...

//////////////////Shake Parameters and Variables/////////////
//#define ATOMS_IN_CLUSTER  4
//#define CONSTRAINTS_PER_ATOM  3
//#define CLUSTER_SIZE  3
//#define MAX_CLUSTERS  100000

int2* constraints;
int2* constraintsd;

unsigned char* constraints_by_atom;
unsigned char* constraints_by_atomd;

float2* constraintsprm;
float2* constraintsprmd;

int* atoms_in_cluster;
int* atoms_in_clusterd;

int num_constraints = 0;

dim3 num_cluster_blocks;
dim3 cluster_blocksize;

float shaketol = 1e-6f; //shake tolerance...
//__constant__ __device__ float shaketold;

int nClusters = 0;
//__constant__ __device__ int nClustersd;
char shake = 0;
float hfac = 1;
//////////////////End Shake Parameters and Variables/////////////

#ifdef PCONSTANT
float4* kineticd;
float4* viriald;

#ifdef PME_CALC
//__constant__ __device__ float eEwselfd;
#endif

#endif //PCONSTANT

#ifdef UREY_BRADLEY
float* eureyb;
float* eureybd;
#endif

#ifdef IMPROPER
float* eimprop;
float* eimpropd;
#endif

// Total Values...
float kinEnergy; //Kinetic energy
float potEnergy; //Potential energy
float eBond; //potential energy terms for bonds, angles, etc.
float eAngle;
float eLJ;
float eCoulomb;
double eEwself;
float totEnergy; //Total energy
float temperature; //Current temperature
int stepCount; //Current time step
int nupdate; //flag whether nonbond list needs updating
float vscale; //velocity scaling factor for thermostat

// intermediate values used for calculating shifted force vdw
float rc7;
float rc13;

//FILE* virfile;

// arrays for performing parallel sum (reduction)
//for velocity scaling (Berendsen thermostat)
//float* sum;
//float* sumd;
//int blocksizeT = 128; //separate block size for reduction

//-----------------Dihed Lists, Indices and Parameters------------------------

int* dihedrals_index; //index of all dihedrals per each atom...
int* dihedrals_indexd; //index of all dihedrals per atom...

//count of different folds for the a specific dihedral type...
//Range 1-5, for folds 1, 2, 3, 4, 6.
unsigned char dihedral_type_count[MAX_DIHED_TYPE];
unsigned char* dihedral_type_countd;

int* dihedrals_set; //set of dihedrals present in the system.
int* dihedrals_setd; //set of dihedrals present in the system.

////#define Dihed_Countd  32000
////#define Dihed_Count  Dihed_Countd

int Dihed_Count;
//__constant__ __device__ int Dihed_Countd;

//set of dihedrals types (atom types) present in the system.
int dihedrals_set_type[4 * MAX_DIHED_TYPE + 1];

// struct dihedralParameter{
	// float x;
	// short int n;
	// float d;
// };

dihedralParameter dihedral_prm[MAX_DIHED_TYPE * MAXDPRM];
dihedralParameter* dihedral_prmd; //device memory for dihedrals prms...

char wildcardstatus[MAX_DIHED_TYPE];

#ifdef IMPROPER
int* impropers_indexd;
int* impropers_setd;

int* impropers_index;
int* impropers_set;

int Improper_Count;
//__constant__ __device__ int Improper_Countd;

int impropers_set_type[4 * MAX_DIHED_TYPE + 1];

float2 improper_prm[MAX_DIHED_TYPE];
float2* improper_prmd;
int Num_ImprTypes = 0; //num_improper=0;
#endif

int Num_DihedTypes = 0;

//-----------------Angle Lists, Indices and Parameters-------------------------
int* angles_indexd;
int* angles_setd;

int* angles_index; //index of all angles per each atom...
int* angles_set; //set of angles present in the system.

////#define Angle_Countd  32000
////#define Angle_Count  Angle_Countd

int Angle_Count;
//__constant__ __device__ int Angle_Countd;

//set of angles types (atom types) present in the system.
int angles_set_type[3 * MAX_ANGLE_TYPE + 1];

float2 angleprm[MAX_ANGLE_TYPE];
float2* angleprmd; //device memory for angle prms...

#ifdef UREY_BRADLEY
float2 ureybprm[MAX_ANGLE_TYPE];
float2* ureybprmd;

int* ureyb_index;
int* ureyb_indexd;

int ureyb_set_type[3 * MAX_ANGLE_TYPE + 1];

int Num_UreyBTypes = 0;
#endif

//variable to hold number of angles types from prm file...
int Num_AngleTypes = 0;

//variable to hold number of angles types from psf file...
int Num_Hash_Angles = 0;

//-----------------Bond Lists, Indices and Parameters--------------------------
int* bonds_indexd;
//int* bonds_setd;

int* bonds_index; //index of all bonds per each atom...

int Bond_Count;

//int bonds_set[3 * MAX_BOND_COUNT + 1]; //set of angles present in the system.

//set of angles types (atom types) present in the system.
int bonds_set_type[2 * MAX_BOND_TYPE + 1];

float2 bondprm[MAX_BOND_TYPE];
float2* bondprmd; //device memory for bond prms...

int Num_BondTypes = 0; //variable to hold number of bond types from prm file...
int Num_Hash_Bonds = 0; //variable to hold number of bond types from psf file...
//--------------------------------------------

//-----------------NBFIX Lists, Indices and Parameters-------------------------
int* nbfix_indexd;
int* nbfix_index; //index of all nbfix per each atom...
int nbfix_count;

//set of angles types (atom types) present in the system...
int nbfix_set_type[2 * MAX_BOND_TYPE + 1];

float2 nbfixprm[MAX_NBFIX_TYPE];
float2* nbfixprmd; //device memory for nbfix prms...

//variable to hold number of nbfix types from prm file...
int Num_nbfixTypes = 0;

//variable to hold number of nbfix types from psf file...
// int Num_Hash_nbfix = 0;
//--------------------------------------------

//-----------------Non-Bond Lists---------------------------
unsigned int* nblistd; // Pointer to GPU device memory for storing nonbond list

// Exclusion list = Set Union of Bond Angle and dihedrals for each atom
int* excllist;

// Pointer to GPU device memory for storing exclusion list
int* excllistd;

unsigned long long* excl_bitvec;
unsigned long long* excl_bitvecd;

char* excl_bitvec_offset;
char* excl_bitvec_offsetd;

#ifdef PME_CALC //list for ewald exclusion correction = bond + angle atoms...
int* ewlist;
int* ewlistd;
#endif

#if (NBXMOD==5)
//int* list1_4d; //Pointer to GPU device memory for storing the 1-4 list
//int* list1_4; //1-4 list with only one other 1-4 atom per dihedral...

// 1-4 VDW parameters by atom type (y = epsilon1-4, z = sigma1-4/2)
float2 prm1_4[MAXTYP];

//Pointer to GPU device memory for storing atomic 1-4 vdw parameters
float4* prm1_4d;
#endif

//Atomic and Nonbond parameters by atom type
//(x = mass, y = LJ epsilon, z = LJ sigma/2)
float4 prm[MAXTYP];

float4* prmd; // Pointer to GPU device memory for storing atomic parameters

int Num_NBTypes = 0;
//------------------ NBFIX parameters
int Num_NBFixTypes = 0;

//-------------------Other Parameters and Constants-------------------------

//Pointer to GPU device memory storing flag whether nonbond list needs updating
int* nupdated;

////////Minimization////////////
float cgfac = 1e-10f; //Conjugated Gradient Factor
float* cgfacd; //Conjugated Gradient Factor necessary for minimization...

float sdfac = 1e-4f; //Steepest Descent Factor
float* sdfacd; //Steepest Descent Factor necessary for minimization...

//0 = no(or quit) minimization, 1 = conjugate gradient, 2 = steepest descent...
char minimization_method = 1;

float cgrate = 0.1f;
float sdrate = 0.01f;
///////////////////////////////

//Device variables to store atomic positions, velocity and accleration
float4* r4d; //Pointer to GPU device memory to store atomic coordinates

//Pointer to GPU device memory to store previous
//atomic coordinates for constraints...
float4* r4shaked;

float4* v4d; //Pointer to GPU device memory to store velocities

// npt
float4* p1h;
float4* p1d; //Pointer to GPU device memory to store P1

float4* p2h;
float4* p2d; //Pointer to GPU device memory to store P2

float4* p3h;
float4* p3d; //Pointer to GPU device memory to store P3

float4* p4h;
float4* p4d; //Pointer to GPU device memory to store P4

float4* r1h;
float4* r1d; //Pointer to GPU device memory to store temporarly coordinates
// end npt

float4* f4d; //Pointer to GPU device memory to hold atomic forces...

float4* f4d_bonded;
float4* f4d_nonbond;

#ifdef USE_CONSFIX
float4* f4d_nonbond0;
float4* f4d_nonbond1;
#endif

//Pointer to GPU device memory to store previous forces for energy minimization
float4* prev_f4d;

//Pointer to GPU device memory to store search direction for minimzation...
float3* Htd;

//Pointer to GPU device memory to store previously
//known good minimized structure...
float4* min_r4d;

float minEnergy = 0; //Minimized energy...

//Pointer to GPU device memory for old coordinates (last nonbond list update)
float4* roldd;

//Host variables to store atomic positions, velocity and accleration
float4* r; //Atomic coordinates in x, y, and z; charge in w
float4* rv; //Atomic velocities
float4* f4h; //Host memory for Atomic Forces..
float4* f4h_bonded;
float4* f4h_nonbond;

#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
float4* f4h_nonbond0;
float4* f4h_nonbond1;
#endif
#endif

size_t f3size;
size_t f4size; //Size of r4d and f4d, equal to WorkgroupSize * sizeof(float4)
size_t fsize; //WorkgroupSize * sizeof(float)
size_t f2size;

//Size of bond and angle lists, needs to be multiplied by # elements, equals
size_t isize;

size_t uisize; //WorkgroupSize * sizeof(int)
size_t i2size;
size_t i4size;

//Size of shared memory to allocate per block,
//equal to BlockSize * sizeof(float4)
size_t shSize;

//Smallest Number of threads greater than nAtom,
//that is an integral multiple of BlockSize....
int WorkgroupSize;

float Cuton = 7.0f; //cut-on for calculation of Switching function(VdW)

//cut-off for calculation of nonbonded forces(elec and VdW)
//and Switching Function(VdW)..
float Cutoff = 8.0f;

// cutoff for calculation of nonbond list; list is updated only if an atom has
// moved more than 0.5*(CutMax-Cutoff)
float CutMax = 9.5f;

float Swcoeff1; //Numerator coefficient for Switching Function...
float Swcoeff2; //Numerator coefficient for Switching Function...

//Reciprocal of Denominator coefficient for Switching Function...
float Swcoeff3;

//__constant__ __device__ float beta; //Constant GPU memory for KAPPa
//__constant__ __device__ float betaSqr;
//__constant__ __device__ int natomd; //Constant GPU memory for number of atoms

//Constant GPU memory for number of atoms
//__constant__ __device__ int natomwosegd;

//Constant GPU memory for number of constaints
//__constant__ __device__ int numconstraintsd;

 //Constant GPU memory for number of constaints
//__constant__ __device__ int num_types_presentd;

#ifdef PCONSTANT
float4* boxVelocd;
float4* boxAcceld;

//device memory to store half the box lengths to
//compute minimum image distance..
float4* boxHd;
float4* boxLengthd;
float4* prev_boxLengthd;
float4* ReciprocalBoxSquareRatiod;

//ratio of the sides of the boxes to preserve geometry...
//__constant__ __device__ float4 boxLengthRatiod;
//__constant__ __device__ float pmass_cubicd;
//__constant__ __device__ float pRefd;
float pmass_cubic; //Andersen Piston Mass for constant pressure simulations..
float pRef; //External reference pressure for pconstant simulations...

//host variable to keep track of Region Velocity...
float4 RegionVeloc = {0.0f, 0.0f, 0.0f, 0.0f};

#else
//Constant GPU memory for storing box lengths to calculate minimum
//__constant__ __device__ float4 boxH;
//__constant__ __device__ float4 box;
#endif

//__constant__ __device__ float cutoffd; //Constant GPU memory for cutoff
//__constant__ __device__ float cutmaxd; //Constant GPU memory for cutmax

//Constant GPU memory for cutcheck = 0.5*(cutmax-cutoff)
//__constant__ __device__ float cutcheckd;
//__constant__ __device__ float deltaTd; //Constant GPU memory for deltaT

//Constant GPU memory for initial (reference) temperature
////__constant__ __device__ float initTempd;

//Constant GPU memory for initial (reference) temperature
////__constant__ __device__ float currentTempd;

//__constant__ __device__ float taud; //Constant GPU memory for tau

// intermediate values for shifted force vdw
//__constant__ __device__ float cutoff7; //cutoff^7
//__constant__ __device__ float cutoff13; //cutoff^13

#ifdef VSHIFT
//__constant__ __device__ float vdwC1d; //Constant Shifting for vdw energy
//__constant__ __device__ float vdwC2d; //Constant Shifting for vdw energy
#endif

#ifdef VSWITCH
//Additive numerator coefficient for Switching Function
//__constant__ __device__ float Swcoeff1d;

//Additive numerator coefficient for Switching Function
//__constant__ __device__ float Swcoeff2d;

//Reciprocal of Denominator of Switching function = (Cuttoff^2 - Cuton^2)^3;
//__constant__ __device__ float Swcoeff3d;

//__constant__ __device__ float Cutond;
#endif

//device variable for WorkgroupSize..
//__constant__ __device__ int WorkgroupSized;

FILE* coordfile; //File pointer to md.conf for reading coordinates
FILE* trajfile; //File pointer to xyz file to write trajectories
FILE* conffile; //File pointer to input file for reading configurations
FILE* outfile; //File pointer to xyz file to write trajectories
FILE* psffile; //File pointer to psf file
FILE* prmfile; //File pointer to prm file

//File pointer to topology file... only used to read atom type information..
FILE* topfile;

FILE* logfile;

unsigned int device_memory_usage = 0;
int type[NMAX]; //Atomic type numeric ids...
int atom_typeid[NMAX]; //Atomic type numeric ids...

//Atom type in string format..
//CHARMM atom types in psf file different from global atom types..
char atom_type[NMAX][10];

//Atom types actually present in the system read from psf file..
char types_present[MAXTYP][10];

char type_2_name[MAXTYP][10];
int num_types_present = 0;

// constfix
//Segment type in string format..
//CHARMM segment types in psf file - second column
char seg_type[NMAX][10];

//Segment type id in integer format.. 0: water or segments not considered;
//1 or higher segment id, i.e., SEG1=1, SEG2=2, and so on
int seg_typeid[NMAX];

int* seg_typeidd; //Pointer to GPU device memory for storing segment id

// npt
int molid[NMAX]; //Molecule/group assignments
float4 molmass[NMAX]; //Mass of molecule the atom belongs to
int* typed; //Pointer to GPU device memory for storing types
int* molidd; //Pointer to GPU device memory for storing molecule assignments

//Pointer to GPU device memory for storing molecule mass for the atom's molecule
float4* molmassd;

//Pointer to GPU device memory for storing nonbond Cell list
unsigned int* cell_nonbond;

//Pointer to GPU device memory for storing nonbond cell neighbor count
int* num_nonbond;

//__constant__ __device__ float CELL_X;
//__constant__ __device__ float CELL_Y;
//__constant__ __device__ float CELL_Z;

#ifdef PCONSTANT
//int4 to store number of cells in the x y and z directions,
//w = total number of cells...
int4* numcellsd;

//__constant__ __device__ int4 max_numcells;

#else	//no PCONSTANT
//__constant__ __device__ int4 num_cells;
//__constant__ __device__ int total_cells;

////__constant__ __device__ int num_cells_x;
////__constant__ __device__ int num_cells_y;
////__constant__ __device__ int num_cells_z;
#endif

#ifdef USE_CONSFIX
//consfix texture memory
//texture reference for accessing parameter array
//texture<int, 1, cudaReadModeElementType> texsegtype;
#endif

//npt
//texture reference for accessing parameter array
//texture<float4, 1, cudaReadModeElementType> texmolmass;

//texture reference for accessing atomic coordinates
//texture<float4, 1, cudaReadModeElementType> texcrd;

//texture reference for accessing scaled atomic coordinates
//texture<float4, 1, cudaReadModeElementType> texsclcrd;

//texture reference for accessing parameter array
//texture<float4, 1, cudaReadModeElementType> texprm;

//texture reference for accessing parameter array
//texture<float, 1, cudaReadModeElementType> texmolid;

//texture reference for accessing parameter array
//texture<float2, 1, cudaReadModeElementType> texprm1_4;

//texture reference for accessing parameter array
//texture<float2, 1, cudaReadModeElementType> texnbfixprm;

//texture reference for storing the spline M4...
//texture<float, 1, cudaReadModeElementType> texM4;

//texture reference for storing the derivative of M4...
//texture<float, 1, cudaReadModeElementType> texdM4;

//texture reference for accessing type array
//texture<int, 1, cudaReadModeElementType> textype;

#ifdef PCONSTANT
//texture reference for accessing half box lengths...
//texture<float4, 1, cudaReadModeElementType> texboxH;

//texture reference for accessing box lengths...
//texture<float4, 1, cudaReadModeElementType> texbox;

//texture reference for accessing fraction of boxlegth/fftsize
//texture<float4, 1, cudaReadModeElementType> texfrac;

//texture reference for accessing ReciprocalBoxSquareRatio
//texture<float4, 1, cudaReadModeElementType> texRBSRat;

//texture reference for accessing number of cells...
//texture<int4, 1, cudaReadModeElementType> texnumcells;
#endif

#ifdef PROFILING

double profile_times[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

//#define BOND  0
//#define ANGLE  1
//#define DIHED  2
//#define BONDED  0
//#define NONBOND  3
//#define CELLBUILD  4
//#define CELLUPDATE  5
//#define CELLCLEAN  6
//#define CHARGESPREAD  7
//#define BCMULTIPLY  8
//#define PMEFORCE  9
//#define NBBUILD  10
//#define LATTICEBUILD  11
//#define HALFKICK  12
//#define COORDSUPDATE  13
//#define LATTICEUPDATE  14
//#define CUDAFFT  15
//#define INIT  16
//#define REDUCE  17
//#define CONSTRAINTS  18
//#define EVALPROPS  19

int nblist_call_count = 0;
#endif

double cpu1 = 0;

//===diagnosis======
#ifdef DIAGNOSE
//#define NUM_DIAG  7
float* diagnose_d;
#endif

#ifdef ANGLE_DEBUG
//#define NUM_ANGLE_DEBUG  3
float* angles_debugd;
#endif

#ifdef BOND_DEBUG
//#define NUM_BOND_DEBUG  3
float* bonds_debugd;
#endif

#ifdef DIHED_DEBUG
//#define NUM_DIHED_DEBUG  3
float* dihedrals_debugd;
#endif

#ifdef DEBUG_NONB
//#define NUM_DEBUG_NONB  4
float* debug_nonbd;
#endif

#ifdef NUM_DEBUG_CELL
float* debug_celld;
#endif

#ifdef DEBUG_PME
//#define NUM_DEBUG_PME  1
float* pme_debug_d;
#endif

#ifdef DEBUG_NBLIST
//#define NUM_DEBUG_NBLIST  4
float* debug_nblistd;
#endif

//====end diagnosis===

//--------    GLOBAL VARIABLES    ---    cell_based_kernels.cu     ------

//__device__ unsigned int retirementCount = 0;
