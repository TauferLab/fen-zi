/*********************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
//            Michela Taufer
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/*********************************************************************************/

// #define DEBUG_CONSFIX


// Turn on features to assess and guarantee numerical reproducibility
// 1. Prints nonbond lists of each atomid to file
// 2. Sorts the nonbond lists of each atomid prior to calling nonbondforce kernel 
//    in ComputeAccelGPU() host function. 
// 3. 
#define REPRO

#ifndef _DEFS_H
#define _DEFS_H

#define FENZI_VERSION_MAJOR @FENZI_VERSION_MAJOR@
#define FENZI_VERSION_MINOR @FENZI_VERSION_MINOR@

#define FENZI_VERSION FENZI_VERSION_(FENZI_VERSION_MAJOR, FENZI_VERSION_MINOR)
#define FENZI_VERSION_(M, N) (0x##M##N)

#define FENZI_VERSION_STRING FENZI_VERSION_STRING_(FENZI_VERSION_MAJOR, FENZI_VERSION_MINOR)
#define FENZI_VERSION_STRING_(M, N) FENZI_VERSION_STRING__(#M, #N)
#define FENZI_VERSION_STRING__(M, N) (M "." N)

// Depending on whether we're running inside the CUDA compiler, define the __host__
// and __device__ intrinsics, otherwise just make the functions static to prevent
// linkage issues (duplicate symbols and such)

#ifdef __CUDACC__
#  define HOST         __host__
#  define DEVICE                __device__
#  define HOSTDEVICE   __host__ __device__
#  define M_HOST       __host__
#  define M_HOSTDEVICE __host__ __device__
#else /* !__CUDACC__ */
#  define HOST         static inline
#  define DEVICE       static inline
#  define HOSTDEVICE   static inline
#  define M_HOST              inline /* note there is no static here */
#  define M_HOSTDEVICE        inline /* static has a different meaning
                                      * for class member functions    */
#endif /* !__CUDACC__ */

/* define functions */
#define STRINGIFY(x) #x
#define EXPAND(x) STRINGIFY(x)

#define __my_fadd(x,y) ((x)+(y))
#define __my_fmul(x,y) ((x)*(y))

#define __my_fdiv   __fdividef
#define __my_fsin     sinf
#define __my_fcos     cosf
#define __my_fasin    asinf
#define __my_facos    acosf
#define __my_fsqrt    sqrtf
#define __my_frsqrt   rsqrtf

#define __mycopysignf copysignf

#define __my_add3(x,y,z)   __my_fadd ((x), __my_fadd ((y),                (z)))
#define __my_add4(x,y,z,w) __my_fadd ((x), __my_fadd ((y), __my_fadd ((z),(w))))
#define __my_mul3(x,y,z)   __my_fmul ((x), __my_fmul ((y),                (z)))
#define __my_mul4(x,y,z,w) __my_fmul ((x), __my_fmul ((y), __my_fmul ((z),(w))))

#define nearest_image(x,LH)    ((x) -     ( ((x)>(LH))?(2*(LH)):0 ) + ( ((x)<(-1*(LH)))?(2*(LH)):0 ))
#define update_coords(x,L )    ((x) -     ( ((x)>=(L))?(L)     :0 ) + ( ((x)<        0)?(L)     :0 ))
#define update_coords_cpu(x,L) ((x) - (L)*( (int)(x)/(L)          ) + ( ((x)<        0)?(L)     :0 ))

#ifdef _WIN32
#  define NAN -5000000
#  define __myisnan(x) ((((x)>NAN)&&((x)<-1*NAN))?0:1)
#else /* !_WIN32 */
#  define __myisnan(x) isnan(x)
#endif /* !_WIN32 */

/* these are system constants and should
 * not be changed to arbitrary values.  */
#define MAX_BLOCK_SIZE        512                  /* maximum GPU thread
                                                    * block size        */
#define GPU_WARP_SIZE          32
#define NMAX               300000
#define PI                      3.141592653589793f
#define PI_OVER_2               1.570796326794897f
#define SQRT_PI                 1.772453850905516f
#define TWO_OVER_SQRT_PI        1.128379167095513f
#define FOUR_OVER_SQRT_PI       2.256758334191025f
#define MEGA_BYTE         1048576
#define CCTWO_OVER_SQRT_PI      374.702675424073f

/* Constants for the random number generator */
#define D2P31M      2147483647.0f
#define DMUL             16807.0f
#define PMASS             2000.0f    /* piston mass for constant pressure
                                      * simulations                      */
#define ATM             101325.0f
#define PREF                 1.0f    /* reference pressure in atmospheres */
                                     /* (9.862E-6, 98066.5f)              */

#define PMASS_CUBIC       1.0000e-5f
#define PU                1.4586e-5  /* pressure units
                                      * (Kcal/mol/A^3 -> N/m^2) */

#define SHIFTING
#define PRECISION float4 /* double4 */
// #define RESTRAINTS
// #define MY_NEAREST_IMAGE
// #define PRINT_TO_FILE
// #define BINARY_CHKPT
// #define ASCII_CHKPT

#define NBPART  2 // 2 
#define BPART   2 // 2  

/* PERFORMANCE FEATURES */
// #define PROFILING
#define NBXMOD  5
#define VSWITCH 1
#define VSHIFT  0
#define HFAC    2

#define PME_CALC
#define UREY_BRADLEY
#define IMPROPER
#define NBBUILD_SHAREDMEM
#define EXCL_BITVEC
#define COMBINED_KERNELS
#define DIHED_SHAREDMEM
/* END PERFORMANCE FEATURES */

/* CHARMM COMPLIANCE */
/* #define HALF_NBUPDATE */
#define NONBOND_RADIUS          cutmaxd /* (cutmaxd, cutoffd) */
#define CELLSIZE                CutMax  /* (CutMax , Cutoff ) */
#define BCMULTIPLY_BLOCKSIZE         64  
#define LATTICEUPDATE_BLOCKSIZE     128 
#define PMEFORCE_BLOCKSIZE          512 /* (128, 512, dynadimBlock) */
#define CHARGESPREAD_BLOCKSIZE      512 /* (128,  64, dynadimBlock) */
#define CELL_BLOCKSIZE              256 // 128 /* (256) */
#define CELL_ATOMS                  256 // 128 /* (256) */
/* END CHARMM COMPLIANCE */

#ifdef PRINT_TO_FILE
#  define PRINTOUT fprintf(outfile, 
#else /* !PRINT_TO_FILE */
#  define PRINTOUT printf(
#endif /* !PRINT_TO_FILE */

/* misc. constants (for conversion
 * to AKMA units used by CHARMM)  */
//#define EPS1 80              /* dieletric constant outside the cavity */
#define RC       0.0019870000f /* gas constant = Nk (kcal/K mol)        */
#define CC     332.0716000000f /* 1/(4 pi epsilon)                      */
#define TU      20.4548000000f /* ps -> time units                      */
#define B0       0.9813664600f /* 2*(EPS1-1)/(2*EPS1+1)                 */
#define RAD2DEG 57.2957795131f /* radians -> degrees                    */

#define MAXTYP                64 /* max # of atom types                       */
#define MAXDPRM                6 /* max # of dihedral parameter sets/entry    */
#define MAX_DIHED_TYPE        64 /* max # of dihedral types (256, 484, 512)   */
#define DIHED_COUNT_PERATOM   64 /* max # of dihedrals an atom can be part of */
#define MAX_ANGLE_TYPE        64 /* max # of angle types (256, 318, 512)      */
#define ANGLE_COUNT_PERATOM   64 /* max # of angles an atom can be part of    */
#define MAX_BOND_TYPE         64 /* max # of bond types                       */
#define BOND_COUNT_PERATOM    32 /* max # of bonds an atom can be part of     */
#define EXCL_COUNT_PERATOM    32
#define MAX_NBFIX_TYPE       512  /* max # of nbfix types                       */
#define NUM_SPLINE_SAMPLES  1000

#define COMPUTE_GLOBAL_THREADID  ( gridDim.x*blockIdx.y + blockIdx.x ) * \
                                   blockDim.x + threadIdx.x;

/* #define PME_BLOCK_X 512 */
/* #define PME_GRID_X  512 */

#ifdef PME_CALC
#  define MAXFFT_X             256
#  define MAXFFT_Y             256
#  define MAXFFT_Z             256
// #  define MAXLATTICE_NEIGHBORS  512 // 256 // 192 // 64     /* (64, 192) */
// #  define beta                 0.33f
// #  define betaSqr beta*beta

#  define M2(x)           ((((x)>0.0f)&&((x)<2.0f))?(1.0f-abs((x)-1.0f)):0.0f)
#  define M3(x)           (M2((x))*((x)/2.0f) + M2(((x)-1.0f))*((3.0f-(x))/2.0f))
                          /* (x*M2(x)/2.0 + (3.0-x)*M2(x-1.0)/2.0) */
#  define M4(x)           (M3((x))*((x)/3.0f) + M3(((x)-1.0f))*((4.0f-(x))/3.0f))
#  define spline_index(x) ((int)((((x)>2)?(4-(x)):(x))*(NUM_SPLINE_SAMPLES)/2.0))
#endif /* PME_CALC */

/* set SYSTEM_SIZE to one of the following to define the system size */
#define SYSTEM_SIZE_CUSTOM (-1)
#define SYSTEM_SIZE_VSMALL ( 0)
#define SYSTEM_SIZE_SMALL  ( 1)
#define SYSTEM_SIZE_MEDIUM ( 2)
#define SYSTEM_SIZE_LARGE  ( 3)
#define SYSTEM_SIZE_CCO    ( 4)

#define SYSTEM_SIZE @FENZI_SYSTEM_SIZE@

#define CELL_BLOCKSIZE @FENZI_CELL_BLOCKSIZE@
#define CELL_ATOMS     @FENZI_CELL_ATOMS@
#define MAX_DIHED_TYPE @FENZI_MAX_DIHED_TYPE@
#define MAX_ANGLE_TYPE @FENZI_MAX_ANGLE_TYPE@
#define MAX_BOND_TYPE  @FENZI_MAX_BOND_TYPE@

#define MAXNB @FENZI_MAX_NB@
#define MAXLATTICE_NEIGHBORS @FENZI_MAX_LATTICE@

#define PCONST @FENZI_PCONST@
#if PCONST == 1
#define PCONSTANT
#endif

#define NPT @FENZI_NPT@
#if NPT == 1
#define USE_NPT
#endif

#define CONSFIX @FENZI_CONSFIX@
#if CONSFIX == 1
#define USE_CONSFIX
#endif

#endif /* !_DEFS_H */

