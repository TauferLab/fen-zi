/*****************************************************************************/
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
/*****************************************************************************/

#ifndef _CUCOMPLEXOPS_H_
#define _CUCOMPLEXOPS_H_

#include "globals.h"

#include <vector_types.h>  // required for float2

// Struct alignment is handled differently between the CUDA compiler and other
// compilers (e.g. GCC, MS Visual C++ .NET)
#ifdef __CUDACC__
#   define ALIGN(x)  __align__(x)
#else
#   if defined(_MSC_VER) && (_MSC_VER >= 1300)
    //Visual C++ .NET and later
#      define ALIGN(x) __declspec(align(x)) 
#   else
#      if defined(__GNUC__)
// GCC
#         define ALIGN(x)  __attribute__ ((aligned (x)))
#      else
// all other compilers
#         define ALIGN(x) 
#      endif //defined(__GNUC__)
#   endif //defined(_MSC_VER) && (_MSC_VER >= 1300)
#endif //__CUDACC__

//Somehow in emulation mode the code won't compile Mac OS X 1.1 CUDA SDK when
//the operators below make use of references (compiler bug?). So instead we
//compile the code to pass everything through the stack. Slower, but works.
//I am not sure how the Linux CUDA SDK will behave, so currently when I detect
//Microsoft's Visual C++.NET I always allow it to use references.
#if !defined(__DEVICE_EMULATION__) || (defined(_MSC_VER) && (_MSC_VER >= 1300))
#   define REF(x) &x
#   define ARRAYREF(x,y) (&x)[y]
#else
#   define REF(x) x
#   define ARRAYREF(x,y) x[y]
#endif

/**
 * A complex number type for use with CUDA, single precision accuracy.
 * This is deliberately designed to use few C++ features in order to work
 * with most CUDA SDK versions. It is friendlier to use than the cuComplex
 * type because it provides more operator overloads.
 * The class should work in host code and in device code and also in
 * emulation mode.
 * Also this has been tested on any OS that the CUDA SDK is available for.
 */

#define LOWER_BITMASK  0x000FFFFF
#define UPPER_BITMASK  0xFFF00000

#define pack_type_and_atomid(id)  ((tex1Dfetch(textype, (id))<<20)|(id))
#define unpack_typeid(id)         ((id)>>20)
#define unpack_atomid(id)         ((id)&LOWER_BITMASK)


__forceinline__ __device__ float2 lower_float2(float4 a);

//add float3 numbers..
__forceinline__ __device__ float3 operator+(const float3 REF(a),
                                            const float3 REF(b));

//subtract float3 numbers..
__forceinline__ __device__ float3 operator-(const float3 REF(a),
                                            const float3 REF(b));

//multiply by scalar..
__forceinline__ __device__ float3 operator*(const float3 REF(a),
                                            const float REF(b));

//divide float3 by scalar..
__forceinline__ __device__ float3 operator/(const float3 REF(a),
                                            const float REF(b));

//divide float3 by another float3 elementwise..
__forceinline__ __device__ float3 operator/(const float3 REF(a),
                                            const float3 REF(b));

//vector dot product of 2 float3 variables....
__forceinline__ __device__ float operator%(const float3 REF(a),
                                           const float3 REF(b));

//multiply 2 float3 numbers elementwise...
__forceinline__ __device__ float3 operator*(const float3 REF(a),
                                            const float3 REF(b));

//vector cross product of 2 float3 variables....
__forceinline__ __device__ float3 operator^(const float3 REF(a),
                                            const float3 REF(b));

__forceinline__ __host__ __device__ float sum(const float3 REF(a));

__forceinline__ __device__ void operator+=(float3 REF(a), const float3 REF(b));

__forceinline__ __device__ void operator-=(float3 REF(a), const float3 REF(b));

__forceinline__ __device__ int3 int4_to_int3(const int4 REF(a));

__forceinline__ __host__ __device__ void operator+=(int2 &a, int2 b);


//convert float4 to float3....
__forceinline__ __device__ float3 float4_to_float3(const float4 REF(a));

//pad float3 to float4....
__forceinline__ __device__ float4 float3_to_float4(const float3 REF(a));

//add 2 float4 variables..
__forceinline__ __device__ float4 operator+(const float4 REF(a),
                                            const float4 REF(b));

//multiply float4 by scalar..
__forceinline__ __device__ float4 operator*(const float4 REF(a),
                                            const float REF(b));

//divide float4 by scalar..
__forceinline__ __device__ float4 operator/(const float4 REF(a),
                                            const float REF(b));

//divide float4 by float4 elementwise..
__forceinline__ __device__ float4 operator/(const float4 REF(a),
                                            const float4 REF(b));

//subtract 2 float4 numbers..
__forceinline__ __device__ float4 operator-(const float4 REF(a),
                                            const float4 REF(b));

//multiply 2 float4 numbers elementwise...
__forceinline__ __device__ float4 operator*(const float4 REF(a),
                                            const float4 REF(b));

__forceinline__ __device__ float4 To_float4(const float3 REF(a),
                                            const float REF(b));

__forceinline__ __device__ float4 To_float4(const float2 REF(a),
                                            const float2 REF(b));

//add 2 float4 numbers and assign the result to the first
__forceinline__ __device__ void operator+=(float4 REF(a), const float4 REF(b));

//add float4 and float3 numbers and assign the result to the first
__forceinline__ __device__ void operator+=(float4 REF(a), const float3 REF(b));

__forceinline__ __device__ void operator-=(float4 REF(a), const float4 REF(b));

__forceinline__ __host__ __device__ float sum(const float4 REF(a));

/*
inline __device__ float4 make_float4(float &x, float &y, float &z, float &w){
	float4 result = {x, y, z, w};
	return result;
}
*/

// float3 to float4 assignment..
//__device__ void operator=(const float4 REF(a), const float3 REF(b)){
//	//the last component is unchanged in float3 to float4 assignment...
//	float4 result = { b.x, b.y, b.z, a.w};
//	return result;
//}


//a possible alternative to a cufftComplex constructor
HOSTDEVICE cufftComplex make_cufftComplex(float a, float b);

namespace constants{
	extern const cufftComplex zero;
	extern const cufftComplex one;
	extern const cufftComplex I;
};

//add complex numbers
HOSTDEVICE cufftComplex operator+(const cufftComplex REF(a),
                                  const cufftComplex REF(b));

//add scalar to complex
HOSTDEVICE cufftComplex operator+(const cufftComplex REF(a),
                                  const float REF(b));

//add complex to scalar
HOSTDEVICE cufftComplex operator+(const float REF(a),
                                  const cufftComplex REF(b));

//subtract complex numbers
HOSTDEVICE cufftComplex operator-(const cufftComplex REF(a),
                                  const cufftComplex REF(b));

//negate a complex number
HOSTDEVICE cufftComplex operator-(const cufftComplex REF(a));

//subtract scalar from complex
HOSTDEVICE cufftComplex operator-(const cufftComplex REF(a),
                                  const float REF(b));

//subtract complex from scalar
HOSTDEVICE cufftComplex operator-(const float REF(a),
                                  const cufftComplex REF(b));

//multiply complex numbers
HOSTDEVICE cufftComplex operator*(const cufftComplex REF(a),
                                  const cufftComplex REF(b));

//multiply complex with scalar
HOSTDEVICE cufftComplex operator*(const cufftComplex REF(a),
                                  const float REF(b));

//multiply scalar with complex
HOSTDEVICE cufftComplex operator*(const float REF(a),
                                  const cufftComplex REF(b));

//divide complex numbers
HOSTDEVICE cufftComplex operator/(const cufftComplex REF(a),
                                  const cufftComplex REF(b));

//divide complex by scalar
HOSTDEVICE cufftComplex operator/(const cufftComplex REF(a),
                                  const float REF(b));

//divide scalar by complex
HOSTDEVICE cufftComplex operator/(const float REF(a),
                                  const cufftComplex REF(b));

//complex conjugate
HOSTDEVICE cufftComplex operator~(const cufftComplex REF(a));

HOSTDEVICE cufftComplex exp(const cufftComplex REF(a));

#include "cucomplexops.cu"

#endif //_CUCOMPLEXOPS_H_
