/******************************************************************************/
//
// Md code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan
//            Joseph E. Davis
//            Michela Taufer
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/******************************************************************************/

#include "globals.h"
#include "cucomplexops.h"

#define USE_NEW_NONBOND
#ifdef USE_NEW_NONBOND
// -----------------------------------------------------------------------------
// New version of nonbondforces from mt - Use nbfix parameters if available
// Precompute extended eps and sigma - Save values in texture memory
//
//------------------------------------------------------------------------------
__global__ void nonbondforce(float4* f4d_nonbond
#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
                           , float4* f4d_nonbond0,
                             float4* f4d_nonbond1
#endif
#endif
                           , unsigned int* nblistd,
                             float* evdwd,
                             float* eelecd
#ifdef PCONSTANT
                           , float4* viriald
#endif
#ifdef DEBUG_NONB
                           , float* debug_nonbd
#endif
                                               ){
//------------------------------------------------------------------------------
//  Kernel function, calculates accelerations from nonbonded interactions
//  in parallel on GPU using nonbond list generated on GPU
//------------------------------------------------------------------------------

	unsigned int gtid = COMPUTE_GLOBAL_THREADID;
	int nnb = 0;
	unsigned int atomid = gtid % WorkgroupSized;

	int i;
	int j;
	int h;
	int t1;
	int t0 = tex1Dfetch(textype, atomid);

	//each thread will iterate through the neighbors for one atom
	float4 r1;
	float4 r;
	float4 r0 = tex1Dfetch(texcrd, atomid); //r4d[atomid];
	float2 parameter;

	int segval;
#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
	int segtypeid0;
	int segtypeid1;
	segtypeid0 = tex1Dfetch(texsegtype,atomid);
	float4 force_energy0 = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 force_energy1 = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
#endif
	float eCoul = 0.0f;
	float eVDW = 0.0f;
	float eps;
	float sigma2;
	//copy Switching function coefficients to register memory...
	float cutoffd_sqr = cutoffd * cutoffd;
	float distSqr;
	float4 force_energy = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 tempForce0; // ={0.0f, 0.0f, 0.0f, 0.0f};

	////////////pairInteraction variables///////////////////////
	float fC;
	float fLJ;
	float fr;
	float enerC;
	float enerLJ;

	float betadist;
	float erf_c;
	float d_erfcxr;
	float dist;
	float invDistSqr;
	float invDist;

	float swfunc;
	float sigma6xinvDist6;
	float diff_swfunc_div_r;
	float var1;
	float var2;
	///////////////////////////////////////////////////////////

#ifdef PCONSTANT
	float3 VirSum = {0.0f, 0.0f, 0.0f};
	float4 tempForce;
	float3 RBSRat = float4_to_float3(tex1Dfetch(texRBSRat, 0));
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	short int start = gtid / WorkgroupSized + 1;
	nnb = nblistd[atomid];

	for(i = start; i <= nnb; i += NBPART){
		//for(i=start; i<=nnb; i+=1) {
		//load the coordinates and parameters of current neighbor
		t1 = nblistd[i*WorkgroupSized + atomid];
		//tex1Dfetch(texnblist, i*WorkgroupSized + atomid);
		//nblistd[i*WorkgroupSized + atomid];
		j = unpack_atomid(t1);
		r1 = tex1Dfetch(texcrd, j);

	#ifdef USE_CONSFIX
	#ifdef DEBUG_CONSFIX
		//seg id for second atom
		segtypeid1 = tex1Dfetch(texsegtype, j);
		//segval = (segtypeid0 == segtypeid1)?0:1; // ! (segtypeid1 == segtypeid0);
	#endif
	#endif
		t1 = unpack_typeid(t1);

		r.x = __my_fadd(r0.x, -r1.x);
		r.y = __my_fadd(r0.y, -r1.y);
		r.z = __my_fadd(r0.z, -r1.z);

	//choose nearest image
		r.x = __my_add3(r.x,-__mycopysignf(boxH.x, __my_fadd(r.x, -boxH.x)),
		                -__mycopysignf(boxH.x, __my_fadd(r.x, boxH.x)));

		r.y = __my_add3(r.y,-__mycopysignf(boxH.y,__my_fadd(r.y,-boxH.y)),
		                -__mycopysignf(boxH.y,__my_fadd(r.y,boxH.y)));

		r.z = __my_add3(r.z,-__mycopysignf(boxH.z,__my_fadd(r.z,-boxH.z)),
		                -__mycopysignf(boxH.z,__my_fadd(r.z,boxH.z)));

		r.w = __my_fmul(r0.w, r1.w);

		h = t0 * num_types_presentd + t1;
		parameter = tex1Dfetch(texnbfixprm, h);

		distSqr = __my_add3(__my_fmul(r.x, r.x),
		                    __my_fmul(r.y, r.y),
		                    __my_fmul(r.z, r.z));

		segval = (distSqr < cutoffd_sqr);

		//compute forces only if within cutoff...
		//if ((distSqr = __my_add3(__my_fmul(r.x, r.x),
		//__my_fmul(r.y, r.y),
		//__my_fmul(r.z, r.z)))<cutoffd_sqr){
		// calculate product of charges (4th coord.)
		// distSqr = __my_add3(__my_fmul(r.x, r.x),
		//__my_fmul(r.y, r.y), __my_fmul(r.z, r.z));

		//calculate epsilon and sigma2
		eps = parameter.x; //__my_fmul(4.0f, parameter.x);
		sigma2 = parameter.y; //__my_fmul(parameter.y,parameter.y);

		//distSqr = __my_add3(__my_fmul(r.x, r.x),
		//__my_fmul(r.y, r.y), __my_fmul(r.z, r.z));
		dist = sqrt(distSqr);

		//PAIR_INTERACTION(r)
		invDist = __my_fdiv(1.0, dist);

		invDistSqr = __my_fdiv(1.0, distSqr);

		betadist = __my_fmul(beta, dist);

		erf_c = erfc(betadist);

		//__my_mul3(CC*TWO_OVER_SQRT_PI*r.w, beta, exp(-betadist*betadist));

		d_erfcxr = CCTWO_OVER_SQRT_PI * beta * r.w * exp(-betadist * betadist);

		enerC = CC * r.w * invDist * erf_c;

		fC = (enerC + d_erfcxr) * invDistSqr;

		swfunc = 1.0f;

		sigma6xinvDist6 = __my_fmul(__my_mul3(sigma2, sigma2, sigma2),
		                            __my_mul3(invDistSqr, invDistSqr, invDistSqr));

		var1 = eps * sigma6xinvDist6;
		var2 = var1 * sigma6xinvDist6;

		enerLJ = (var2 - var1);

		fLJ = (12.0f * var2 - 6.0f * var1) * invDistSqr;

		diff_swfunc_div_r = 0.0f;

		var1 = (Swcoeff1d - distSqr);
		var2 = (Swcoeff2d + 2.0f * distSqr);

		if(dist > Cutond){
			swfunc  = __my_mul4(var1, var1, var2, Swcoeff3d);

			diff_swfunc_div_r = __my_mul3(var1,
			                              Swcoeff3d,
			                              __my_fadd(4.0f * var2,
			                                        -var1 * (Swcoeff2d *
			                                          invDist + 4.0f)));
		}

		fLJ = __my_fadd(__my_fmul(fLJ, swfunc),
		                __my_fmul(enerLJ, diff_swfunc_div_r));

		enerLJ = __my_fmul(enerLJ, swfunc);

		fr = __my_fadd(fLJ, fC) * segval;

		eCoul = __my_fadd(eCoul, enerC * segval);

		eVDW = __my_fadd(eVDW, enerLJ * segval);

		tempForce0 = make_float4(__my_fmul(fr, r.x),
		                         __my_fmul(fr, r.y),
		                         __my_fmul(fr, r.z),
		                         0.5f * __my_fadd(enerC, enerLJ));

		force_energy += tempForce0;

#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
		force_energy0 += tempForce0 * ((segtypeid0 == -10) * (segtypeid1 == -20));
		force_energy1 += tempForce0 * ((segtypeid0 == -20) * (segtypeid1 == -10));
#endif
#endif

		//////////////////////////////////////////////////////
		// }//if ((distSqr = r.x*r.x + r.y*r.y + r.z*r.z)<cutoffd_sqr)

	}//for(i=1; i<=nnb; i++)

	//Save the result in global memory
	evdwd[gtid] += 0.5f * eVDW;
	eelecd[gtid] += 0.5f * eCoul;
	f4d_nonbond[gtid] = force_energy;
#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
	f4d_nonbond0[gtid] = force_energy0;
	f4d_nonbond1[gtid] = force_energy1;
#endif
#endif

#ifdef PCONSTANT
	viriald[gtid] += float3_to_float4(VirSum * 0.5f);
#endif

#ifdef DEBUG_NONB
	debug_nonbd[NUM_DEBUG_NONB * gtid + 0] = debug_var1;
	debug_nonbd[NUM_DEBUG_NONB * gtid + 1] = debug_var2;
	debug_nonbd[NUM_DEBUG_NONB * gtid + 2] = debug_var3;
	//VDWe1d[(int) norm_tex_addr]; //
	debug_nonbd[NUM_DEBUG_NONB * gtid + 3] = 0.0f;//force_energy.w;
#endif

}

#else

// -----------------------------------------------------------------------------
// Old nonbond forces of ng - this version computes eps and sigma at runtime
// Use of shared memory is activated with this definition
// #define USE_SHMEM

//------------------------------------------------------------------------------
__global__ void nonbondforce(float4* f4d_nonbond
#ifdef USE_CONSFIX
                           , float4* f4d_nonbond0,
                             float4* f4d_nonbond1
#endif
                           , unsigned int* nblistd,
                             float* evdwd,
                             float* eelecd
#ifdef PCONSTANT
                           , float4* viriald
#endif
#ifdef DEBUG_NONB
                           , float* debug_nonbd
#endif
                                               ){
//------------------------------------------------------------------------------
//  Kernel function, calculates accelerations from nonbonded interactions
//  in parallel on GPU using nonbond list generated on GPU
//------------------------------------------------------------------------------

	unsigned int gtid = COMPUTE_GLOBAL_THREADID;
	int nnb = 0;
	unsigned int atomid = gtid % WorkgroupSized;

	int i;
	int j;
	int t1;
	int t0 = tex1Dfetch(textype, atomid);

	//each thread will iterate through the neighbors for one atom
	float4 r0 = tex1Dfetch(texcrd, atomid); //r4d[atomid];

	float4 r1;
	float4 r;
	float eps;
	float sigma2;
	float eCoul = 0.0f;
	float eVDW = 0.0f;

	float2 prm1;
	float2 prm0 = lower_float2(tex1Dfetch(texprm, t0));

	//copy Switching function coefficients to register memory...
	//float3 Swcoeff = {Swcoeff1d, Swcoeff2d, Swcoeff3d};
	//float Cuton = Cutond;

	float cutoffd_sqr = cutoffd * cutoffd;
	float distSqr;
	float4 force_energy={0.0f, 0.0f, 0.0f, 0.0f};

	////////////pairInteraction variables///////////////////////
	float fC;
	float fLJ;
	float fr;
	float enerC;
	float enerLJ;

	float betadist;
	float erf_c;
	float d_erfcxr;
	float dist;
	float invDistSqr;
	float invDist;

	float swfunc;
	float sigma6xinvDist6;
	float diff_swfunc_div_r;
	float var1;
	float var2;
	///////////////////////////////////////////////////////////

#ifdef PCONSTANT
	float3 VirSum = {0.0f, 0.0f, 0.0f};
	float4 tempForce = {0.0f, 0.0f, 0.0f};
	float3 RBSRat = float4_to_float3(tex1Dfetch(texRBSRat, 0));
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

#ifdef USE_SHMEM
	//copy nonbond parameters to shared memory...
	__shared__ float2 neighbor_prm[MAXTYP];

	for(i = threadIdx.x; i < MAXTYP; i += blockDim.x){
		neighbor_prm[i] = lower_float2(tex1Dfetch(texprm, i));
		// prm1 = tex1Dfetch(texprm, i);
		// neighbor_prm[i].x = prm1.y;
		// neighbor_prm[i].y = prm1.z;
	}
	__syncthreads();
#else

#endif
	short int start = gtid / WorkgroupSized + 1;

	nnb = nblistd[atomid];
	for(i = start; i <= nnb; i += NBPART){
		//for(i=start; i<=nnb; i+=1) {

		//load the coordinates and parameters of current neighbor
		t1 = nblistd[i * WorkgroupSized + atomid];

		//tex1Dfetch(texnblist, i*WorkgroupSized + atomid);
		//nblistd[i*WorkgroupSized + atomid];

		j = unpack_atomid(t1);

		r1 = tex1Dfetch(texcrd, j);

		t1 = unpack_typeid(t1);
		// read from shared memory...

#ifdef USE_SHMEM
		//copy nonbond parameters ...
		prm1 = neighbor_prm[t1];
#else
		prm1 = lower_float2(tex1Dfetch(texprm, t1));
#endif
		r.x = __my_fadd(r0.x, -r1.x);
		r.y = __my_fadd(r0.y, -r1.y);
		r.z = __my_fadd(r0.z, -r1.z);

		// choose nearest image
		r.x = __my_add3(r.x,
		                -__mycopysignf(boxH.x,__my_fadd(r.x,-boxH.x)),
		                -__mycopysignf(boxH.x,__my_fadd(r.x,boxH.x)));

		r.y = __my_add3(r.y,
		                -__mycopysignf(boxH.y,__my_fadd(r.y,-boxH.y)),
		                -__mycopysignf(boxH.y,__my_fadd(r.y,boxH.y)));

		r.z = __my_add3(r.z,
		                -__mycopysignf(boxH.z,__my_fadd(r.z,-boxH.z)),
		                -__mycopysignf(boxH.z,__my_fadd(r.z,boxH.z)));

		distSqr = __my_add3(__my_fmul(r.x, r.x),
		                    __my_fmul(r.y, r.y),
		                    __my_fmul(r.z, r.z));

		int setval = (distSqr < cutoffd_sqr);

		//compute forces only if within cutoff...
		//if ((distSqr = __my_add3(__my_fmul(r.x, r.x),
		//__my_fmul(r.y, r.y), __my_fmul(r.z, r.z)))<cutoffd_sqr){
		//calculate product of charges (4th coord.)
		//distSqr = __my_add3(__my_fmul(r.x, r.x),
		//__my_fmul(r.y, r.y), __my_fmul(r.z, r.z));

		r.w = __my_fmul(r0.w, r1.w);

		//calculate epsilon and sigma2
		eps = __my_fmul(4.0f, sqrtf(__my_fmul(prm0.x, prm1.x)));

		sigma2 = __my_fmul(__my_fadd(prm0.y, prm1.y), __my_fadd(prm0.y, prm1.y));

		//distSqr = __my_add3(__my_fmul(r.x, r.x),
		//__my_fmul(r.y, r.y), __my_fmul(r.z, r.z));
		dist = sqrt(distSqr);

		// PAIR_INTERACTION(r)
		invDist = __my_fdiv(1.0, dist);

		invDistSqr = __my_fdiv(1.0, distSqr);

		betadist = __my_fmul(beta,dist);

		erf_c = erfc(betadist);

		d_erfcxr = __my_mul3(CCTWO_OVER_SQRT_PI * r.w,
		                     beta,
		                     exp(-betadist * betadist));

		enerC = CC * r.w * invDist * erf_c;

		fC = (enerC + d_erfcxr) * invDistSqr;

		swfunc = 1.0f;

		sigma6xinvDist6 = __my_fmul(__my_mul3(sigma2, sigma2, sigma2),
		                            __my_mul3(invDistSqr, invDistSqr, invDistSqr));

		var1 = eps * sigma6xinvDist6;
		var2 = var1 * sigma6xinvDist6;

		enerLJ = (var2 - var1);

		fLJ = (12.0f * var2 - 6.0f * var1) * invDistSqr;

		diff_swfunc_div_r = 0.0f;

		var1 = (Swcoeff1d - distSqr);
		var2 = (Swcoeff2d + 2.0f * distSqr);

		if(dist > Cutond){
			swfunc  = __my_mul4(var1, var1, var2, Swcoeff3d);

			diff_swfunc_div_r = __my_mul3(var1,
			                              Swcoeff3d,
			                              __my_fadd(4.0f * var2,
			                                        -var1 * (Swcoeff2d *
			                                          invDist + 4.0f)));
		}

		fLJ = __my_fadd(__my_fmul(fLJ, swfunc),
		                __my_fmul(enerLJ, diff_swfunc_div_r));

		enerLJ = __my_fmul(enerLJ, swfunc);

		fr = __my_fadd(fLJ, fC) * setval;

		eCoul = __my_fadd(eCoul, enerC * setval);

		eVDW = __my_fadd(eVDW, enerLJ * setval);

#ifdef PCONSTANT
		tempForce = make_float4(__my_fmul(fr, r.x),
		                        __my_fmul(fr, r.y),
		                        __my_fmul(fr, r.z),
		                        0.5f * __my_fadd(enerC, enerLJ));

		force_energy += tempForce;

		VirSum += float4_to_float3(tempForce * r);

		//K-space Virial..
		VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * r.w *
		                    exp(-betaSqr * (r.x * r.x + r.y *
		                                    r.y + r.z * r.z)));
#else
		force_energy += make_float4(__my_fmul(fr, r.x),
		                            __my_fmul(fr, r.y),
		                            __my_fmul(fr, r.z),
		                            0.5f * __my_fadd(enerC, enerLJ));
#endif

		//////////////////////////////////////////////////////
		// }//if ((distSqr = r.x*r.x + r.y*r.y + r.z*r.z)<cutoffd_sqr)

	}//for(i=1; i<=nnb; i++)

	//Save the result in global memory
	evdwd[gtid] += 0.5f * eVDW;
	eelecd[gtid] += 0.5f * eCoul;
	f4d_nonbond[gtid] = force_energy;

#ifdef PCONSTANT
	viriald[gtid] += float3_to_float4(VirSum * 0.5f);
#endif

#ifdef DEBUG_NONB
	debug_nonbd[NUM_DEBUG_NONB * gtid + 0] = debug_var1;
	debug_nonbd[NUM_DEBUG_NONB * gtid + 1] = debug_var2;
	debug_nonbd[NUM_DEBUG_NONB * gtid + 2] = debug_var3;
	//VDWe1d[(int) norm_tex_addr]; //
	debug_nonbd[NUM_DEBUG_NONB * gtid + 3] = 0.0f;//force_energy.w;
#endif
}

#endif
