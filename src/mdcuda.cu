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

#include "globals.h"
#include "cucomplexops.h"
#include "timer.h"

//==============================================================================
#include "bonded_forces.h"
#include "nonbonded_forces.h"
#include "cell_based_kernels.h"
#include "pme_kernels.h"
//==============================================================================

/*----------------------------------------------------------------------------*/
__global__ void HalfKickGPU(float4* v4d,
                            float4* f4d,
                            float4* f4d_nonbond,
                            float4* f4d_bonded
#ifdef PCONSTANT
                          , float4* boxVelocd,
                            float4* boxAcceld
#endif
                                             ){
/*------------------------------------------------------------------------------
//Accelerates atomic velocities by half the time step on the gpu
------------------------------------------------------------------------------*/
	// update each atomic velocities...
	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	float4 v = v4d[gtid];
	float4 r = tex1Dfetch(texcrd, gtid);

	float mymass = (gtid < natomd) ? tex1Dfetch(texprm,
	                                            tex1Dfetch(textype, gtid)).x :
	                                 1.0f;

	//add the nonbond forces and energies...
	float4 acc = ((gtid < natomd) ? ((f4d[gtid] +
														 f4d_nonbond[gtid] +
	                           f4d_nonbond[WorkgroupSized + gtid] +
	                           f4d_bonded[gtid] +
	                           f4d_bonded[WorkgroupSized + gtid])) :
	                         make_float4(0.0f, 0.0f, 0.0f, 0.0f)) / mymass;
	//force/mass

#ifdef PCONSTANT
	float4 box = tex1Dfetch(texbox, 0);

	float4 boxAccel = boxAcceld[0];
	acc.x += r.x * (__my_fdiv(boxAccel.x, box.x));
	acc.y += r.y * (__my_fdiv(boxAccel.y, box.y));
	acc.z += r.z * (__my_fdiv(boxAccel.z, box.z));
#endif

	//v.x += TU*0.5*deltaTd*a.x;
	//v.y += TU*0.5*deltaTd*a.y;
	//v.z += TU*0.5*deltaTd*a.z;
	v.x = __my_fadd(v.x,
	                __my_fmul(TU,__my_fmul(0.5f,__my_fmul(deltaTd, acc.x)))) *
	      v.w;

	v.y = __my_fadd(v.y,
	                __my_fmul(TU,__my_fmul(0.5f,__my_fmul(deltaTd, acc.y)))) *
	      v.w;

	v.z = __my_fadd(v.z,__my_fmul(TU,
	                __my_fmul(0.5f,__my_fmul(deltaTd, acc.z)))) *
	      v.w;

	v4d[gtid] = v;

#ifdef PCONSTANT
	if(gtid == 0){
		boxVelocd[0].x = boxVelocd[0].x + boxAcceld[0].x * (0.5f * TU * deltaTd);
		boxVelocd[0].y = boxVelocd[0].y + boxAcceld[0].y * (0.5f * TU * deltaTd);
		boxVelocd[0].z = boxVelocd[0].z + boxAcceld[0].z * (0.5f * TU * deltaTd);
	}
#endif

}

//------------------------------------------------------------------------------
__device__ float4 compute_nearest_image(float4 r0, float4 r1){
//------------------------------------------------------------------------------
//Computes nearest image between two vectors...
//------------------------------------------------------------------------------

	float4 r;
#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
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

	return r;
}

//------------------------------------------------------------------------------
__global__ void ConjugatedGradient(float4* r4d,
                                   float4* roldd,
                                   int* nupdated,
                                   float4* v4d,
                                   float4* f4d,
                                   float4* f4d_nonbond,
                                   float4* f4d_bonded,
                                   float4* prev_f4d,
                                   float3* Htd,
                                   float* cgfacd
#ifdef PME_CALC
                                 , float4* r4d_scaled,
                                   float4* prev_r4d_scaled,
                                   int4* disp
#endif
                                             ){
//------------------------------------------------------------------------------
//Move the atoms along the path of steepest energy descent
//periodic boundary conditions
//------------------------------------------------------------------------------

	unsigned int gtid = COMPUTE_GLOBAL_THREADID
	float4 prm0 = tex1Dfetch(texprm, tex1Dfetch(textype, gtid));
	float4 dr;
	float4 r = r4d[gtid];
	float4 force = f4d[gtid] +
	               f4d_nonbond[gtid] +
	               f4d_nonbond[WorkgroupSized + gtid] +
	               f4d_bonded[gtid] +
	               f4d_bonded[WorkgroupSized + gtid];

#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	float3 Ft = float4_to_float3(force);
	float3 Ft1 = float4_to_float3(prev_f4d[gtid]);
	Ft1 = ((Ft1 % Ft1) < 1e-8) ? Ft : Ft1;

	float gamma = (Ft % Ft) / (Ft1 % Ft1);
	//float gamma = 0.0f;

	float3 Ht = (Ft + Htd[gtid] * gamma) * (*cgfacd);

	r.x += Ht.x;
	r.y += Ht.y;
	r.z += Ht.z;

	//apply periodic boundary conditions
	//r.x = r.x - __mycopysignf(boxH.x, r.x) -
	//__mycopysignf(boxH.x, r.x-2*boxH.x);
	//r.y = r.y - __mycopysignf(boxH.y, r.y) -
	//__mycopysignf(boxH.y, r.y-2*boxH.y);
	//r.z = r.z - __mycopysignf(boxH.z, r.z) -
	//__mycopysignf(boxH.z, r.z-2*boxH.z);
	r.x = __my_add3(r.x,
	                -__mycopysignf(boxH.x, r.x),
	                -__mycopysignf(boxH.x,
	                               __my_fadd(r.x, -__my_fmul(2.0f, boxH.x))));

	r.y = __my_add3(r.y,
	                -__mycopysignf(boxH.y, r.y),
	                -__mycopysignf(boxH.y,
	                               __my_fadd(r.y, -__my_fmul(2.0f, boxH.y))));

	r.z = __my_add3(r.z,
	                -__mycopysignf(boxH.z, r.z),
	                -__mycopysignf(boxH.z,
	                               __my_fadd(r.z, -__my_fmul(2.0f, boxH.z))));

	//check for the need for nonbond list update
	dr = r - roldd[gtid];

	dr.x = __my_fadd(dr.x,
	                 -__my_fadd(__mycopysignf(boxH.x, __my_fadd(dr.x, -boxH.x)),
	                            __mycopysignf(boxH.x, __my_fadd(dr.x, boxH.x))));

	dr.y = __my_fadd(dr.y,
	                 -__my_fadd(__mycopysignf(boxH.y, __my_fadd(dr.y, -boxH.y)),
	                            __mycopysignf(boxH.y, __my_fadd(dr.y, boxH.y))));

	dr.z = __my_fadd(dr.z,
	                 -__my_fadd(__mycopysignf(boxH.z, __my_fadd(dr.z, -boxH.z)),
	                            __mycopysignf(boxH.z, __my_fadd(dr.z, boxH.z))));

	dr.w = __my_fadd(__my_fmul(dr.x, dr.x),
	                 __my_fadd(__my_fmul(dr.y, dr.y), __my_fmul(dr.z, dr.z)));

	// check if atom has moved more than CutCheck, if so set flag
	if((dr.w >= __my_fmul(cutcheckd, cutcheckd)) && (gtid < natomd)){
		*nupdated = 1;
	}

	r4d[gtid] = r;
	Htd[gtid] = Ht;
	prev_f4d[gtid] = force;

	f4d[gtid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	v4d[gtid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef PME_CALC
#ifdef PCONSTANT
	float4 frac = tex1Dfetch(texfrac, 0);
#endif

	r4d_scaled[gtid].x = __my_fmul(r.x, frac.x);
	r4d_scaled[gtid].y = __my_fmul(r.y, frac.y);
	r4d_scaled[gtid].z = __my_fmul(r.z, frac.z);
	r4d_scaled[gtid].w = r.w;

	//calculate the integer displacement array...
	disp[gtid].x = floor(r4d_scaled[gtid].x) - floor(prev_r4d_scaled[gtid].x);
	disp[gtid].y = floor(r4d_scaled[gtid].y) - floor(prev_r4d_scaled[gtid].y);
	disp[gtid].z = floor(r4d_scaled[gtid].z) - floor(prev_r4d_scaled[gtid].z);
#endif

}

//------------------------------------------------------------------------------
__global__ void SteepestDescent(float4* r4d,
                                float4* roldd,
                                int* nupdated,
                                float4* v4d,
                                float4* f4d,
                                float4* f4d_nonbond,
                                float4* f4d_bonded,
                                float* sdfacd
#ifdef PME_CALC
                              , float4* r4d_scaled,
                                float4* prev_r4d_scaled,
                                int4* disp
#endif
                                          ){
//------------------------------------------------------------------------------
//Move the atoms along the path of steepest energy descent
//periodic boundary conditions
//------------------------------------------------------------------------------

	unsigned int gtid = COMPUTE_GLOBAL_THREADID
	float4 prm0 = tex1Dfetch(texprm, tex1Dfetch(textype, gtid));
	float4 dr;
	float4 r = r4d[gtid];
#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	float4 force = f4d[gtid] +
	               f4d_nonbond[gtid] +
	               f4d_nonbond[WorkgroupSized + gtid] +
	               f4d_bonded[gtid] +
	               f4d_bonded[WorkgroupSized + gtid];

	float3 Ft = (float4_to_float3(force)) * (*sdfacd);

	r.x += Ft.x;
	r.y += Ft.y;
	r.z += Ft.z;

	// apply periodic boundary conditions
	//r.x = r.x - __mycopysignf(boxH.x, r.x) -
	//__mycopysignf(boxH.x, r.x-2*boxH.x);
	//r.y = r.y - __mycopysignf(boxH.y, r.y) -
	//__mycopysignf(boxH.y, r.y-2*boxH.y);
	//r.z = r.z - __mycopysignf(boxH.z, r.z) -
	//__mycopysignf(boxH.z, r.z-2*boxH.z);

	r.x = __my_add3(r.x,
	                -__mycopysignf(boxH.x, r.x),
	                -__mycopysignf(boxH.x,
	                               __my_fadd(r.x, -__my_fmul(2.0f, boxH.x))));

	r.y = __my_add3(r.y,
	                -__mycopysignf(boxH.y, r.y),
	                -__mycopysignf(boxH.y,
	                               __my_fadd(r.y, -__my_fmul(2.0f, boxH.y))));

	r.z = __my_add3(r.z,
	                -__mycopysignf(boxH.z, r.z),
	                -__mycopysignf(boxH.z,
	                               __my_fadd(r.z, -__my_fmul(2.0f, boxH.z))));

	//check for the need for nonbond list update
	dr = r - roldd[gtid];

	dr.x = __my_fadd(dr.x,
	                 -__my_fadd(__mycopysignf(boxH.x, __my_fadd(dr.x, -boxH.x)),
	                 __mycopysignf(boxH.x, __my_fadd(dr.x, boxH.x))));

	dr.y = __my_fadd(dr.y,
	                 -__my_fadd(__mycopysignf(boxH.y, __my_fadd(dr.y, -boxH.y)),
	                 __mycopysignf(boxH.y, __my_fadd(dr.y, boxH.y))));

	dr.z = __my_fadd(dr.z,
	                 -__my_fadd(__mycopysignf(boxH.z, __my_fadd(dr.z, -boxH.z)),
	                 __mycopysignf(boxH.z, __my_fadd(dr.z, boxH.z))));

	dr.w = __my_fadd(__my_fmul(dr.x, dr.x),
	                 __my_fadd(__my_fmul(dr.y, dr.y), __my_fmul(dr.z, dr.z)));

	// check if atom has moved more than CutCheck, if so set flag
	if((dr.w >= __my_fmul(cutcheckd, cutcheckd)) && (gtid < natomd)){
		*nupdated = 1;
	}

	r4d[gtid] = r;

	f4d[gtid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	v4d[gtid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef PME_CALC
#ifdef PCONSTANT
	float4 frac = tex1Dfetch(texfrac, 0);
#endif

	r4d_scaled[gtid].x = __my_fmul(r.x, frac.x);
	r4d_scaled[gtid].y = __my_fmul(r.y, frac.y);
	r4d_scaled[gtid].z = __my_fmul(r.z, frac.z);
	r4d_scaled[gtid].w = r.w;

	//calculate the integer displacement array...
	disp[gtid].x = floor(r4d_scaled[gtid].x) - floor(prev_r4d_scaled[gtid].x);
	disp[gtid].y = floor(r4d_scaled[gtid].y) - floor(prev_r4d_scaled[gtid].y);
	disp[gtid].z = floor(r4d_scaled[gtid].z) - floor(prev_r4d_scaled[gtid].z);
#endif

}

//------------------------------------------------------------------------------
__global__ void UpdateCoords(float4* r4d,
                             float4* v4d
#ifdef PCONSTANT
                           , float4* viriald
#endif
                                            ){
//------------------------------------------------------------------------------
	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	r4d[gtid].x = __my_fadd(r4d[gtid].x,
	                        __my_fmul(__my_fmul(TU, deltaTd), v4d[gtid].x));

	r4d[gtid].y = __my_fadd(r4d[gtid].y,
	                        __my_fmul(__my_fmul(TU, deltaTd), v4d[gtid].y));

	r4d[gtid].z = __my_fadd(r4d[gtid].z,
	                        __my_fmul(__my_fmul(TU, deltaTd), v4d[gtid].z));

#ifdef PCONSTANT
	viriald[gtid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif
}

//------------------------------------------------------------------------------
__global__ void CoordsUpdate(float4* r4d,
                             float4* roldd,
                             int* nupdated
//, float4 *f4d
#ifdef PME_CALC
                           , float4* r4d_scaled,
                             float4* prev_r4d_scaled,
                             int4* disp
#endif
#ifdef PCONSTANT
                           , float4* boxLengthd,
                             float4* boxHd,
                             float4* boxVelocd,
                             float4* boxAcceld,
                             int4* numcellsd,
                             //float4 *kineticd, float4 *viriald,
                             float4* ReciprocalBoxSquareRatiod
#ifdef PME_CALC
                           , float4* fracd
#endif
#endif
                                          ){
//------------------------------------------------------------------------------
//Update atomic coordinates to r(t+Dt) on the gpu, also apply
//periodic boundary conditions
//------------------------------------------------------------------------------

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	float4 r = r4d[gtid];
	//float4 v = v4d[gtid];
	float3 dr;

	// apply periodic boundary conditions
	// r.x = r.x - __mycopysignf(boxH.x, r.x-boxH.x) -
	//__mycopysignf(boxH.x, r.x+boxH.x);
	// r.y = r.y - __mycopysignf(boxH.y, r.y-boxH.y) -
	//__mycopysignf(boxH.y, r.y+boxH.y);
	// r.z = r.z - __mycopysignf(boxH.z, r.z-boxH.z) -
	//__mycopysignf(boxH.z, r.z+boxH.z);

	r.x = __my_add3(r.x,
	                -__mycopysignf(boxH.x, r.x),
	                -__mycopysignf(boxH.x,
	                               __my_fadd(r.x, -__my_fmul(2.0f, boxH.x))));

	r.y = __my_add3(r.y,
	                -__mycopysignf(boxH.y, r.y),
	                -__mycopysignf(boxH.y,
	                               __my_fadd(r.y, -__my_fmul(2.0f, boxH.y))));

	r.z = __my_add3(r.z,
	                -__mycopysignf(boxH.z, r.z),
	                -__mycopysignf(boxH.z,
	                               __my_fadd(r.z, -__my_fmul(2.0f, boxH.z))));

	r.x = nearest_image(r.x, boxH.x);
	r.y = nearest_image(r.y, boxH.y);
	r.z = nearest_image(r.z, boxH.z);

	//check for the need for nonbond list update
	dr = float4_to_float3(r - roldd[gtid]);
	// nearest image...
	dr.x = __my_fadd(dr.x,
	                 -__my_fadd(__mycopysignf(boxH.x, __my_fadd(dr.x, -boxH.x)),
	                            __mycopysignf(boxH.x, __my_fadd(dr.x, boxH.x))));

	dr.y = __my_fadd(dr.y,
	                 -__my_fadd(__mycopysignf(boxH.y, __my_fadd(dr.y, -boxH.y)),
	                            __mycopysignf(boxH.y, __my_fadd(dr.y, boxH.y))));

	dr.z = __my_fadd(dr.z,
	                 -__my_fadd(__mycopysignf(boxH.z, __my_fadd(dr.z, -boxH.z)),
	                            __mycopysignf(boxH.z, __my_fadd(dr.z, boxH.z))));
															
	// check if atom has moved more than CutCheck, if so set flag
	if(((dr.x * dr.x + dr.y * dr.y + dr.z * dr.z) >= cutcheckd * cutcheckd) &&
	   (gtid < natomd)){

		*nupdated = 1;
	}

	///End Update...

	r4d[gtid] = r;

	// f4d[gtid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef PME_CALC
	r4d_scaled[gtid].x = __my_fmul(r.x, frac.x);
	r4d_scaled[gtid].y = __my_fmul(r.y, frac.y);
	r4d_scaled[gtid].z = __my_fmul(r.z, frac.z);
	r4d_scaled[gtid].w = r.w;

	//calculate the integer displacement array...
	disp[gtid].x = floor(r4d_scaled[gtid].x) - floor(prev_r4d_scaled[gtid].x);
	disp[gtid].y = floor(r4d_scaled[gtid].y) - floor(prev_r4d_scaled[gtid].y);
	disp[gtid].z = floor(r4d_scaled[gtid].z) - floor(prev_r4d_scaled[gtid].z);
#endif

}

//------------------------------------------------------------------------------
__global__ void restraint_force(float3* com0d,
                                float3* com1d,
                                char* segidd,
                                float4* f4d,
                                float4* r4d){
//------------------------------------------------------------------------------
// This kernel computes the restraint interaction..

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	float3 dr = consharmdistd + (com0d[0] - com1d[0]);

#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	if((segidd[gtid] == 1) && (gtid < natomd)){

		//nearest image...
		dr.x = __my_add3(dr.x,
		                 -__mycopysignf(boxH.x, dr.x - boxH.x),
		                 -__mycopysignf(boxH.x, dr.x + boxH.x));

		dr.y = __my_add3(dr.y,
		                 -__mycopysignf(boxH.y, dr.y - boxH.y),
		                 -__mycopysignf(boxH.y, dr.y + boxH.y));

		dr.z = __my_add3(dr.z,
		                 -__mycopysignf(boxH.z, dr.z - boxH.z),
		                 -__mycopysignf(boxH.z, dr.z + boxH.z));

		f4d[gtid] += make_float4(dr.x, dr.y, dr.z, 0.0f) * consharmfcd;
		//r4d[gtid] += make_float4(0.0f, 0.0f, .01f, 0.0f);
	}

}

#ifdef PME_CALC
//------------------------------------------------------------------------------
__device__ float4 ewaldCorrection(float4 r, float& eEwexcl){
//------------------------------------------------------------------------------
//Calculate the Ewald Correction term for the exclusion atoms..
//(bonds and 1-3 angles)
//------------------------------------------------------------------------------
	float errf;
	float betadist;
	float d_erfxr;
	float dist;
	float fC;
	float egy = 0.0f;

	float4 force_energy = {0.0f, 0.0f, 0.0f, 0.0f};

	dist = sqrt(__my_add3(__my_fmul(r.x, r.x),
	            __my_fmul(r.y, r.y),
	            __my_fmul(r.z, r.z)));

	betadist = __my_fmul(beta, dist);
	errf = erff(betadist);
	d_erfxr = betadist * TWO_OVER_SQRT_PI * expf(-betadist * betadist);
	egy = __my_mul3(-0.5f * CC, __my_fdiv(r.w, dist), errf);
	fC = __my_mul3(CC,
	               __my_fdiv(r.w, __my_mul3(dist, dist, dist)),
	               __my_fadd(d_erfxr, -errf));

	eEwexcl += egy;
	force_energy.x = __my_fmul(fC, r.x);
	force_energy.y = __my_fmul(fC, r.y);
	force_energy.z = __my_fmul(fC, r.z);
	force_energy.w = egy;

	return force_energy;
}
#endif //PME_CALC

//------------------------------------------------------------------------------
void ComputeAccelGPU(){
//------------------------------------------------------------------------------

#ifdef PROFILING
	cpu1 = clock();//tic();//
#endif
	if(c36 == 0){

		bondedforce<<<BondeddimGrid, BondeddimBlock>>>(f4d_bonded,
		                                               //bond Data
		                                               bonds_indexd,
		                                               bondprmd,
		                                               ebndd,
		                                               //angle Data
		                                               angles_indexd,
		                                               angles_setd,
		                                               angleprmd,
		                                               eangd
#ifdef UREY_BRADLEY
		                                             , ureyb_indexd,
		                                               ureybprmd,
		                                               eureybd
#endif
		                                               // dihedral Data
		                                             , dihedrals_indexd,
		                                               dihedrals_setd,
		                                               dihedral_prmd,
		                                               dihedral_type_countd,
		                                               edihedd,
		                                               evdwd,
		                                               eelecd
#ifdef IMPROPER
		                                             , impropers_indexd,
		                                               impropers_setd,
		                                               improper_prmd,
		                                               eimpropd
#endif
#ifdef PME_CALC
		                                             , ewlistd,
		                                               eEwexcld
#endif
#ifdef PCONSTANT
		                                             , viriald
#endif
		                                               //debug Data
#ifdef BOND_DEBUG
		                                             , bonds_debugd
#endif
#ifdef ANGLE_DEBUG
		                                             , angles_debugd
#endif
#ifdef DIHED_DEBUG
		                                             , dihedrals_debugd
#endif
		                                                               );

	}
	else{
		bondedforce_c36<<<BondeddimGrid, BondeddimBlock>>>(f4d_bonded,
		                                                   //bond Data
		                                                   bonds_indexd,
		                                                   bondprmd,
		                                                   ebndd,
		                                                   //angle Data
		                                                   angles_indexd,
		                                                   angles_setd,
		                                                   angleprmd,
		                                                   eangd
#ifdef UREY_BRADLEY
		                                                 , ureyb_indexd,
		                                                   ureybprmd,
		                                                   eureyb
#endif
		                                                   //dihedral Data
		                                                 , dihedrals_indexd,
		                                                   dihedrals_setd,
		                                                   dihedral_prmd,
		                                                   dihedral_type_countd,
		                                                   edihedd,
		                                                   evdwd,
		                                                   eelecd
#ifdef IMPROPER
		                                                 , impropers_indexd,
		                                                   impropers_setd,
		                                                   improper_prmd,
		                                                   eimpropd
#endif
#ifdef PME_CALC
		                                                 , ewlistd,
		                                                   eEwexcld
#endif
#ifdef PCONSTANT
		                                                 , viriald
#endif
		                                                   //debug Data
#ifdef BOND_DEBUG
		                                                 , bonds_debugd
#endif
#ifdef ANGLE_DEBUG
		                                                 , angles_debugd
#endif
#ifdef DIHED_DEBUG
		                                                 , dihedrals_debugd
#endif
		                                                                   );
	}
#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[BONDED] += (clock() - cpu1) / CLOCKS_PER_SEC;//toc();//
#endif
	checkCUDAError("BondedInteractions");

//==============================================================================
#ifdef PROFILING
	cpu1 = clock();
#endif

	nonbondforce<<<NBdimGrid, NBdimBlock>>>(f4d_nonbond
#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
	                                      , f4d_nonbond0,
	                                        f4d_nonbond1
#endif
#endif
	                                      , nblistd,
	                                        evdwd,
	                                        eelecd
#ifdef DEBUG_NONB
	                                      , debug_nonbd
#endif
	                                                   );

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[NONBOND] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("Nonbond");

	//EvalProps();
	//TakeSnapshot(1);
//==============================================================================
//////////////////////////////////////////////////////////////////////////
#ifdef PME_CALC
//////////////////////////////////////////////////////////////////////////
	//Implementation of the PME Method...
	//SpreadCharges();

//==============================================================================
#ifdef PROFILING
	cpu1 = clock();
#endif

	if(nAtom < 40000){
		ChargeSpread_small<<<ChargeSpreadGrid, ChargeSpreadBlock>>>(Qd,
		                                                            r4d_scaled,
		                                                            cellL,
		                                                            numL
#ifdef DEBUG_PME
		                                                          , pme_debug_d
#endif
		                                                                       );
	}
	else{

		ChargeSpread_medium<<<ChargeSpreadGrid, ChargeSpreadBlock>>>(Qd,
		                                                             r4d_scaled,
		                                                             cellL,
		                                                             numL
#ifdef DEBUG_PME
		                                                           , pme_debug_d
#endif
		                                                                        );

	}

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[CHARGESPREAD] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("Spread Charges");

	//TakeSnapshot(1);

	//capture_position(2,0);
	//capture_Q();
//==============================================================================

#ifdef PROFILING
	cpu1 = clock();
#endif

	cufftExecC2C(plan, Qd, Qd, CUFFT_INVERSE);//compute inverse FFT

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[CUDAFFT] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
//==============================================================================
	//capture_Q();
//==============================================================================
#ifdef PROFILING
	cpu1 = clock();
#endif

	//Implemented as CUDA kernel
	BCMultiply<<<BCMuldimGrid, BCMuldimBlock>>>(Qd
#ifdef DEBUG_PME
	                                          , pme_debug_d
#endif
	                                                       );

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[BCMULTIPLY] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("BCMultiply");
//==============================================================================
	//capture_Q();
//==============================================================================
#ifdef PROFILING
	cpu1 = clock();
#endif

	cufftExecC2C(plan, Qd, Qd, CUFFT_FORWARD);//compute forward FFT

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[CUDAFFT] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
//==============================================================================

	//capture_position(2,0);
	//capture_Q();
//==============================================================================
#ifdef PROFILING
	cpu1 = clock();
#endif

	if(nAtom < 100000){
		//Calculate the PME Force and Energy on Particles.
		PMEForce_medium<<<PMEForceGrid, PMEForceBlock>>>(Qd,
		                                                 f4d,
		                                                 ePMEd
#ifdef PCONSTANT
		                                               , viriald
#endif
#ifdef DEBUG_PME
		                                               , pme_debug_d
#endif
		                                                            );
	}
	else{
		//Calculate the PME Force and Energy on Particles.
		PMEForce_large<<<PMEForceGrid, PMEForceBlock>>>(Qd,
		                                                f4d,
		                                                ePMEd
#ifdef PCONSTANT
		                                              , viriald
#endif
#ifdef DEBUG_PME
		                                              , pme_debug_d
#endif
		                                                           );
	}

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[PMEFORCE] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("PMEForce");

//////////////////////////////////////////////////////////////////////////
#endif		//PME_CALC
//////////////////////////////////////////////////////////////////////////
	//EvalProps();
	//TakeSnapshot(1);
//==============================================================================

	if(restraints){
		reduce_COM<<<nAtom / MAX_BLOCK_SIZE + 1, MAX_BLOCK_SIZE>>>(segidd,
		                                                           com0d,
		                                                           com1d,
		                                                           mass_segid0d,
		                                                           mass_segid1d);

		restraint_force<<<DynadimGrid, DynadimBlock>>>(com0d,
		                                               com1d,
		                                               segidd,
		                                               f4d,
		                                               r4d);
	}

}
