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

/*------------------------------------------------------------------------------
Updates acceleration of one atom from a single bonded pair interaction (on GPU),
4th coord of acceleration used for potential energy
------------------------------------------------------------------------------*/
__device__ float4 bondInteraction(float4 r, float2 bprm, float& eBond){

	//float3 r;
	float4 force_energy = {0.0f, 0.0f, 0.0f, 0.0f};

	//calculate r^2
	float rsqr = __my_add3(r.x * r.x, r.y * r.y, r.z * r.z);

	//calculate 1 / r
	float ir = rsqrtf(rsqr);
	float rb = 1.0f / ir;

	//fr = F / r = kb(1 - r0/r)
	//float fr = -KB * (1.0f - ROH*ir);
	float fr = __my_fmul(-bprm.x, __my_fadd(1.0f , -(__my_fmul(bprm.y, ir))));

	//energy, cut by half since bonds will be double counted
	//float ener = 0.25 * KB * (rb - ROH)*(rb - ROH);
	force_energy.w = __my_mul3(__my_fmul(0.25, bprm.x),
	                           __my_fadd(rb, -bprm.y),
	                           __my_fadd(rb, -bprm.y));

	force_energy.x = __my_fmul(fr, r.x);
	force_energy.y = __my_fmul(fr, r.y);
	force_energy.z = __my_fmul(fr, r.z);

	eBond += force_energy.w;

	return force_energy;
}

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
                                 , float3 &VirSum
#endif
                                                 ){

	float3 force0;
	float3 force1;
	float3 force2;

	float4 force_energy;

	float rv1sqr = __my_fdiv(1.0f, v1 % v1);
	float rv2sqr = __my_fdiv(1.0f, v2 % v2);
	float v1v2 = v1 % v2;

	//calculate cos theta, sin theta, and theta
	float q1 = sqrtf(__my_fmul(rv1sqr, rv2sqr));
	float costheta = -__my_fmul(v1 % v2, q1);

	costheta = (abs(costheta) > 0.9999f) ?
	           __mycopysignf(0.9999f, costheta) : costheta;

	float sintheta = sqrtf(__my_fadd(1.0f, -__my_fmul(costheta, costheta)));

	float dtheta = __my_fadd(__my_facos(costheta), -aprm.y);

	float kfactor = __my_fmul(-2 * aprm.x, dtheta) / sintheta;

	force0 = (v1 * __my_fmul(v1v2, rv1sqr) - v2);

	force2 = (v1 - v2 * __my_fmul(v1v2, rv2sqr));

	force1 = (force0 + force2) * (-1);

#ifdef PCONSTANT
	VirSum += (force2 * v2 - force0 * v1) *
	          __my_fmul((2.0f / 3.0f) * kfactor, q1);
	//2.0f/3.0f because each angle is counted thrice
	//and multiplied by 0.5 before writing to global...
	//VirSum += (force2*(v2*(id>=1)) -
	//force0*(v1*(id<=1)))*__my_fmul(kfactor, q1);
#endif
	//multiply the appropriate force by the derivative factor...

	force_energy = float3_to_float4(force0 * (id == 0) +
	                                force1 * (id == 1) +
	                                force2 * (id == 2)) *
	                                __my_fmul(kfactor, q1);

	//float ener = KA * dtheta * dtheta / 2;
	//multiply by 1/3 as angle is triple counted...
	force_energy.w = __my_mul4(aprm.x, dtheta, dtheta, 1.0f / 3.0f);

	eAngle += force_energy.w;

	return force_energy;
}


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
                                                    ){
//------------------------------------------------------------------------------

	float cospsi;
	float sinpsi;
	float psi;
	float dE;

	float4 force_energy; // = {0.0f, 0.0f, 0.0f, 0.0f};

	float t1;
	float t2;
	float t3;
	float t4;
	float t5;
	float t6;

	float c11;
	float c12;
	float c13;
	float c22;
	float c23;
	float c33;

	float q1;
	float q2;
	float q12;

	float3 f0;
	float3 f1;
	float3 f2;
	float3 f3;

	c11 = v1 % v1;
	c22 = v2 % v2;
	c33 = v3 % v3;
	c12 = v1 % v2;
	c13 = v1 % v3;
	c23 = v2 % v3;

	t1 = __my_fadd(__my_fmul(c13, c22), -__my_fmul(c12, c23));
	t2 = __my_fadd(__my_fmul(c11, c23), -__my_fmul(c12, c13));
	t3 = __my_fadd(__my_fmul(c12, c12), -__my_fmul(c11, c22));
	t4 = __my_fadd(__my_fmul(c22, c33), -__my_fmul(c23, c23));
	t5 = __my_fadd(__my_fmul(c13, c23), -__my_fmul(c12, c33));
	t6 = -t1;

	q12 = sqrt(__my_fmul(-t3, t4));

	cospsi = -__my_fdiv(t1, q12);
	cospsi = (cospsi > 1.0f) ? 1.0f : cospsi;
	psi = acos(cospsi);
	sinpsi = sqrt(__my_fadd(1.0f, -__my_fmul(cospsi, cospsi)));

	f0 = (v1 * t1 + v2 * t2 + v3 * t3) * __fdividef(c22, __my_fmul(q12, -t3));
	f3 = (v1 * t4 + v2 * t5 + v3 * t6) * __fdividef(c22, __my_fmul(q12, t4));

	q1 = __my_fdiv(c12, c22);
	q2 = __my_fdiv(c23, c22);

	f1 = f3 * q2 - f0 * __my_fadd(1.0f, q1);
	f2 = f0 * q1 - f3 * __my_fadd(1.0f, q2);

	dE = (psi == 0.0f) ? (-2 * prm.x) : (-2 * prm.x * psi / sinpsi);

	//force is the negative of gradient...
	//f[id] = f[id]*dE; //;
	f0 = f0 * dE;
	f1 = f1 * dE;
	f2 = f2 * dE;
	f3 = f3 * dE;

	force_energy = float3_to_float4(f0 * (id == 0) +
	                                f1 * (id == 1) +
	                                f2 * (id == 2) +
	                                f3 * (id == 3));
#ifdef PCONSTANT
	VirSum += (f2 * v2 + f3 * (v2 + v3) - f0 * v1) * 0.5f;
	//0.5f because, each dihedral is counted fourtimes,
	//and multiplied by 0.5 before writing to global
#endif

	//multiply by 0.25 as dihed is quad-counted..
	force_energy.w = __my_mul4(0.25f, prm.x, psi, psi);

	eImprop += force_energy.w;

	return force_energy;
}

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
                                                   ){
//------------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////
///// For Ref, See: D. C. Rapaport, The Art of Molecular
///// Dynamics Simulation, Sec, 10.4 - Internal Forces..
////////////////////////////////////////////////////////////////////////////////
	float t1;
	float t2;
	float t3;
	float t4;
	float t5;
	float t6;

	float c11;
	float c12;
	float c13;
	float c22;
	float c23;
	float c33;

	//float invrasqr, invrbsqr, invrab;//, gab;
	float cosines1;
	float cosines2;
	float cosines3;
	float cosines4;
	float cosines5;
	float cosines6;
	float cosNphi;

	float sines1;
	float sines2;
	float sines3;
	float sines4;
	float sines5;
	float sines6;
	float sinNphi;

	float4 force_energy = {0.0f, 0.0f, 0.0f, 0.0f};

	float3 f0;
	float3 f1;
	float3 f2;
	float3 f3;

	float q1;
	float q2;
	float q12;


	float3 a = v1 ^ v2;
	float3 b = (v3 ^ v2) * (-1.0f);

	//define a few variables here...
	//dot products...
	c11 = v1 % v1;
	c22 = v2 % v2;
	c33 = v3 % v3;
	c12 = v1 % v2;
	c13 = v1 % v3;
	c23 = v2 % v3;

	t1 = __my_fadd(__my_fmul(c13, c22), -__my_fmul(c12, c23));
	t2 = __my_fadd(__my_fmul(c11, c23), -__my_fmul(c12, c13));
	t3 = __my_fadd(__my_fmul(c12, c12), -__my_fmul(c11, c22));
	t4 = __my_fadd(__my_fmul(c22, c33), -__my_fmul(c23, c23));
	t5 = __my_fadd(__my_fmul(c13, c23), -__my_fmul(c12, c33));
	t6 = -t1;

	q12 = sqrt(__my_fmul(-t3, t4));

	cosines1 = -__my_fdiv(t1, q12);
	//sines1 = -__my_fdiv(t2, q12);
	sines1 = __my_fdiv((a % v3) * sqrt(c22), sqrt((a % a) * (b % b)));
	//if (abs(cosines1)>1.0f) sines1 = 0;
	//else sines1 = sqrt(1-cosines1*cosines1);

	//enumerate all sines and cosines to avoid iterative looping and
	//reuse the values for multiple dihedral folds
	cosines2 = cosines1 * cosines1 - sines1 * sines1;
	sines2 = cosines1 * sines1 + sines1 * cosines1;

	cosines3 = cosines2 * cosines1 - sines2 * sines1;
	sines3 = cosines2 * sines1 + sines2 * cosines1;

	cosines4 = cosines3 * cosines1 - sines3 * sines1;
	sines4 = cosines3 * sines1 + sines3 * cosines1;

	cosines5 = cosines4 * cosines1 - sines4 * sines1;
	sines5 = cosines4 * sines1 + sines4 * cosines1;

	cosines6 = cosines5 * cosines1 - sines5 * sines1;
	sines6 = cosines5 * sines1 + sines5 * cosines1;

	char j;
	dihedralParameter dprm;
	float sind;
	float cosd;
	f0 = (v1 * t1 + v2 * t2 + v3 * t3) * __fdividef(c22, __my_fmul(q12, -t3));
	f3 = (v1 * t4 + v2 * t5 + v3 * t6) * __fdividef(c22, __my_fmul(q12, t4));

	q1 = __my_fdiv(c12, c22);
	q2 = __my_fdiv(c23, c22);

	f1 = f3 * q2 - f0 * __my_fadd(1.0f, q1);
	f2 = f0 * q1 - f3 * __my_fadd(1.0f, q2);


	for (j = 0; j < dcount; j++){
		dprm = dihedprm[j * MAX_DIHED_TYPE + h];

		cosNphi = cosines1 * (dprm.n == 1) +
		          cosines2 * (dprm.n == 2) +
		          cosines3 * (dprm.n == 3) +
		          cosines4 * (dprm.n == 4) +
		          cosines5 * (dprm.n == 5) +
		          cosines6 * (dprm.n == 6);

		sinNphi = sines1 * (dprm.n == 1) +
		          sines2 * (dprm.n == 2) +
		          sines3 * (dprm.n == 3) +
		          sines4 * (dprm.n == 4) +
		          sines5 * (dprm.n == 5) +
		          sines6 * (dprm.n == 6);

		sind = sin(dprm.d);
		cosd = cos(dprm.d);

		cosNphi = cosNphi * cosd + sinNphi * sind;
		sinNphi = cosNphi * sind - sinNphi * cosd;

		force_energy += To_float4( (f0 * (id == 0) +
		                            f1 * (id == 1) +
		                            f2 * (id == 2) +
		                            f3 * (id == 3)) *
		                            (-1.0f * dprm.n * sinNphi),
		                          0.25f * dprm.x * (1.0f + cosNphi));

#ifdef PCONSTANT
		VirSum += ((f2 + f3) * v2 + f3 * v3 - f0v * v1) * 0.5f;
		//VirSum += ((f2 + f3)*v2 + f3*v3 -f0*v1)*
		//__my_fmul(0.5f*sign*prm.x, dcosines1);
		//0.5f because, each dihedral is counted fourtimes,
		//and multiplied by 0.5 before writing to global = 2/4
#endif
	}

	eDihed += force_energy.w;

	return force_energy;
}

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
                                                        ){
//------------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////
///// For Ref, See: D. C. Rapaport, The Art of Molecular Dynamics Simulation,
///// Sec, 10.4 - Internal Forces..
////////////////////////////////////////////////////////////////////////////////

	float t1;
	float t2;
	float q;
	float rq;
	float sqrtq;
	float rsqrtq;

	float c11;
	float c22;
	float c33;
	float c12;
	float c13;
	float c23;
	float q1;
	float q2;

	c11 = v1 % v1;
	c22 = v2 % v2;
	c33 = v3 % v3;
	c12 = v1 % v2;
	c13 = v1 % v3;
	c23 = v2 % v3;

	/*
	t1 = __my_fadd(__my_fmul(c13, c22), -__my_fmul(c12, c23));
	t3 = __my_fadd(__my_fmul(c12, c12), -__my_fmul(c11, c22));
	t4 = __my_fadd(__my_fmul(c22, c33), -__my_fmul(c23, c23));
	q12 = sqrt(__my_fmul(-t3, t4));
	float3 a = v1^v2;

	cosines1 = -__my_fdiv(t1, q12);
	sines1 = __my_fdiv(sqrt(c22)*(v3%a), q12);
	*/

	t1 = c13 * c22 - c12 * c23;
	t2 = sqrt(c22) * (v3 % (v1 ^ v2));
	q = (c11 * c22 - c12 * c12) * (c22 * c33 - c23 * c23);
	rq = 1.0f / q;
	sqrtq = sqrt(q);
	rsqrtq = rsqrt(q);

	float cosNphi;
	float tempcosN;
	float cosphi;

	float sinNphi;
	float tempsinN;
	float sinphi;

	cosphi = -t1 / sqrtq;
	cosNphi = cosphi;

	sinphi = t2/sqrtq;
	sinNphi = sinphi;

	float3 dt1_dr0 = (v3 * (-c22) + v2 * c23);
	float3 dt1_dr3 = (v1 * c22 - v2 * c12);

	float3 dt2_dr0 = (v3 ^ v2) * sqrt(c22);
	float3 dt2_dr3 = (v1 ^ v2) * sqrt(c22);

	float3 dq_dr0 = (v1 * (-2.0f * c22) - v2 *
	                (-2.0f * c12)) * (c22 * c33 - c23 * c23);

	float3 dq_dr3 = (v3 * (2.0f * c22) - v2 *
	                (2.0f * c23)) * (c11 * c22 - c12 * c12);

	float3 dtempcos_dr0;
	float3 dtempcos_dr3;
	float3 dcosNphi_dr0;
	float3 dcosNphi_dr3;
	float3 dcosphi_dr0;
	float3 dcosphi_dr3;

	float3 dtempsin_dr0;
	float3 dtempsin_dr3;
	float3 dsinNphi_dr0;
	float3 dsinNphi_dr3;
	float3 dsinphi_dr0;
	float3 dsinphi_dr3;

	dcosphi_dr0 = dt1_dr0 * (-rsqrtq) + dq_dr0 * (t1 * -0.5f * rq * (-rsqrtq));
	dcosNphi_dr0 = dcosphi_dr0;

	dcosphi_dr3 = dt1_dr3 * (-rsqrtq) + dq_dr3 * (t1 * -0.5f * rq * (-rsqrtq));
	dcosNphi_dr3 = dcosphi_dr3;

	dsinphi_dr0 = dt2_dr0 * rsqrtq + dq_dr0 * (t2 * -0.5f * rq * rsqrtq);
	dsinNphi_dr0 = dsinphi_dr0;

	dsinphi_dr3 = dt2_dr3 * rsqrtq + dq_dr3 * (t2 * -0.5f * rq * rsqrtq);
	dsinNphi_dr3 = dsinphi_dr3;

	float3 f0;
	float3 f1;
	float3 f2;
	float3 f3;

	float4 force_energy = {0.0f, 0.0f, 0.0f, 0.0f};

	q1 = __my_fdiv(c12, c22);
	q2 = __my_fdiv(c23, c22);

	char j;
	char k;
	char nprev=1;

	float sind;
	float cosd;

	dihedralParameter dprm;

	for (j = 0; j < dcount; j++){
		dprm = dihedprm[j * MAX_DIHED_TYPE + h];
		/*
		cosNphi = cosphi;
		sinNphi = sinphi;
		dcosNphi_dr0 = dcosphi_dr0;
		dcosNphi_dr3 = dcosphi_dr3;
		dsinNphi_dr0 = dsinphi_dr0;
		dsinNphi_dr3 = dsinphi_dr3;
		*/

		//now compute the force and the energies...
		for (k = nprev; k < dprm.n; k++){
			tempcosN = cosNphi * cosphi - sinNphi * sinphi;
			tempsinN = sinNphi * cosphi + cosNphi * sinphi;

			dtempcos_dr0 = dcosNphi_dr0 * cosphi +
			               dcosphi_dr0 * cosNphi -
			               dsinNphi_dr0 * sinphi -
			               dsinphi_dr0 * sinNphi;

			dtempsin_dr0 = dsinNphi_dr0 * cosphi +
			               dcosphi_dr0 * sinNphi +
			               dcosNphi_dr0 * sinphi +
			               dsinphi_dr0 * cosNphi;

			dtempcos_dr3 = dcosNphi_dr3 * cosphi +
			               dcosphi_dr3 * cosNphi -
			               dsinNphi_dr3 * sinphi -
			               dsinphi_dr3 * sinNphi;

			dtempsin_dr3 = dsinNphi_dr3 * cosphi +
			               dcosphi_dr3 * sinNphi +
			               dcosNphi_dr3 * sinphi +
			               dsinphi_dr3 * cosNphi;

			cosNphi = tempcosN;
			sinNphi = tempsinN;

			dcosNphi_dr0 = dtempcos_dr0;
			dsinNphi_dr0 = dtempsin_dr0;

			dcosNphi_dr3 = dtempcos_dr3;
			dsinNphi_dr3 = dtempsin_dr3;
		}

		nprev = dprm.n;

		sind = sin(dprm.d);
		cosd = cos(dprm.d);

		f0 = (dcosNphi_dr0 * cosd + dsinNphi_dr0 * sind) * (-dprm.x);
		f3 = (dcosNphi_dr3 * cosd + dsinNphi_dr3 * sind) * (-dprm.x);

		f1 = f3 * q2 - f0 * __my_fadd(1.0f, q1);
		f2 = f0 * q1 - f3 * __my_fadd(1.0f, q2);

		force_energy += To_float4( (f0 * (id == 0) +
		                            f1 * (id == 1) +
		                            f2 * (id == 2) +
		                            f3 * (id == 3)),
		                          0.25 * dprm.x * (1.0f + cosNphi * cosd +
		                                           sinNphi * sind));

#ifdef PCONSTANT
		VirSum += ((f2 + f3) * v2 + f3 * v3 - f0 * v1) * 0.5f;
		//0.5f because, each dihedral is counted fourtimes,
		//and multiplied by 0.5 before writing to global = 2/4
#endif
	}

	eDihed += force_energy.w;
	return force_energy;
}



//------------------------------------------------------------------------------
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
                                                   ){
//------------------------------------------------------------------------------

	int idx1;
	int idx2;
	int idx3;
	int idx4;

	int di;
	int ai;

	float4 r1;
	float4 r2;
	float4 r3;
	float4 r4;

	float4 dr1;
	float4 dr2;
	float4 dr3;

	short int h = 0;
	short int i;
	short int count;

	//float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

	char id = 0;

	unsigned int gtid = COMPUTE_GLOBAL_THREADID
	int atomid = gtid % WorkgroupSized;
	short int start_count = gtid / WorkgroupSized;
	int t1 = tex1Dfetch(textype, atomid); // , t2;

	float2 prm14_0 = tex1Dfetch(texprm1_4, t1);
#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	//float myMass = tex1Dfetch(texprm, t1).x;

	float eVDW = 0.0f;
	float eCoul = 0.0f;
	float eEwexcl = 0.0f;

	float eBond = 0.0f;
	float eAngle = 0.0f;
	float eDihed = 0.0f;
	float4 force_energy = {0.0f, 0.0f, 0.0f, 0.0f};
	float2 parameter;

#ifdef PCONSTANT
	float3 VirSum = {0.0f, 0.0f, 0.0f};
	float4 tempForce;
	float3 RBSRat = float4_to_float3(tex1Dfetch(texRBSRat, 0));
#endif

//#define EXPLICIT_EWALD_LIST
//===========================begin Bond Computation=============================

	r1 = tex1Dfetch(texcrd, atomid); //gtid
	count = bonds_indexd[atomid]; //gtid

	for(i = start_count; i < count; i += BPART){ //for(i=0; i<count; i++) {

		h = bonds_indexd[WorkgroupSized * (2 * i + 1) + atomid]; //gtid
		idx2 = bonds_indexd[WorkgroupSized * (2 * i + 2) + atomid]; //gtid
		r2 = tex1Dfetch(texcrd, idx2);

		dr1.x = __my_fadd(r1.x, -r2.x);
		dr1.y = __my_fadd(r1.y, -r2.y);
		dr1.z = __my_fadd(r1.z, -r2.z);

		//nearest image...
		dr1.x = __my_add3(dr1.x,
		                  -__mycopysignf(boxH.x, dr1.x - boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x + boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  -__mycopysignf(boxH.y, dr1.y - boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y + boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  -__mycopysignf(boxH.z, dr1.z - boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z + boxH.z));

		//bi = bonds_indexd[WorkgroupSized*(i+1) + atomid];

		//h = bonds_setd[MAX_BOND_COUNT*0 + (bi+1)];
		//idx[0] = bonds_setd[MAX_BOND_COUNT*1 + (bi+1)];
		//idx[1] = bonds_setd[MAX_BOND_COUNT*2 + (bi+1)];

		//id = (gtid==idx[1]);

		//x1 = tex1Dfetch(texcrd, idx[id]);
		//x2 = tex1Dfetch(texcrd, idx[1-id]);

		parameter = bondprmd[h];

#ifdef PCONSTANT
		tempForce = bondInteraction(dr1, parameter, eBond);
		force_energy += tempForce;
		VirSum += float4_to_float3(tempForce * dr1);

#ifndef EXPLICIT_EWALD_LIST
		//Ewald Correction calculation...
		dr1.w = __my_fmul(r1.w, r2.w);
		tempForce = ewaldCorrection(dr1, eEwexcl) * (parameter.x > 0.0f);

		force_energy += tempForce;
		VirSum += float4_to_float3(tempForce * dr1);

		//Virial due to kspace Sum...
		VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * dr1.w *
		                    exp(-betaSqr * (dr1.x * dr1.x + dr1.y *
		                                    dr1.y + dr1.z * dr1.z)));
#endif //EXPLICIT_EWALD_LIST

#else //NO PCONSTANT

		force_energy += bondInteraction(dr1, parameter, eBond);

#ifndef EXPLICIT_EWALD_LIST	//Ewald Correction calculation...
		dr1.w = __my_fmul(r1.w, r2.w);
		force_energy += ewaldCorrection(dr1, eEwexcl) * (parameter.x > 0.0f);
#endif //EXPLICIT_EWALD_LIST

#endif
	}

	ebndd[gtid] = eBond;

#ifdef BOND_DEBUG
	bonds_debugd[gtid * NUM_BOND_DEBUG + 0] = gtid;
	bonds_debugd[gtid * NUM_BOND_DEBUG + 1] = blockIdx.x;
	bonds_debugd[gtid * NUM_BOND_DEBUG + 2] = blockIdx.y;
#endif

//========================begin Ewald Exclusion Computation=====================

#ifdef EXPLICIT_EWALD_LIST

//////Ewald Correction Calculation...
	count = ewlistd[atomid];

	for(i = 1 + start_count; i <= count; i += NBPART){
		idx1 = ewlistd[i * WorkgroupSized + atomid];
		r2 = tex1Dfetch(texcrd, idx1);

		dr1.x = __my_fadd(r1.x, -r2.x);
		dr1.y = __my_fadd(r1.y, -r2.y);
		dr1.z = __my_fadd(r1.z, -r2.z);

		// choose nearest image
		dr1.x = __my_add3(dr1.x,
		                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z+boxH.z));

		dr1.w = __my_fmul(r1.w, r2.w);

#ifdef PCONSTANT
		tempForce = ewaldCorrection(dr1, eEwexcl);
		force_energy += tempForce;
		VirSum += float4_to_float3(tempForce * dr1);

		//Virial due to kspace Sum...
		VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * dr1.w *
		                    exp(-betaSqr * (dr1.x * dr1.x + dr1.y *
		                                    dr1.y + dr1.z * dr1.z)));
#else
		force_energy += ewaldCorrection(dr1, eEwexcl);
#endif
	}

///////End Ewald Correction Calculation...
#endif //EXPLICIT_EWALD_LIST

//=====================begin Angle and UreyB Computation========================

	// loop through angle list for this atom, get indices of relevant atoms,
	// calculate forces
	count = angles_indexd[atomid];
	for(i = start_count; i < count; i += BPART){ //for(i=0; i<count; i++) {

		ai = angles_indexd[WorkgroupSized * (i + 1) + atomid];

		h = angles_setd[Angle_Countd * 0 + (ai + 1)];
		idx1 = angles_setd[Angle_Countd * 1 + (ai + 1)];
		idx2 = angles_setd[Angle_Countd * 2 + (ai + 1)];
		idx3 = angles_setd[Angle_Countd * 3 + (ai + 1)];

		r1 = tex1Dfetch(texcrd, idx1);

		r2 = tex1Dfetch(texcrd, idx2);

		r3 = tex1Dfetch(texcrd, idx3);

		dr1 = r2 - r1;
		dr2 = r3 - r2;

		//r1 = x1 - x2;
		//r2 = x3 - x2;

		id = (atomid == idx1) * 0 + (atomid == idx2) * 1 + (atomid == idx3) * 2;
		//id = 0 if gtid==idx1, 1 if gtid==idx2, 2 if gtid==idx3;

		//nearest images...
		dr1.x = __my_add3(dr1.x,
		                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z+boxH.z));


		dr2.x = __my_add3(dr2.x,
		                  -__mycopysignf(boxH.x, dr2.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr2.x+boxH.x));

		dr2.y = __my_add3(dr2.y,
		                  -__mycopysignf(boxH.y, dr2.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr2.y+boxH.y));

		dr2.z = __my_add3(dr2.z,
		                  -__mycopysignf(boxH.z, dr2.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr2.z+boxH.z));

		parameter = angleprmd[h];

		force_energy = force_energy + angleInteraction(float4_to_float3(dr1),
		                                               float4_to_float3(dr2),
		                                               id,
		                                               parameter,
		                                               eAngle
#ifdef PCONSTANT
                                                 , VirSum
#endif
                                                         );

#ifndef EXPLICIT_EWALD_LIST
		if((id == 0) || (id == 2)){
			//Ewald Correction calculation...
			dr1 = r1 - r3;
			dr1 = dr1 * ((id == 0) ? 1 : -1);

			dr1.x = __my_add3(dr1.x,
			                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
			                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

			dr1.y = __my_add3(dr1.y,
			                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
			                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

			dr1.z = __my_add3(dr1.z,
			                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
			                  -__mycopysignf(boxH.z, dr1.z+boxH.z));

			dr1.w = __my_fmul(r1.w, r3.w);

#ifdef PCONSTANT
			tempForce = ewaldCorrection(dr1, eEwexcl) * (parameter.x > 0.0f);
			force_energy += tempForce;
			VirSum += float4_to_float3(tempForce * dr1);

			//Virial due to kspace Sum...
			VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * dr1.w *
			                    exp(-betaSqr * (dr1.x * dr1.x + dr1.y *
			                                    dr1.y + dr1.z * dr1.z)));
#else
			force_energy += ewaldCorrection(dr1, eEwexcl);
#endif
		}
#endif //EXPLICIT_EWALD_LIST

	} //for(i = 0; i < count; i++){

	eangd[gtid] = eAngle;
	eEwexcld[gtid] = eEwexcl;

#ifdef UREY_BRADLEY
	float egy;
	float eUreyB = 0.0f;

	r1 = tex1Dfetch(texcrd, atomid);
	count = ureyb_indexd[atomid];

	for(i = start_count; i < count; i += BPART){ //for(i = 0; i < count; i++){

		h = ureyb_indexd[WorkgroupSized * (2 * i + 1) + atomid];
		idx3 = ureyb_indexd[WorkgroupSized * (2 * i + 2) + atomid];
		r3 = tex1Dfetch(texcrd, idx3);

		dr1 = r1 - r3;
		dr1.x = __my_add3(dr1.x,
		                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z+boxH.z));

		float r = sqrt(__my_add3(dr1.x * dr1.x, dr1.y * dr1.y, dr1.z * dr1.z));
		//float S_minus_S0 = r - ureybprmd[h].y;
		//float KS = ureybprmd[h].x*S_minus_S0;

		float2 C_UB = ureybprmd[h];
		//rewrite C_UB with partial computation..
		C_UB.y = r - C_UB.y;
		C_UB.x = C_UB.x * C_UB.y;
		egy = __my_fmul(0.5 * C_UB.x, C_UB.y);

#ifdef PCONSTANT
		tempForce.x = __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.x, r));
		tempForce.y = __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.y, r));
		tempForce.z = __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.z, r));
		tempForce.w = egy;

		force_energy += tempForce;

		VirSum += float4_to_float3(tempForce * dr1);
#else
		//force is negative of gradient...
		force_energy.x += __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.x, r));
		force_energy.y += __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.y, r));
		force_energy.z += __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.z, r));
		force_energy.w += egy;
#endif	//PCONSTANT

		eUreyB += egy;
	}//for(i=0; i<count; i++)

	eureybd[gtid] = eUreyB;

#endif //UREY_BRADLEY

#ifdef ANGLE_DEBUG
	angles_debugd[gtid * NUM_ANGLE_DEBUG + 0] = -force_energy.x;
	angles_debugd[gtid * NUM_ANGLE_DEBUG + 1] = -force_energy.y;
	angles_debugd[gtid * NUM_ANGLE_DEBUG + 2] = -force_energy.z;
#endif

//=========================begin dihedral computation===========================

////////////pairInteraction variables///////////////////////
	float fC;
	float fLJ;
	float fr;
/*
#ifdef USE_CONSFIX
        eCoul = 0.0f;
	eVDW = 0.0f;
	fC = 0.0f;
	fLJ = 0.0f;
	fr = __my_fadd(fLJ, fC);
	force_energy += make_float4(__my_fmul(fr, dr1.x),
	                            __my_fmul(fr, dr1.y),
	                            __my_fmul(fr, dr1.z),
	                            0.0f);
#else
*/
	int t2;
	float eps = 0.0f;
	float sigma2 = 0.0f;
#if (NBXMOD==5)
	float2 prm14_1;
#endif

////////////pairInteraction variables///////////////////////
	float betadist;
	float erf_c;
	float d_erfcxr;
	float enerC;
	float dist;
	float distSqr;
	float invDistSqr;
	float invDist;

	float swfunc;
	float sigma6xinvDist6;
	float enerLJ;
	float diff_swfunc_div_r;
	float var1;
	float var2;

	short int n;
///////////////////////////////////////////////////////////
	dihedralParameter * dprm = dihedral_prmd;
#ifdef DIHED_SHAREDMEM
	__shared__ dihedralParameter dihe_prm_sh[MAX_DIHED_TYPE * MAXDPRM];
	__shared__ int dihedral_count_sh[MAX_DIHED_TYPE];

	for(i = threadIdx.x; i < MAX_DIHED_TYPE * MAXDPRM; i += blockDim.x){
		dihe_prm_sh[i] = dihedral_prmd[i];
	}

	for (i = threadIdx.x; i < MAX_DIHED_TYPE; i += blockDim.x){
			dihedral_count_sh[i] = dihedral_type_countd[i];
	}

	__syncthreads();
	dprm = dihe_prm_sh;
#endif //DIHED_SHAREDMEM

	count = dihedrals_indexd[atomid];
	for(i = start_count; i < count; i += BPART){ //for (i=0; i<count; i++){
		//list of all the dihedrals that this atom belongs to...

		di = dihedrals_indexd[(i + 1) * WorkgroupSized + atomid];
		//index of the i'th dihedral that atom gtid belongs to..

		//with the dihedral index, now determine the individual
		//atoms and the corresponding index in the parameter array...
		h = dihedrals_setd[Dihed_Countd * 0 + (di + 1)];
		idx1 = dihedrals_setd[Dihed_Countd * 1 + (di + 1)];
		idx2 = dihedrals_setd[Dihed_Countd * 2 + (di + 1)];
		idx3 = dihedrals_setd[Dihed_Countd * 3 + (di + 1)];
		idx4 = dihedrals_setd[Dihed_Countd * 4 + (di + 1)];

		//id={0 if idx==gtid1, 1 if idx==gtid2, 2 if idx==gtid3, 3 if idx==gtid4}
		id = (idx1 == atomid) * 0 +
		     (idx2 == atomid) * 1 +
		     (idx3 == atomid) * 2 +
		     (idx4 == atomid) * 3;

		r1 = tex1Dfetch(texcrd, idx1);
		r2 = tex1Dfetch(texcrd, idx2);
		r3 = tex1Dfetch(texcrd, idx3);
		r4 = tex1Dfetch(texcrd, idx4);

		dr1 = r2 - r1;
		dr2 = r3 - r2;
		dr3 = r4 - r3;

		// ** nearest images!!! **
		dr1.x = __my_add3(dr1.x,
		                  - __mycopysignf(boxH.x, dr1.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  - __mycopysignf(boxH.y, dr1.y-boxH.y),
		                  - __mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  - __mycopysignf(boxH.z, dr1.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr1.z+boxH.z));


		dr2.x = __my_add3(dr2.x,
		                  - __mycopysignf(boxH.x, dr2.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr2.x+boxH.x));

		dr2.y = __my_add3(dr2.y,
		                  - __mycopysignf(boxH.y, dr2.y-boxH.y),
		                  - __mycopysignf(boxH.y, dr2.y+boxH.y));

		dr2.z = __my_add3(dr2.z,
		                  - __mycopysignf(boxH.z, dr2.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr2.z+boxH.z));



		dr3.x = __my_add3(dr3.x,
		                  - __mycopysignf(boxH.x, dr3.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr3.x+boxH.x));

		dr3.y = __my_add3(dr3.y,
		                  - __mycopysignf(boxH.y, dr3.y-boxH.y),
		- __mycopysignf(boxH.y, dr3.y+boxH.y));

		dr3.z = __my_add3(dr3.z,
		                  - __mycopysignf(boxH.z, dr3.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr3.z+boxH.z));

#ifdef DIHED_SHAREDMEM
		n = dihedral_count_sh[h];
#else
		n = dihedral_type_countd[h];
#endif	//DIHED_SHAREDMEM

		force_energy = force_energy +
                   torsionInteraction(float4_to_float3(dr1),
		                                  float4_to_float3(dr2),
		                                  float4_to_float3(dr3),
		                                  id,
		                                  dprm,
		                                  n,
		                                  h,
		                                  eDihed
//torsionInteraction_new(float4_to_float3(dr1), float4_to_float3(dr2),
// float4_to_float3(dr3), id, dprm, n, h, eDihed
#ifdef PCONSTANT
		                                , VirSum
#endif
		                                        );

#if (NBXMOD==5)
		//1-4 nonbond interactions...
		if((id == 0) || (id == 3)){
			dr1 = r1 - r4;
			dr1 = dr1 * ((id == 0) ? 1 : -1);

			dr1.x = __my_add3(dr1.x,
			                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
			                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

			dr1.y = __my_add3(dr1.y,
			                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
			                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

			dr1.z = __my_add3(dr1.z,
			                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
			                  -__mycopysignf(boxH.z, dr1.z+boxH.z));
			//order of r1.w and r4.w doesnot matter as we take only the product.
			dr1.w = __my_fmul(r1.w, r4.w);

			//now add the modified 1-4 interactions for VDW and Elec
			t2 = (id == 0) ? tex1Dfetch(textype, idx4) : tex1Dfetch(textype, idx1);
			prm14_1 = tex1Dfetch(texprm1_4, t2);

			eps = __my_fmul(4.0f, sqrt(__my_fmul(prm14_0.x, prm14_1.x)));
			sigma2 = __my_fmul(__my_fadd(prm14_0.y, prm14_1.y),
			                   __my_fadd(prm14_0.y, prm14_1.y));

			distSqr = __my_add3(__my_fmul(dr1.x, dr1.x),
			                    __my_fmul(dr1.y, dr1.y),
			                    __my_fmul(dr1.z, dr1.z));

			dist = sqrt(distSqr);

			// PAIR_INTERACTION(dr1)
			invDist = __my_fdiv(1.0, dist);

			invDistSqr = __my_fdiv(1.0, distSqr);

			betadist = __my_fmul(beta, dist);

			erf_c = erfc(betadist);

			d_erfcxr = __my_mul3(CC * TWO_OVER_SQRT_PI * dr1.w,
			                     beta,
			                     exp(-betadist * betadist));

			enerC = CC * dr1.w * invDist * erf_c;

			fC = (enerC + d_erfcxr) * invDistSqr;

			swfunc = 1.0f;

			sigma6xinvDist6 = __my_fmul(__my_mul3(sigma2, sigma2, sigma2),
			                            __my_mul3(invDistSqr,
			                                      invDistSqr,
			                                      invDistSqr));

			var1 = eps * sigma6xinvDist6;
			var2 = var1 * sigma6xinvDist6;

			enerLJ = var2 - var1;

			fLJ = (12.0f * var2 - 6.0f * var1) * invDistSqr;
			diff_swfunc_div_r = 0.0f;
			var1 = (Swcoeff1d - distSqr);
			var2 = (Swcoeff2d + 2.0f * distSqr);

			if(dist > Cutond){
				swfunc  = __my_mul4(var1, var1, var2, Swcoeff3d);
				diff_swfunc_div_r = __my_mul3(var1,
				                              Swcoeff3d,
				                              __my_fadd(4.0f * var2,
				                                        -var1 *
				                                        (Swcoeff2d * invDist + 4.0f)));
			}

			fLJ = __my_fadd(__my_fmul(fLJ, swfunc),
			                __my_fmul(enerLJ,
			                diff_swfunc_div_r));

			fr = __my_fadd(fLJ, fC);

			enerLJ = __my_fmul(enerLJ, swfunc);


#ifdef PCONSTANT
			tempForce = make_float4(__my_fmul(fr, dr1.x),
			                        __my_fmul(fr, dr1.y),
			                        __my_fmul(fr, dr1.z),
			                        __my_fadd(enerC, enerLJ));

			force_energy += tempForce;
			VirSum += float4_to_float3(tempForce * dr1);

			//K-space Virial..
			VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * dr1.w *
			                    exp(-betaSqr * (dr1.x * dr1.x + dr1.y *
			                                    dr1.y + dr1.z * dr1.z)));
#else
			force_energy += make_float4(__my_fmul(fr, dr1.x),
			                            __my_fmul(fr, dr1.y),
			                            __my_fmul(fr, dr1.z),
			                            __my_fadd(enerC, enerLJ));
#endif
			//force_energy.x += __my_fmul(fr, dr1.x);
			//force_energy.y += __my_fmul(fr, dr1.y);
			//force_energy.z += __my_fmul(fr, dr1.z);
			//force_energy.w += __my_fadd(enerC, enerLJ);

			eCoul = __my_fadd(eCoul, enerC);
			eVDW = __my_fadd(eVDW, enerLJ);

		}//if ((id==0)||(id==3)){
#endif

	}//for (i=0; i<dcount; i++){
	//list of all the dihedrals that this atom belongs to...

	//#endif  // ifdef USE_CONSFIX

	edihedd[gtid] = eDihed;
	eelecd[gtid] = 0.5f * eCoul;
	evdwd[gtid] = 0.5f * eVDW;

#ifdef IMPROPER
	float eImprop = 0.0f;
	count = impropers_indexd[atomid];

	for(i = start_count; i < count; i += BPART){ //for (i=0; i<count; i++){
		//list of all the dihedrals that this atom belongs to...

		//index of the i'th improper dihedral that atom gtid belongs to..
		di = impropers_indexd[(i + 1) * WorkgroupSized + atomid];

		//with the dihedral index, now determine the individual atoms
		//and the corresponding index in the parameter array...
		h = impropers_setd[Improper_Countd * 0 + (di + 1)];
		idx1 = impropers_setd[Improper_Countd * 1 + (di + 1)];
		idx2 = impropers_setd[Improper_Countd * 2 + (di + 1)];
		idx3 = impropers_setd[Improper_Countd * 3 + (di + 1)];
		idx4 = impropers_setd[Improper_Countd * 4 + (di + 1)];

		id = (idx1 == atomid) * 0 +
		     (idx2 == atomid) * 1 +
		     (idx3 == atomid) * 2 +
		     (idx4 == atomid) * 3;

		r1 = tex1Dfetch(texcrd, idx1);
		r2 = tex1Dfetch(texcrd, idx2);
		r3 = tex1Dfetch(texcrd, idx3);
		r4 = tex1Dfetch(texcrd, idx4);

		dr1 = r2 - r1;
		dr2 = r3 - r2;
		dr3 = r4 - r3;

		dr1.x = __my_add3(dr1.x,
		                  - __mycopysignf(boxH.x, dr1.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  - __mycopysignf(boxH.y, dr1.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  - __mycopysignf(boxH.z, dr1.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z+boxH.z));


		dr2.x = __my_add3(dr2.x,
		                  - __mycopysignf(boxH.x, dr2.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr2.x+boxH.x));

		dr2.y = __my_add3(dr2.y,
		                  - __mycopysignf(boxH.y, dr2.y-boxH.y),
		                  - __mycopysignf(boxH.y, dr2.y+boxH.y));

		dr2.z = __my_add3(dr2.z,
		                  - __mycopysignf(boxH.z, dr2.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr2.z+boxH.z));


		dr3.x = __my_add3(dr3.x,
		                  - __mycopysignf(boxH.x, dr3.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr3.x+boxH.x));

		dr3.y = __my_add3(dr3.y,
		                  - __mycopysignf(boxH.y, dr3.y-boxH.y),
		                  - __mycopysignf(boxH.y, dr3.y+boxH.y));

		dr3.z = __my_add3(dr3.z,
		                  - __mycopysignf(boxH.z, dr3.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr3.z+boxH.z));

		force_energy = force_energy +
		               improperInteraction(float4_to_float3(dr1),
		                                   float4_to_float3(dr2),
		                                   float4_to_float3(dr3),
		                                   id,
		                                   improper_prmd[h],
		                                   eImprop
#ifdef PCONSTANT
		                                 , VirSum
#endif
		                                         );
	}

	eimpropd[gtid] = eImprop;

#endif


	f4d_bonded[gtid] = force_energy;

#ifdef PCONSTANT
	viriald[gtid] += float3_to_float4(VirSum * 0.5f);
#endif

#ifdef DIHED_DEBUG
	dihedrals_debugd[gtid * NUM_DIHED_DEBUG + 0] = gtid; //-force_energy.x;
	dihedrals_debugd[gtid * NUM_DIHED_DEBUG + 1] = blockIdx.x; //-force_energy.y;
	dihedrals_debugd[gtid * NUM_DIHED_DEBUG + 2] = blockIdx.y; //-force_energy.z;
#endif

	return;
}

//------------------------------------------------------------------------------
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
                                                   ){
//------------------------------------------------------------------------------

	int idx1;
	int idx2;
	int idx3;
	int idx4;

	int di;
	int ai;

	float4 r1;
	float4 r2;
	float4 r3;
	float4 r4;

	float4 dr1;
	float4 dr2;
	float4 dr3;

	short int h = 0;
	short int i;
	short int count;

	//float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

	char id = 0;

	unsigned int gtid = COMPUTE_GLOBAL_THREADID
	int atomid = gtid % WorkgroupSized;
	short int start_count = gtid / WorkgroupSized;
	int t1 = tex1Dfetch(textype, atomid);
	int t2;


	float2 prm14_0 = tex1Dfetch(texprm1_4, t1);
#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	//float myMass = tex1Dfetch(texprm, t1).x;
#if (NBXMOD==5)
	float2 prm14_1;
#endif

	float eVDW = 0.0f;
	float eCoul = 0.0f;
	float eps = 0.0f;
	float sigma2 = 0.0f;
	float eEwexcl = 0.0f;

	float eBond = 0.0f;
	float eAngle = 0.0f;
	float eDihed = 0.0f;
	float4 force_energy = {0.0f, 0.0f, 0.0f, 0.0f};
	float2 parameter;

#ifdef PCONSTANT
	float3 VirSum = {0.0f, 0.0f, 0.0f};
	float4 tempForce;
	float3 RBSRat = float4_to_float3(tex1Dfetch(texRBSRat, 0));
#endif

//#define EXPLICIT_EWALD_LIST
//===========================begin Bond Computation=============================

	r1 = tex1Dfetch(texcrd, atomid); //gtid
	count = bonds_indexd[atomid]; //gtid

	for(i = start_count; i < count; i += BPART){ //for(i=0; i<count; i++) {

		h = bonds_indexd[WorkgroupSized * (2 * i + 1) + atomid]; //gtid
		idx2 = bonds_indexd[WorkgroupSized * (2 * i + 2) + atomid]; //gtid
		r2 = tex1Dfetch(texcrd, idx2);

		dr1.x = __my_fadd(r1.x, -r2.x);
		dr1.y = __my_fadd(r1.y, -r2.y);
		dr1.z = __my_fadd(r1.z, -r2.z);

		//nearest image...
		dr1.x = __my_add3(dr1.x,
		                  -__mycopysignf(boxH.x, dr1.x - boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x + boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  -__mycopysignf(boxH.y, dr1.y - boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y + boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  -__mycopysignf(boxH.z, dr1.z - boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z + boxH.z));

		//bi = bonds_indexd[WorkgroupSized*(i+1) + atomid];

		//h = bonds_setd[MAX_BOND_COUNT*0 + (bi+1)];
		//idx[0] = bonds_setd[MAX_BOND_COUNT*1 + (bi+1)];
		//idx[1] = bonds_setd[MAX_BOND_COUNT*2 + (bi+1)];

		//id = (gtid==idx[1]);

		//x1 = tex1Dfetch(texcrd, idx[id]);
		//x2 = tex1Dfetch(texcrd, idx[1-id]);

		parameter = bondprmd[h];

#ifdef PCONSTANT
		tempForce = bondInteraction(dr1, parameter, eBond);
		force_energy += tempForce;
		VirSum += float4_to_float3(tempForce * dr1);

#ifndef EXPLICIT_EWALD_LIST
		//Ewald Correction calculation...
		dr1.w = __my_fmul(r1.w, r2.w);
		tempForce = ewaldCorrection(dr1, eEwexcl) * (parameter.x > 0.0f);

		force_energy += tempForce;
		VirSum += float4_to_float3(tempForce * dr1);

		//Virial due to kspace Sum...
		VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * dr1.w *
		                    exp(-betaSqr * (dr1.x * dr1.x + dr1.y *
		                                    dr1.y + dr1.z * dr1.z)));
#endif //EXPLICIT_EWALD_LIST

#else //NO PCONSTANT

		force_energy += bondInteraction(dr1, parameter, eBond);

#ifndef EXPLICIT_EWALD_LIST	//Ewald Correction calculation...
		dr1.w = __my_fmul(r1.w, r2.w);
		force_energy += ewaldCorrection(dr1, eEwexcl) * (parameter.x > 0.0f);
#endif //EXPLICIT_EWALD_LIST

#endif
	}

	ebndd[gtid] = eBond;

#ifdef BOND_DEBUG
	bonds_debugd[gtid * NUM_BOND_DEBUG + 0] = gtid;
	bonds_debugd[gtid * NUM_BOND_DEBUG + 1] = blockIdx.x;
	bonds_debugd[gtid * NUM_BOND_DEBUG + 2] = blockIdx.y;
#endif

//========================begin Ewald Exclusion Computation=====================

#ifdef EXPLICIT_EWALD_LIST

//////Ewald Correction Calculation...
	count = ewlistd[atomid];

	for(i = 1 + start_count; i <= count; i += NBPART){
		idx1 = ewlistd[i * WorkgroupSized + atomid];
		r2 = tex1Dfetch(texcrd, idx1);

		dr1.x = __my_fadd(r1.x, -r2.x);
		dr1.y = __my_fadd(r1.y, -r2.y);
		dr1.z = __my_fadd(r1.z, -r2.z);

		// choose nearest image
		dr1.x = __my_add3(dr1.x,
		                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z+boxH.z));

		dr1.w = __my_fmul(r1.w, r2.w);

#ifdef PCONSTANT
		tempForce = ewaldCorrection(dr1, eEwexcl);
		force_energy += tempForce;
		VirSum += float4_to_float3(tempForce * dr1);

		//Virial due to kspace Sum...
		VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * dr1.w *
		                    exp(-betaSqr * (dr1.x * dr1.x + dr1.y *
		                                    dr1.y + dr1.z * dr1.z)));
#else
		force_energy += ewaldCorrection(dr1, eEwexcl);
#endif
	}

///////End Ewald Correction Calculation...
#endif //EXPLICIT_EWALD_LIST

//=====================begin Angle and UreyB Computation========================

	// loop through angle list for this atom, get indices of relevant atoms,
	// calculate forces
	count = angles_indexd[atomid];
	for(i = start_count; i < count; i += BPART){ //for(i=0; i<count; i++) {

		ai = angles_indexd[WorkgroupSized * (i + 1) + atomid];

		h = angles_setd[Angle_Countd * 0 + (ai + 1)];
		idx1 = angles_setd[Angle_Countd * 1 + (ai + 1)];
		idx2 = angles_setd[Angle_Countd * 2 + (ai + 1)];
		idx3 = angles_setd[Angle_Countd * 3 + (ai + 1)];

		r1 = tex1Dfetch(texcrd, idx1);

		r2 = tex1Dfetch(texcrd, idx2);

		r3 = tex1Dfetch(texcrd, idx3);

		dr1 = r2 - r1;
		dr2 = r3 - r2;

		//r1 = x1 - x2;
		//r2 = x3 - x2;

		id = (atomid == idx1) * 0 + (atomid == idx2) * 1 + (atomid == idx3) * 2;
		//id = 0 if gtid==idx1, 1 if gtid==idx2, 2 if gtid==idx3;

		//nearest images...
		dr1.x = __my_add3(dr1.x,
		                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z+boxH.z));


		dr2.x = __my_add3(dr2.x,
		                  -__mycopysignf(boxH.x, dr2.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr2.x+boxH.x));

		dr2.y = __my_add3(dr2.y,
		                  -__mycopysignf(boxH.y, dr2.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr2.y+boxH.y));

		dr2.z = __my_add3(dr2.z,
		                  -__mycopysignf(boxH.z, dr2.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr2.z+boxH.z));

		parameter = angleprmd[h];

		force_energy = force_energy + angleInteraction(float4_to_float3(dr1),
		                                               float4_to_float3(dr2),
		                                               id,
		                                               parameter,
		                                               eAngle
#ifdef PCONSTANT
                                                 , VirSum
#endif
                                                         );

#ifndef EXPLICIT_EWALD_LIST
		if((id == 0) || (id == 2)){
			//Ewald Correction calculation...
			dr1 = r1 - r3;
			dr1 = dr1 * ((id == 0) ? 1 : -1);

			dr1.x = __my_add3(dr1.x,
			                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
			                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

			dr1.y = __my_add3(dr1.y,
			                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
			                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

			dr1.z = __my_add3(dr1.z,
			                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
			                  -__mycopysignf(boxH.z, dr1.z+boxH.z));

			dr1.w = __my_fmul(r1.w, r3.w);

#ifdef PCONSTANT
			tempForce = ewaldCorrection(dr1, eEwexcl) * (parameter.x > 0.0f);
			force_energy += tempForce;
			VirSum += float4_to_float3(tempForce * dr1);

			//Virial due to kspace Sum...
			VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * dr1.w *
			                    exp(-betaSqr * (dr1.x * dr1.x + dr1.y *
			                                    dr1.y + dr1.z * dr1.z)));
#else
			force_energy += ewaldCorrection(dr1, eEwexcl);
#endif
		}
#endif //EXPLICIT_EWALD_LIST

	} //for(i = 0; i < count; i++){

	eangd[gtid] = eAngle;
	eEwexcld[gtid] = eEwexcl;

#ifdef UREY_BRADLEY
	float egy;
	float eUreyB = 0.0f;

	r1 = tex1Dfetch(texcrd, atomid);
	count = ureyb_indexd[atomid];

	for(i = start_count; i < count; i += BPART){ //for(i = 0; i < count; i++){

		h = ureyb_indexd[WorkgroupSized * (2 * i + 1) + atomid];
		idx3 = ureyb_indexd[WorkgroupSized * (2 * i + 2) + atomid];
		r3 = tex1Dfetch(texcrd, idx3);

		dr1 = r1 - r3;
		dr1.x = __my_add3(dr1.x,
		                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z+boxH.z));

		float r = sqrt(__my_add3(dr1.x * dr1.x, dr1.y * dr1.y, dr1.z * dr1.z));
		//float S_minus_S0 = r - ureybprmd[h].y;
		//float KS = ureybprmd[h].x*S_minus_S0;

		float2 C_UB = ureybprmd[h];
		//rewrite C_UB with partial computation..
		C_UB.y = r - C_UB.y;
		C_UB.x = C_UB.x * C_UB.y;
		egy = __my_fmul(0.5 * C_UB.x, C_UB.y);

#ifdef PCONSTANT
		tempForce.x = __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.x, r));
		tempForce.y = __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.y, r));
		tempForce.z = __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.z, r));
		tempForce.w = egy;

		force_energy += tempForce;

		VirSum += float4_to_float3(tempForce * dr1);
#else
		//force is negative of gradient...
		force_energy.x += __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.x, r));
		force_energy.y += __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.y, r));
		force_energy.z += __my_fmul(-2.0f * C_UB.x, __fdividef(dr1.z, r));
		force_energy.w += egy;
#endif	//PCONSTANT

		eUreyB += egy;
	}//for(i=0; i<count; i++)

	eureybd[gtid] = eUreyB;

#endif //UREY_BRADLEY

#ifdef ANGLE_DEBUG
	angles_debugd[gtid * NUM_ANGLE_DEBUG + 0] = -force_energy.x;
	angles_debugd[gtid * NUM_ANGLE_DEBUG + 1] = -force_energy.y;
	angles_debugd[gtid * NUM_ANGLE_DEBUG + 2] = -force_energy.z;
#endif

//=========================begin dihedral computation===========================
////////////pairInteraction variables///////////////////////
	float fC;
	float fLJ;
	float fr;
	float betadist;
	float erf_c;
	float d_erfcxr;
	float enerC;
	float dist;
	float distSqr;
	float invDistSqr;
	float invDist;

	float swfunc;
	float sigma6xinvDist6;
	float enerLJ;
	float diff_swfunc_div_r;
	float var1;
	float var2;

	short int n;
///////////////////////////////////////////////////////////
	dihedralParameter * dprm = dihedral_prmd;
#ifdef DIHED_SHAREDMEM
	__shared__ dihedralParameter dihe_prm_sh[MAX_DIHED_TYPE * MAXDPRM];
	__shared__ int dihedral_count_sh[MAX_DIHED_TYPE];

	for(i = threadIdx.x; i < MAX_DIHED_TYPE * MAXDPRM; i += blockDim.x){
		dihe_prm_sh[i] = dihedral_prmd[i];
	}

	for (i = threadIdx.x; i < MAX_DIHED_TYPE; i += blockDim.x){
			dihedral_count_sh[i] = dihedral_type_countd[i];
	}

	__syncthreads();
	dprm = dihe_prm_sh;
#endif //DIHED_SHAREDMEM

	count = dihedrals_indexd[atomid];
	for(i = start_count; i < count; i += BPART){ //for (i=0; i<count; i++){
		//list of all the dihedrals that this atom belongs to...

		di = dihedrals_indexd[(i + 1) * WorkgroupSized + atomid];
		//index of the i'th dihedral that atom gtid belongs to..

		//with the dihedral index, now determine the individual
		//atoms and the corresponding index in the parameter array...
		h = dihedrals_setd[Dihed_Countd * 0 + (di + 1)];
		idx1 = dihedrals_setd[Dihed_Countd * 1 + (di + 1)];
		idx2 = dihedrals_setd[Dihed_Countd * 2 + (di + 1)];
		idx3 = dihedrals_setd[Dihed_Countd * 3 + (di + 1)];
		idx4 = dihedrals_setd[Dihed_Countd * 4 + (di + 1)];

		//id={0 if idx==gtid1, 1 if idx==gtid2, 2 if idx==gtid3, 3 if idx==gtid4}
		id = (idx1 == atomid) * 0 +
		     (idx2 == atomid) * 1 +
		     (idx3 == atomid) * 2 +
		     (idx4 == atomid) * 3;

		r1 = tex1Dfetch(texcrd, idx1);
		r2 = tex1Dfetch(texcrd, idx2);
		r3 = tex1Dfetch(texcrd, idx3);
		r4 = tex1Dfetch(texcrd, idx4);

		dr1 = r2 - r1;
		dr2 = r3 - r2;
		dr3 = r4 - r3;

		// ** nearest images!!! **
		dr1.x = __my_add3(dr1.x,
		                  - __mycopysignf(boxH.x, dr1.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  - __mycopysignf(boxH.y, dr1.y-boxH.y),
		                  - __mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  - __mycopysignf(boxH.z, dr1.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr1.z+boxH.z));


		dr2.x = __my_add3(dr2.x,
		                  - __mycopysignf(boxH.x, dr2.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr2.x+boxH.x));

		dr2.y = __my_add3(dr2.y,
		                  - __mycopysignf(boxH.y, dr2.y-boxH.y),
		                  - __mycopysignf(boxH.y, dr2.y+boxH.y));

		dr2.z = __my_add3(dr2.z,
		                  - __mycopysignf(boxH.z, dr2.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr2.z+boxH.z));



		dr3.x = __my_add3(dr3.x,
		                  - __mycopysignf(boxH.x, dr3.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr3.x+boxH.x));

		dr3.y = __my_add3(dr3.y,
		                  - __mycopysignf(boxH.y, dr3.y-boxH.y),
		- __mycopysignf(boxH.y, dr3.y+boxH.y));

		dr3.z = __my_add3(dr3.z,
		                  - __mycopysignf(boxH.z, dr3.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr3.z+boxH.z));

#ifdef DIHED_SHAREDMEM
		n = dihedral_count_sh[h];
#else
		n = dihedral_type_countd[h];
#endif	//DIHED_SHAREDMEM

		force_energy = force_energy +
                   torsionInteraction(float4_to_float3(dr1),
		                                  float4_to_float3(dr2),
		                                  float4_to_float3(dr3),
		                                  id,
		                                  dprm,
		                                  n,
		                                  h,
		                                  eDihed
//torsionInteraction_new(float4_to_float3(dr1), float4_to_float3(dr2),
// float4_to_float3(dr3), id, dprm, n, h, eDihed
#ifdef PCONSTANT
		                                , VirSum
#endif
		                                        );

#if (NBXMOD==5)
		//1-4 nonbond interactions...
		if((id == 0) || (id == 3)){
			dr1 = r1 - r4;
			dr1 = dr1 * ((id == 0) ? 1 : -1);

			dr1.x = __my_add3(dr1.x,
			                  -__mycopysignf(boxH.x, dr1.x-boxH.x),
			                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

			dr1.y = __my_add3(dr1.y,
			                  -__mycopysignf(boxH.y, dr1.y-boxH.y),
			                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

			dr1.z = __my_add3(dr1.z,
			                  -__mycopysignf(boxH.z, dr1.z-boxH.z),
			                  -__mycopysignf(boxH.z, dr1.z+boxH.z));
			//order of r1.w and r4.w doesnot matter as we take only the product.
			dr1.w = __my_fmul(r1.w, r4.w);

			//now add the modified 1-4 interactions for VDW and Elec
			t2 = (id == 0) ? tex1Dfetch(textype, idx4) : tex1Dfetch(textype, idx1);
			prm14_1 = tex1Dfetch(texprm1_4, t2);

			eps = __my_fmul(4.0f, sqrt(__my_fmul(prm14_0.x, prm14_1.x)));
			sigma2 = __my_fmul(__my_fadd(prm14_0.y, prm14_1.y),
			                   __my_fadd(prm14_0.y, prm14_1.y));

			distSqr = __my_add3(__my_fmul(dr1.x, dr1.x),
			                    __my_fmul(dr1.y, dr1.y),
			                    __my_fmul(dr1.z, dr1.z));

			dist = sqrt(distSqr);

			// PAIR_INTERACTION(dr1)
			invDist = __my_fdiv(1.0, dist);

			invDistSqr = __my_fdiv(1.0, distSqr);

			betadist = __my_fmul(beta, dist);

			erf_c = erfc(betadist);

			d_erfcxr = __my_mul3(CC * TWO_OVER_SQRT_PI * dr1.w,
			                     beta,
			                     exp(-betadist * betadist));

			enerC = CC * dr1.w * invDist * erf_c;

			fC = (enerC + d_erfcxr) * invDistSqr;

			swfunc = 1.0f;

			sigma6xinvDist6 = __my_fmul(__my_mul3(sigma2, sigma2, sigma2),
			                            __my_mul3(invDistSqr,
			                                      invDistSqr,
			                                      invDistSqr));

			var1 = eps * sigma6xinvDist6;
			var2 = var1 * sigma6xinvDist6;

			enerLJ = var2 - var1;

			fLJ = (12.0f * var2 - 6.0f * var1) * invDistSqr;
			diff_swfunc_div_r = 0.0f;
			var1 = (Swcoeff1d - distSqr);
			var2 = (Swcoeff2d + 2.0f * distSqr);

			if(dist > Cutond){
				swfunc  = __my_mul4(var1, var1, var2, Swcoeff3d);
				diff_swfunc_div_r = __my_mul3(var1,
				                              Swcoeff3d,
				                              __my_fadd(4.0f * var2,
				                                        -var1 *
				                                        (Swcoeff2d * invDist + 4.0f)));
			}

			fLJ = __my_fadd(__my_fmul(fLJ, swfunc),
			                __my_fmul(enerLJ,
			                diff_swfunc_div_r));

			fr = __my_fadd(fLJ, fC);

			enerLJ = __my_fmul(enerLJ, swfunc);


#ifdef PCONSTANT
			tempForce = make_float4(__my_fmul(fr, dr1.x),
			                        __my_fmul(fr, dr1.y),
			                        __my_fmul(fr, dr1.z),
			                        __my_fadd(enerC, enerLJ));

			force_energy += tempForce;
			VirSum += float4_to_float3(tempForce * dr1);

			//K-space Virial..
			VirSum -= RBSRat * (beta * TWO_OVER_SQRT_PI * CC * dr1.w *
			                    exp(-betaSqr * (dr1.x * dr1.x + dr1.y *
			                                    dr1.y + dr1.z * dr1.z)));
#else
			force_energy += make_float4(__my_fmul(fr, dr1.x),
			                            __my_fmul(fr, dr1.y),
			                            __my_fmul(fr, dr1.z),
			                            __my_fadd(enerC, enerLJ));
#endif
			//force_energy.x += __my_fmul(fr, dr1.x);
			//force_energy.y += __my_fmul(fr, dr1.y);
			//force_energy.z += __my_fmul(fr, dr1.z);
			//force_energy.w += __my_fadd(enerC, enerLJ);

			eCoul = __my_fadd(eCoul, enerC);
			eVDW = __my_fadd(eVDW, enerLJ);

		}//if ((id==0)||(id==3)){
#endif

	}//for (i=0; i<dcount; i++){
	//list of all the dihedrals that this atom belongs to...

#ifdef USE_CONSFIX
	eCoul = 0.0f;
	eVDW = 0.0f;
#endif //ifdef USE_CONSFIX

	edihedd[gtid] = eDihed;
	eelecd[gtid] = 0.5f * eCoul;
	evdwd[gtid] = 0.5f * eVDW;

#ifdef IMPROPER
	float eImprop = 0.0f;
	count = impropers_indexd[atomid];

	for(i = start_count; i < count; i += BPART){ //for (i=0; i<count; i++){
		//list of all the dihedrals that this atom belongs to...

		//index of the i'th improper dihedral that atom gtid belongs to..
		di = impropers_indexd[(i + 1) * WorkgroupSized + atomid];

		//with the dihedral index, now determine the individual atoms
		//and the corresponding index in the parameter array...
		h = impropers_setd[Improper_Countd * 0 + (di + 1)];
		idx1 = impropers_setd[Improper_Countd * 1 + (di + 1)];
		idx2 = impropers_setd[Improper_Countd * 2 + (di + 1)];
		idx3 = impropers_setd[Improper_Countd * 3 + (di + 1)];
		idx4 = impropers_setd[Improper_Countd * 4 + (di + 1)];

		id = (idx1 == atomid) * 0 +
		     (idx2 == atomid) * 1 +
		     (idx3 == atomid) * 2 +
		     (idx4 == atomid) * 3;

		r1 = tex1Dfetch(texcrd, idx1);
		r2 = tex1Dfetch(texcrd, idx2);
		r3 = tex1Dfetch(texcrd, idx3);
		r4 = tex1Dfetch(texcrd, idx4);

		dr1 = r2 - r1;
		dr2 = r3 - r2;
		dr3 = r4 - r3;

		dr1.x = __my_add3(dr1.x,
		                  - __mycopysignf(boxH.x, dr1.x-boxH.x),
		                  -__mycopysignf(boxH.x, dr1.x+boxH.x));

		dr1.y = __my_add3(dr1.y,
		                  - __mycopysignf(boxH.y, dr1.y-boxH.y),
		                  -__mycopysignf(boxH.y, dr1.y+boxH.y));

		dr1.z = __my_add3(dr1.z,
		                  - __mycopysignf(boxH.z, dr1.z-boxH.z),
		                  -__mycopysignf(boxH.z, dr1.z+boxH.z));


		dr2.x = __my_add3(dr2.x,
		                  - __mycopysignf(boxH.x, dr2.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr2.x+boxH.x));

		dr2.y = __my_add3(dr2.y,
		                  - __mycopysignf(boxH.y, dr2.y-boxH.y),
		                  - __mycopysignf(boxH.y, dr2.y+boxH.y));

		dr2.z = __my_add3(dr2.z,
		                  - __mycopysignf(boxH.z, dr2.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr2.z+boxH.z));


		dr3.x = __my_add3(dr3.x,
		                  - __mycopysignf(boxH.x, dr3.x-boxH.x),
		                  - __mycopysignf(boxH.x, dr3.x+boxH.x));

		dr3.y = __my_add3(dr3.y,
		                  - __mycopysignf(boxH.y, dr3.y-boxH.y),
		                  - __mycopysignf(boxH.y, dr3.y+boxH.y));

		dr3.z = __my_add3(dr3.z,
		                  - __mycopysignf(boxH.z, dr3.z-boxH.z),
		                  - __mycopysignf(boxH.z, dr3.z+boxH.z));

		force_energy = force_energy +
		               improperInteraction(float4_to_float3(dr1),
		                                   float4_to_float3(dr2),
		                                   float4_to_float3(dr3),
		                                   id,
		                                   improper_prmd[h],
		                                   eImprop
#ifdef PCONSTANT
		                                 , VirSum
#endif
		                                         );
	}

	eimpropd[gtid] = eImprop;

#endif

	f4d_bonded[gtid] = force_energy;

#ifdef PCONSTANT
	viriald[gtid] += float3_to_float4(VirSum * 0.5f);
#endif

#ifdef DIHED_DEBUG
	dihedrals_debugd[gtid * NUM_DIHED_DEBUG + 0] = gtid; //-force_energy.x;
	dihedrals_debugd[gtid * NUM_DIHED_DEBUG + 1] = blockIdx.x; //-force_energy.y;
	dihedrals_debugd[gtid * NUM_DIHED_DEBUG + 2] = blockIdx.y; //-force_energy.z;
#endif

	return;
}
