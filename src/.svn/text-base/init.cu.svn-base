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
#include "shake.h"

//-----------------------------------------------------------------------------
void InitVelocity(){
//-----------------------------------------------------------------------------
//rv are initialized with a random velocity corresponding to Temperature.
//-----------------------------------------------------------------------------

	float vSum[3];
	float vMag;
	int n;
	int k;
	float4 Val_Zero = {0.0f, 0.0f, 0.0f, 0.0f};

	if(shake){
		solve_bond_constraints<<<num_cluster_blocks,
		                         cluster_blocksize>>>(constraintsd,
		                                              constraints_by_atomd,
		                                              constraintsprmd,
		                                              atoms_in_clusterd,
		                                              r4shaked,
		                                              r4d,
		                                              v4d
#ifdef PCONSTANT
		                                            , viriald
#endif
		                                                     );
	}

	// Generates random velocities
	//seed = 13597.0;
	for(k = 0; k < 3; k++){
		vSum[k] = 0.0;
	}

	/*
	double e[3];
	for(n=0; n<nAtom; n++) {
		RandVec3(e,&seed);
		vMag = (float)sqrt(3*RC*InitTemp/prm[type[n]].x);
		// vMag = (float)sqrt(3*RC*298/prm[type[n]].x);
		// printf("vMag is %f - InitTemp is %f - prm[type[n]].x is %f\n",
		vMag, InitTemp, prm[type[n]].x);
		rv[n].x = vMag*((float) e[0]);
		rv[n].y = vMag*((float) e[1]);
		rv[n].z = vMag*((float) e[2]);
		vSum[0] += prm[type[n]].x*rv[n].x;
		vSum[1] += prm[type[n]].x*rv[n].y;
		vSum[2] += prm[type[n]].x*rv[n].z;
	}

	*/

#ifdef DEBUG_NPT
	srand((unsigned int)seed);
#else
	srand((unsigned int)time(NULL));
#endif

	float mass;
	float2 e;
#ifdef USE_CONSFIX
	for(n = 0; n < nAtomwoseg; n++){
#else
	for(n = 0; n < nAtom; n++){
#endif
		mass = prm[type[n]].x;
		vMag = (float)sqrt(RC * InitTemp / mass);

		genrandfloat2(e);
		rv[n].x = vMag * sqrt(-2.0f * log(e.x)) * cos(2.0f * PI * (e.y));

		genrandfloat2(e);
		rv[n].y = vMag * sqrt(-2.0f * log(e.x)) * cos(2.0f * PI * (e.y));

		genrandfloat2(e);
		rv[n].z = vMag * sqrt(-2.0f * log(e.x)) * cos(2.0f * PI * (e.y));

		vSum[0] += mass * rv[n].x;
		vSum[1] += mass * rv[n].y;
		vSum[2] += mass * rv[n].z;
	}


	//Makes the total momentum zero
#ifdef USE_CONSFIX
	for(k = 0; k < 3; k++){
		vSum[k] = vSum[k] / nAtomwoseg;
	}
#else
	for(k = 0; k < 3; k++){
		vSum[k] = vSum[k] / nAtom;
	}
#endif

#ifdef USE_CONSFIX
	for(n = 0; n < nAtomwoseg; n++){
#else
	for(n = 0; n < nAtom; n++){
#endif
		rv[n].x -= vSum[0] / prm[type[n]].x;
		rv[n].y -= vSum[1] / prm[type[n]].x;
		rv[n].z -= vSum[2] / prm[type[n]].x;

#ifdef PCONSTANT
		rv[n].x += (r[n].x * RegionVeloc.x / Region.x);
		rv[n].y += (r[n].y * RegionVeloc.y / Region.y);
		rv[n].z += (r[n].z * RegionVeloc.z / Region.z);
#endif
	}

#ifdef USE_CONSFIX
	// for (n=0; n<nAtom; n++)
	for(n = nAtomwoseg; n < NMAX; n++){
#else
	for(n = nAtom; n < NMAX; n++){
#endif
		rv[n] = Val_Zero;
	}

////////save velocities//////////
/*
	FILE* testfile;
	testfile = fopen("C:\\UserData\\Narayan\\Data_and_Scripts\\velocities.dat", "w");
	for(n=0; n<nAtom; n++)
		fprintf(testfile, "%d\t%f\t%f\t%f\n", (int) prm[type[n]].x, rv[n].x,
		rv[n].y, rv[n].z);

	fclose(testfile);
*/
////////end save velocities///////

	//exit(0);
	cudaMemcpy(v4d, rv, f4size, cudaMemcpyHostToDevice);

	//--------------------Enable for RATTLE----------------------------
	/*
	if (shake){
		solve_velocity_constraints<<<num_cluster_blocks,
		cluster_blocksize>>>(constraintsd, constraints_by_atomd,
		constraintsprmd, atoms_in_clusterd
		, v4d	//, r4d
#ifdef PCONSTANT
					, viriald
#endif
		);

		cudaThreadSynchronize();
		checkCUDAError("Rattle");
	}
	*/

	//--------------------Enable for SHAKE----------------------------

	if(shake){
		//save the old positions before updating coordinates...
		cudaMemcpy(r4shaked, r4d, f4size, cudaMemcpyDeviceToDevice);

		UpdateCoords<<<DynadimGrid, DynadimBlock>>>(r4d, v4d
#ifdef PCONSTANT
		                                          , viriald
#endif
		                                                   );

		solve_bond_constraints<<<num_cluster_blocks,
		                         cluster_blocksize>>>(constraintsd,
		                                              constraints_by_atomd,
		                                              constraintsprmd,
		                                              atoms_in_clusterd,
		                                              r4shaked,
		                                              r4d,
		                                              v4d
#ifdef PCONSTANT
		                                            , viriald
#endif
		                                                     );

		cudaThreadSynchronize();
		checkCUDAError("SHAKE");

		cudaMemcpy(r4d, r4shaked, f4size, cudaMemcpyDeviceToDevice);
	}

	//Zero out Forces..
	memset(f4h, 0, sizeof(float4) * NMAX);
	memset(f4h_bonded, 0, BPART * sizeof(float4) * NMAX);
	memset(f4h_nonbond, 0, NBPART * sizeof(float4) * NMAX);
#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
	memset(f4h_nonbond0, 0, NBPART * sizeof(float4) * NMAX);
	memset(f4h_nonbond1, 0, NBPART * sizeof(float4) * NMAX);
#endif
#endif

	return;
}

