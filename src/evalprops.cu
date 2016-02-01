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

/*----------------------------------------------------------------------------*/
void EvalProps(){
/*------------------------------------------------------------------------------
Evaluates physical properties: kinetic, potential & total energies.
------------------------------------------------------------------------------*/
	int n;

	float sumelec;
	float sumvdw;
	float sumbond;
	float sumangle;
	float sumdihed;

	float4 vel;

	char add_linebreak = 0;
	char add_prop = 0;
	char add_segidcoms = 0;


	// #ifdef USE_CONSFIX
	// int segval;
	//#endif

#ifdef PCONSTANT
	add_prop = 1;
#endif
	if(restraints){
		add_segidcoms = 1;
	}

	// printf("Start EvalProps\n");

#ifdef PCONSTANT
	float pressure;
	float scalarvirialT = 0.0f;

	float4 kineticE;
	float4 VirialTensor = {0.0f, 0.0f, 0.0f, 0.0f};

	// copy accelerations and velocities from gpu
	cudaMemcpy(&kineticE, kineticd,sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(&VirialTensor, viriald, sizeof(float4), cudaMemcpyDeviceToHost);
#endif

#ifdef PCONSTANT
	float4 boxLength;
	cudaMemcpy(&boxLength, boxLengthd, sizeof(float4), cudaMemcpyDeviceToHost);

	float4 boxVeloc;
	cudaMemcpy(&boxVeloc, boxVelocd, sizeof(float4), cudaMemcpyDeviceToHost);

	float4 boxAccel;
	cudaMemcpy(&boxAccel, boxAcceld, sizeof(float4), cudaMemcpyDeviceToHost);

	float4 RBSRat;
	cudaMemcpy(&RBSRat, ReciprocalBoxSquareRatiod,
	           sizeof(float4), cudaMemcpyDeviceToHost);

	int4 numcells;
	cudaMemcpy(&numcells, numcellsd, sizeof(int4), cudaMemcpyDeviceToHost);

#ifdef PME_CALC
	float4 frac;
	cudaMemcpy(&frac, fracd, sizeof(float4), cudaMemcpyDeviceToHost);
#endif

#endif
	cudaMemcpy(r, r4d, f4size, cudaMemcpyDeviceToHost);
	cudaMemcpy(f4h, f4d, f4size, cudaMemcpyDeviceToHost);
	cudaMemcpy(f4h_nonbond, f4d_nonbond, NBPART * f4size, cudaMemcpyDeviceToHost);
#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
	cudaMemcpy(f4h_nonbond0, f4d_nonbond0, NBPART * f4size,
	           cudaMemcpyDeviceToHost);
	cudaMemcpy(f4h_nonbond1, f4d_nonbond1, NBPART * f4size,
	           cudaMemcpyDeviceToHost);
#endif
#endif
	cudaMemcpy(f4h_bonded, f4d_bonded, BPART * f4size, cudaMemcpyDeviceToHost);
	cudaMemcpy(rv, v4d, f4size, cudaMemcpyDeviceToHost);
	cudaMemcpy(evdw, evdwd, NBPART * fsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(eelec, eelecd, NBPART * fsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(ebnd, ebndd, BPART * fsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(eang, eangd, BPART * fsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(edihed, edihedd, BPART * fsize, cudaMemcpyDeviceToHost);

#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
	// write forces to files
	char filename0[100];
	char filename1[100];
	char filename2[100];

	sprintf(filename0, "force_all%d.dat", stepCount);
	FILE* forcefile0 = fopen(filename0, "wb");
	sprintf(filename1, "force_sega%d.dat", stepCount);
	FILE* forcefile1 = fopen(filename1, "wb");
	sprintf(filename2, "force_segb%d.dat", stepCount);
	FILE* forcefile2 = fopen(filename2, "wb");

	for(int i = 0; i < nAtom; i++){
		fprintf(forcefile0, "%d %f %f %f %f \n", i,
		        f4h_nonbond[i].x + f4h_nonbond[WorkgroupSize + i].x,
		        f4h_nonbond[i].y + f4h_nonbond[WorkgroupSize + i].y,
		        f4h_nonbond[i].z + f4h_nonbond[WorkgroupSize + i].z,
		        f4h_nonbond[i].w + f4h_nonbond[WorkgroupSize + i].w);
		fprintf(forcefile1, "%d %f %f %f %f \n", i,
		        f4h_nonbond0[i].x + f4h_nonbond0[WorkgroupSize + i].x,
		        f4h_nonbond0[i].y + f4h_nonbond0[WorkgroupSize + i].y,
		        f4h_nonbond0[i].z + f4h_nonbond0[WorkgroupSize + i].z,
		        f4h_nonbond0[i].w + f4h_nonbond0[WorkgroupSize + i].w);
		fprintf(forcefile2, "%d %f %f %f %f \n", i,
		        f4h_nonbond1[i].x + f4h_nonbond1[WorkgroupSize + i].x,
		        f4h_nonbond1[i].y + f4h_nonbond1[WorkgroupSize + i].y,
		        f4h_nonbond1[i].z + f4h_nonbond1[WorkgroupSize + i].z,
		        f4h_nonbond1[i].w + f4h_nonbond1[WorkgroupSize + i].w);
	}

	fclose(forcefile0);
	fclose(forcefile1);
	fclose(forcefile2);
#endif
#endif

#ifdef PME_CALC
	double sumPME = 0.0f;
	double sumEwexcl = 0.0f;

	cudaMemcpy(ePME32, ePMEd, fsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(eEwexcl32, eEwexcld, NBPART * fsize, cudaMemcpyDeviceToHost);
#endif
	kinEnergy = 0.0f;
	potEnergy = 0.0f;
	sumelec = 0.0f;
	sumvdw = 0.0f;
	sumbond = 0.0f;
	sumangle = 0.0f;
	sumdihed = 0.0f;

#ifdef UREY_BRADLEY
	float sumureyb = 0.0f;
	cudaMemcpy(eureyb, eureybd, BPART * fsize, cudaMemcpyDeviceToHost);
#endif

#ifdef IMPROPER
	float sumimprop = 0.0f;
	cudaMemcpy(eimprop, eimpropd, BPART * fsize, cudaMemcpyDeviceToHost);
#endif
	/*
	for(n=0; n<nAtom; n++) {
		printf("CONSFIX: Segment %d Atom Segment %s ID %d \n", n,
		       seg_type[n], seg_typeid[n]);
	}
	*/

	for(n = 0; n < nAtom; n++) {
		vel.x = rv[n].x;
		vel.y = rv[n].y;
		vel.z = rv[n].z;
		vel.w = rv[n].w;

		// printf("%d %f, %f, %f, %f \n", n, rv[n].x, rv[n].y, rv[n].z, rv[n].w);

#ifdef PCONSTANT
		vel.x -= (r[n].x * boxVeloc.x / boxLength.x);
		vel.y -= (r[n].y * boxVeloc.y / boxLength.y);
		vel.z -= (r[n].z * boxVeloc.z / boxLength.z);
#endif
		kinEnergy += prm[type[n]].x *
		             (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
		potEnergy += f4h[n].w; //potential energy based on acceleration...

		for(int k = 0; k < NBPART; k++){
			potEnergy += f4h_nonbond[k * WorkgroupSize + n].w;
			sumelec += eelec[k * WorkgroupSize + n];
            

			sumvdw += evdw[k * WorkgroupSize + n];
		}

	  for(int k = 0; k < BPART; k++){
			potEnergy += f4h_bonded[k * WorkgroupSize + n].w;
			sumbond += ebnd[k * WorkgroupSize + n];
			sumangle += eang[k * WorkgroupSize + n];
			sumdihed += edihed[k * WorkgroupSize + n];
#ifdef PME_CALC
			sumEwexcl += eEwexcl32[k * WorkgroupSize + n];
#endif
#ifdef UREY_BRADLEY
			sumureyb += eureyb[k * WorkgroupSize + n];
#endif
#ifdef IMPROPER
			sumimprop += eimprop[k * WorkgroupSize + n];
#endif
		}
#ifdef PME_CALC
		sumPME += ePME32[n];
#endif
	}
#ifdef PCONSTANT
	scalarvirialT = VirialTensor.x + VirialTensor.y + VirialTensor.z;
	pressure = (68558.64100846418f /
	            (3.0f * boxLength.x * boxLength.y * boxLength.z)) *
	           (kineticE.x + kineticE.y + kineticE.z + scalarvirialT);
#endif

	kinEnergy *= 0.5;
#ifdef USE_CONSFIX
	temperature = 2 * kinEnergy / RC / (3 * nAtomwoseg - 6 - num_constraints);
#else
	temperature = 2 * kinEnergy / RC / (3 * nAtom - 6 - num_constraints);
#endif
	totEnergy = potEnergy + eEwself; // + kinEnergy
	//potential Energy based on physical properties...
	potEnergy = sumbond + sumangle + sumdihed + sumelec + sumvdw;

#ifdef UREY_BRADLEY
	potEnergy += sumureyb;
#endif

#ifdef IMPROPER
	potEnergy += sumimprop;
#endif

#ifdef PME_CALC
	//sumPME = (float) 0.5*sumPME*CC*FRAC_VOL;
	potEnergy += (sumPME + sumEwexcl + eEwself);
#endif

	if(add_prop || add_segidcoms){
		printout("DYNA>\t");
	}

////////////////////////////////////////
#ifdef REPRO // HIGHER OUTPUT PRECISION
////////////////////////////////////////
	printout("%-9d %-10.4f %f %f %f ",
	         stepCount, stepCount * DeltaT, temperature, sumbond, sumangle);
#ifdef UREY_BRADLEY
	printout("%f ", sumureyb);
#endif
#ifdef IMPROPER
	printout("%f ", sumimprop);
#endif
	printout("%f %f %f ", sumdihed, sumvdw, sumelec);
#ifdef PME_CALC
	printout("%f %f %f ", sumPME, sumEwexcl, eEwself);
#endif
#ifdef PCONSTANT
	printout("%f %f ", scalarvirialT, pressure);
#endif
	printout("%f %f %f", potEnergy, kinEnergy, totEnergy);
/////////////////////////////////
#else // NORMAL OUTPUT PRECISION
/////////////////////////////////
	printout("%-9d %-10.4f %-10.4f %-11.4f %-11.4f ",
	         stepCount, stepCount * DeltaT, temperature, sumbond, sumangle);
#ifdef UREY_BRADLEY
	printout("%-11.4f ", sumureyb);
#endif
#ifdef IMPROPER
	printout("%-8.4f ", sumimprop);
#endif
	printout("%-11.4f %-11.4f %-13.4f ", sumdihed, sumvdw, sumelec);
#ifdef PME_CALC
	printout("%-11.4f %-13.4f %-13.4f ", sumPME, sumEwexcl, eEwself);
#endif
#ifdef PCONSTANT
	printout("%-12.4f %-10.4f ", scalarvirialT, pressure);
#endif
	printout("%-13.4f %-13.4f %-13.4f", potEnergy, kinEnergy, totEnergy);
////////////////////////////////////////
#endif //REPRO
////////////////////////////////////////

#ifdef PROFILING
	printout("\t%d", nblist_call_count);
#endif
	printout("\n");

#ifdef PCONSTANT
	printout("PROP>\t");
	printout("%f, %f, %f\t|%f, %f, %f\t|%f, %f, %f\t|%f, %f, "
	         "%f\t|%f, %f, %f|\t%f\t|\t%d %d %d\n",
	         // frac = %f, %f, %f; numcells = %d, %d, %d\n",
	         boxAccel.x, boxAccel.y, boxAccel.z,
	         boxVeloc.x, boxVeloc.y, boxVeloc.z,
	         boxLength.x, boxLength.y, boxLength.z,
	         VirialTensor.x, VirialTensor.y, VirialTensor.z,
	         kineticE.x, kineticE.y, kineticE.z,
	         boxLength.x * boxLength.y * boxLength.z,
	         numcells.x, numcells.y, numcells.z);

	add_linebreak = 1;
#endif

	if(restraints){
		float3 com0;
		float3 com1;

		cudaMemcpy(&com0, com0d, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&com1, com1d, sizeof(float3), cudaMemcpyDeviceToHost);

		printout("COM> %f %f %f|\t%f %f %f|\t%f|\t%f\n", com0.x, com0.y, com0.z,
		         com1.x, com1.y, com1.z, mass_segid0, mass_segid1);

		add_linebreak = 1;
	}

	if(add_linebreak){
		printf("\n");
	}

#ifdef PROFILING
	nblist_call_count = 0;
#endif
	//int snapshot=0;
	//TakeSnapshot(snapshot);
	//Write_xyz();
}

//------------------------------------------------------------------------------
void print_output_headers(){
//------------------------------------------------------------------------------
	char add_prop = 0;
	char add_segidcoms = 0;

#ifdef PCONSTANT
	add_prop = 1;
#endif

	if(restraints){
		add_segidcoms = 1;
	}

	if(add_prop || add_segidcoms){
		printout("DYNA>\t");
	}

	printout("%-9s %-10s %-10s %-11s %-11s ", "Step","Time (ps)",
	         "Temp (K)","bond","angle");
#ifdef UREY_BRADLEY
	printout("%-11s ", "urey-b");
#endif
#ifdef IMPROPER
	printout("%-8s ", "impr");
#endif
	printout("%-11s %-11s %-13s ", "dihed","VDW","elec");
#ifdef PME_CALC
	printout("%-11s %-13s %-13s " ,"PME", "Ewexcl", "Ewself");
#endif
#ifdef PCONSTANT
	printout("%-12s %-10s ", "virial", "pressure");
#endif
	printout("%-13s %-13s %-13s\n", "sum PE","Kinetic","Potential");

#ifdef PCONSTANT
	printout("PROP>\t");
	printout("BoxAcc.x, BoxAcc.y, BoxAcc.z\t|\tBoxVel.x, BoxVel.y, "
	         "BoxVel.z\t|\tBoxLen.x, BoxLen.y, BoxLen.z");
	printout("\t|\tVirial.x, Virial.y, Virial.z\n");
#endif

	if(add_segidcoms){
		printout("COM>\t");
		printout("SEGID0.x, SEGID0.y, SEGID0.z | SEGID1.x, SEGID1.y, "
		         "SEGID1.z | MASS_SEGID0 | MASS_SEGID1\n");
	}

	if(add_prop || add_segidcoms){
		printout("\n");
	}

	return;
}
//------------------------------------------------------------------------------

