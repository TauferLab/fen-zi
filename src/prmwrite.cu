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
#include "globals.h"


//----------------------------------------------------------------------------
int accessfile(char filename[512]){
//----------------------------------------------------------------------------
// check to see if file is accessible...
//----------------------------------------------------------------------------
	FILE* filep;
	char status=0;
	if(filep = fopen(filename, "r")){
		status = 1;
		fclose(filep);
	}
	return status;
}

//----------------------------------------------------------------------------
void SaveCheckpointBinary(char restartfilename[512]){
//----------------------------------------------------------------------------
//write velocity in checkpointing file
//----------------------------------------------------------------------------
	int n;

	FILE* restartfile;

	/* Open an MD-configuration file */
	restartfile = fopen(restartfilename, "wb");

	fwrite(&nAtom, sizeof(nAtom), 1, restartfile);
	fwrite(&stepCount, sizeof(stepCount), 1, restartfile);
	fwrite(&seed, sizeof(seed), 1, restartfile);
	fwrite(&Region, sizeof(float3), 1, restartfile);

#ifdef PCONSTANT
	fwrite(&RegionVeloc, sizeof(float3), 1, restartfile);
#else
	float3 Val_Zero = {0.0f, 0.0f, 0.0f};
	fwrite(&Val_Zero, sizeof(Val_Zero), 1, restartfile);
#endif

	for(n = 0; n < nAtom; n++){
		//explicitly specify the number of bytes to copy...
		fwrite(&r[n].x, sizeof(float), 1, restartfile);
		fwrite(&r[n].y, sizeof(float), 1, restartfile);
		fwrite(&r[n].z, sizeof(float), 1, restartfile);

		fwrite(&rv[n].x, sizeof(float), 1, restartfile);
		fwrite(&rv[n].y, sizeof(float), 1, restartfile);
		fwrite(&rv[n].z, sizeof(float), 1, restartfile);
	}

	fclose(restartfile);

}

//----------------------------------------------------------------------------
void LoadCheckpointBinary(char restartfilename[512]){
//----------------------------------------------------------------------------
//write velocity in checkpointing file
//----------------------------------------------------------------------------
	int n;

	FILE* restartfile;
	float4 Val_Zero = {0.0f, 0.0f, 0.0f, 0.0f};

	//Open an MD-configuration file
	restartfile = fopen(restartfilename, "rb");

	fprintf(logfile, "Restarting from file: %s at step %d...",
	        restartfilename, stepCount);

	fread(&nAtom, sizeof(nAtom), 1, restartfile);
	fread(&CheckpointTimestep, sizeof(CheckpointTimestep), 1, restartfile);
	fread(&seed, sizeof(seed), 1, restartfile);

	fread(&Region, sizeof(float3), 1, restartfile);

	RegionH.x = Region.x / 2;
	RegionH.y = Region.y / 2;
	RegionH.z = Region.z / 2;

#ifdef PCONSTANT
	fread(&RegionVeloc, sizeof(float3), 1, restartfile);
#else
	fread(&Val_Zero, sizeof(float3), 1, restartfile);
	if((Val_Zero.x != 0.0) || (Val_Zero.y != 0.0) || (Val_Zero.z != 0.0)){
		printf("Warning!!!, the system may not have been equilibrated in NVT...\n");
	}

	Val_Zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif

	for(n = 0; n < nAtom; n++){
		//explicitly specify the number of bytes to copy...
		fread(&r[n].x, sizeof(float), 1, restartfile);
		fread(&r[n].y, sizeof(float), 1, restartfile);
		fread(&r[n].z, sizeof(float), 1, restartfile);

		fread(&rv[n].x, sizeof(float), 1, restartfile);
		fread(&rv[n].y, sizeof(float), 1, restartfile);
		fread(&rv[n].z, sizeof(float), 1, restartfile);

		f4h[n] = Val_Zero;

		if(feof(restartfile)){
			printf("Unexpected end of file %s\n", restartfile);
			exit(-1);
		}
	}

	for(n = nAtom; n < NMAX; n++){
		f4h[n] = Val_Zero;
		rv[n] = Val_Zero;
	}

	fclose(restartfile);

}

//----------------------------------------------------------------------------
void SaveCheckpointAscii(char restartfilename[512]){
//----------------------------------------------------------------------------
//write velocity in checkpointing file
//----------------------------------------------------------------------------
	int n;

	FILE* restartfile;

	/* Open an MD-configuration file */
	restartfile = fopen(restartfilename, "w");

	fprintf(restartfile, "%12d \n", nAtom);
	fprintf(restartfile, "%12d \n", stepCount);
	fprintf(restartfile, "%le \n", seed);

	fprintf(restartfile, "%12f %12f %12f ", Region.x, Region.y, Region.z);

#ifdef PCONSTANT
	fprintf(restartfile, "%12f %12f %12f",
	        RegionVeloc.x, RegionVeloc.y, RegionVeloc.z);
#else
	float3 Val_Zero = {0.0f, 0.0f, 0.0f};
	fprintf(restartfile, "%12f %12f %12f", Val_Zero.x, Val_Zero.y, Val_Zero.z);
#endif
	fprintf(restartfile, "\n");

	// Write coordinates
	for(n = 0; n < nAtom; n++){
		fprintf(restartfile, "%10f %10f %10f %10f %10f %10f\n",
		        r[n].x, r[n].y, r[n].z, rv[n].x, rv[n].y, rv[n].z);
	}

	fclose(restartfile);

}

//----------------------------------------------------------------------------
void LoadCheckpointAscii(char restartfilename[512]){
//----------------------------------------------------------------------------
//write velocity in checkpointing file
//----------------------------------------------------------------------------
	int n;
	char line[9999];
	char delims[] = " \t\n\r";
	float4 Val_Zero = {0.0f, 0.0f, 0.0f, 0.0f};

	FILE* restartfile;

	if(!(restartfile = fopen(restartfilename, "r"))){
		printf("File %s not found\n", restartfilename);
		exit(-1);
	}

	fprintf(logfile, "Restarting from file: %s at step %d...\n",
	        restartfilename, stepCount);
	fgets(line, 9999, restartfile);
	nAtom = (int)atoi(strtok(line, delims));

	fgets(line, 9999, restartfile);
	CheckpointTimestep = (int)atoi(strtok(line, delims));

	fgets(line, 9999, restartfile);
	seed = (double)atof(strtok(line, delims));

	fgets(line, 9999, restartfile);

	Region.x = (float)atof(strtok(line, delims));
	RegionH.x = Region.x / 2;
	Region.y = (float)atof(strtok(NULL, delims));
	RegionH.y = Region.y / 2;
	Region.z = (float)atof(strtok(NULL, delims));
	RegionH.z = Region.z / 2;

#ifdef PCONSTANT
	RegionVeloc.x = (float)atof(strtok(NULL, delims));
	RegionVeloc.y = (float)atof(strtok(NULL, delims));
	RegionVeloc.z = (float)atof(strtok(NULL, delims));
#else
	Val_Zero.x = (float)atof(strtok(NULL, delims));
	Val_Zero.y = (float)atof(strtok(NULL, delims));
	Val_Zero.z = (float)atof(strtok(NULL, delims));
	if((Val_Zero.x != 0.0) || (Val_Zero.y != 0.0) || (Val_Zero.z != 0.0)){
		printf("Warning!!!, the system may not have been equilibrated in NVT...\n");
	}

	Val_Zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif


	//Read coordinates
	for(n = 0; n < nAtom; n++){
		fgets(line, 9999, restartfile);

		r[n].x = (float)atof(strtok(line, delims));
		r[n].y = (float)atof(strtok(NULL, delims));
		r[n].z = (float)atof(strtok(NULL, delims));

		rv[n].x = (float)atof(strtok(NULL, delims));
		rv[n].y = (float)atof(strtok(NULL, delims));
		rv[n].z = (float)atof(strtok(NULL, delims));

		// printf("%d %f, %f, %f, %f \n", n, rv[n].x, rv[n].y, rv[n].z, rv[n].w);


		f4h[n] = Val_Zero;

		if(feof(restartfile)){
			printf("Unexpected end of file %s\n", restartfile);
			exit(-1);
		}
	}

	for(n = nAtom; n < NMAX; n++){
		f4h[n] = Val_Zero;
		rv[n] = Val_Zero;
	}

	fclose(restartfile);
}

float mag_float3(float4 a, float4 b){
	return sqrt((a.x - b.x) * (a.x - b.x) +
	            (a.y - b.y) * (a.y - b.y) +
	            (a.z - b.z) * (a.z - b.z));
}

//----------------------------------------------------------------------------
void Write_xyz(char* tempfilename, char pbc){
//pbc, periodic boundary conditions...
//----------------------------------------------------------------------------

	float4* r4;
	r4 = (float4*)malloc(f4size);//for debugging purposes...
	cudaMemcpy(r4, r4d, f4size, cudaMemcpyDeviceToHost);

	FILE* xyzfile;

	char filename[100];

	sprintf(filename, "%s", tempfilename);
	//sprintf(filename,"%s_min_%d.xyz", filename_prefix, stepCount);
	printf("---------------------------------\nWriting structure to %s\n-"
	       "--------------------------------\n", filename);

	xyzfile = fopen(filename, "w");
	fprintf(xyzfile, "%d\nMEMBRANE\n",nAtom);

	for(int n = 0;n < nAtom; n++){
		if(pbc){
			r4[n].z = (r4[n].z > Region.z / 2.0f) ? (r4[n].z - Region.z) : r4[n].z;
			r4[n].y = (r4[n].y > Region.y / 2.0f) ? (r4[n].y - Region.y) : r4[n].y;
			r4[n].x = (r4[n].x > Region.x / 2.0f) ? (r4[n].x - Region.x) : r4[n].x;
		}
		fprintf(xyzfile, "%s\t%f\t%f\t%f\n",
		        atom_type[n], r4[n].x, r4[n].y, r4[n].z);
	}

	fclose(xyzfile);

	free(r4);
}

//----------------------------------------------------------------------------
void write_forces(char* tempfilename, int countstep){
// starting
//----------------------------------------------------------------------------

	char filename[100];
	sprintf(filename, "%s_%d", tempfilename, countstep);
	FILE* forcefile = fopen(tempfilename, "a");

	fprintf(forcefile, "ComputeAccel: Step %d \n", StartCount);
	cudaMemcpy(r, r4d, f4size, cudaMemcpyDeviceToHost);
	cudaMemcpy(rv, v4d, f4size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(f4h, f4d, f4size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(f4h_bonded, f4d_bonded, f4size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(f4h_nonbond, f4d_nonbond, f4size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < nAtom; i++) {
		fprintf(forcefile, "r          : %d %f %f %f %f \n",
		        i, r[i].x, r[i].y, r[i].z, r[i].w);

		fprintf(forcefile, "rv         : %d %f %f %f %f \n",
		        i, rv[i].x, rv[i].y, rv[i].z, rv[i].w);

		// fprintf(testfile, "f4h        : %d %f %f %f %f \n",
		//         i, f4h[i].x, f4h[i].y, f4h[i].z, f4h[i].w);

		// fprintf(testfile, "f4h_bonded : %d %f %f %f %f \n",
		//         i, f4h_bonded[i].x, f4h_bonded[i].y,
		//         f4h_bonded[i].z, f4h_bonded[i].w);

		// fprintf(testfile, "f4h_nonbond: %d %f %f %f %f \n",
		//         i, f4h_nonbond[i].x, f4h_nonbond[i].y,
		//         f4h_nonbond[i].z, f4h_nonbond[i].w);

		// fprintf(testfile, "energies   : %d %f %f\n\n", i, evdw[i], eelec[i]);
	}

	fclose(forcefile);
}

//----------------------------------------------------------------------------
void print_mass_charge_type(){
//----------------------------------------------------------------------------

	FILE* mass_file;
	FILE* charge_file;
	FILE* type_file;

	char mass_filename[600];
	char charge_filename[600];
	char type_filename[600];

	sprintf(mass_filename, "%s_mass.dat", filename_prefix);
	sprintf(charge_filename, "%s_charge.dat", filename_prefix);
	sprintf(type_filename, "%s_type.dat", filename_prefix);

	mass_file = fopen(mass_filename, "w");
	charge_file = fopen(charge_filename, "w");
	type_file = fopen(type_filename, "w");

	float4* r4;
	r4 = (float4*)malloc(f4size);//for debugging purposes...
	cudaMemcpy(r4, r4d, f4size, cudaMemcpyDeviceToHost);

	for(int n = 0; n < nAtom; n++){
		fprintf(mass_file, "%f\n", prm[type[n]].x);
		fprintf(charge_file, "%f\n", r4[n].w);
		fprintf(type_file, "%s\n", atom_type[n]);
	}

	fclose(mass_file);
	fclose(charge_file);
	fclose(type_file);

	free(r4);

	printf("Charge, Mass and Atom type printed\n");

	exit(0);
}
