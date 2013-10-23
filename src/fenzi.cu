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

//#define DIAGNOSE
//#define DIHED_DEBUG
//#define BOND_DEBUG
//#define ANGLE_DEBUG
//#define DEBUG_PME
//#define DEBUG_NONB
//#define DEBUG_NBLIST
//#define NUM_DEBUG_CELL    7

#include "globals.h"
#include "globals.cu"

#include <cufft.h>
#include "md.h"
//#include "cucomplexops.h"
#include "io/ffile.h"
#include "io/charmmio.h"

//only include if need to inspect GPU variables..
//otherwise comment out for performance..
//#include "diagnosis.c" //functions to observe and save device variables..

void Check_Static_Parameters(char);

#include "mem.h"
#include "timer.h"
#include "mdcuda.h"
#include "prmwrite.h"
#include "prmread.h"
#include "evalprops.h"
#include "dynamics.h"

#ifdef USE_NPT
#include "npt_kernels.h"
#endif

#include "init.h"


/**
 * Parse all commnand line arguments
 * -f sets the input file .in
 * -d sets the device number to use, defaults to 0
 */
void parseArgs(int argc, char** argv, char** input_filename, int* devicenum){
	bool flaginputexist = false;

	// If there any arguments start parsing else print instructions and exit
	if(argc > 1){

		int i;
		//loop through all arguments
		for(i = 1; i < argc; i++){

			if(argv[i][0] == '-'){

				switch(argv[i][1]){
				case 'f':
					flaginputexist = true;
					i++;
					*input_filename = argv[i];
					break;
				case 'd':
					i++;
					*devicenum = atoi(argv[i]);
					break;
				}
			}
		}

		cudaError_t err = cudaSetDevice(*devicenum);

		// check for valid device, if invalid print error and exit
		if(err == cudaErrorInvalidDevice){
			printf("Invalid Device: %d\n", *devicenum);
			exit(0);
		}
		else{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, *devicenum);
			printf("\nDevice %d: %s\n", *devicenum, deviceProp.name);
		}
	}
	else{
		printf("Please use '-d' for device and '-f' for the input filename\n");
		exit(0);
	}

	// check if input file was specified, if not print instructions and exit
	if(!flaginputexist){
		printf("Please give an input filename using '-f filename'\n");
		exit(0);
	}

	return;
}

int main(int argc, char** argv){

#ifdef PROFILING
	double cpu_singlestep = 0;
	double cpu2 = 0;
#endif

	int devicenum = 0;
	char restart;
	int i;

	char* input_filename;
	char checkpointfilename[512] = "";
	char logfileName[512];
	char outFileName[512]; // File name to output energies

    // parse the command line arguments
    parseArgs(argc, argv, &input_filename, &devicenum);

	//trajectory allocation
	char trajFileNamePrefix[600] = "";
	charmm_trj charmm_trj_obj;
	char charmm_trj_title[80];
	FFILE* f;

	// parse the command line arguments
	parseArgs(argc, argv, &input_filename, &devicenum);

	// parse .in file
	ReadInput(input_filename);

#if (VSWITCH && VSHIFT)
	printf("Error: Enable only one of VSHIFT or VSWITCH\n");
	exit(0);
#endif

	//set up other input files and output files
	// sprintf(filename_prefix, "%s", input_filename);
	sprintf(filename_prefix, "%s", trajFileName);


	//truncate the filename prefix to the last '.' after the last '/'
	char* tempptr1 = strrchr(filename_prefix, '.');
	char* tempptr2 = strrchr(filename_prefix, '/');
	if (tempptr1 > tempptr2){
		*tempptr1 = '\0'; // terminate string with null character
	}

	//generate check point file and log file names
	sprintf(checkpointfilename, "%s.rst", filename_prefix);
	sprintf(logfileName, "%s_dev%d.log", filename_prefix, devicenum);

	printf("----------------------------------------" \
	       "--------------------------------------\n");
	printf("See logfile :'%s' for Molecular System Parameters\n", logfileName);
	sprintf(outFileName, "%s_dev%d.out", filename_prefix, devicenum);
	logfile = fopen(logfileName, "w");
	outfile = fopen(outFileName,"a");

	//restart from checkpointfile...
	if(restart = accessfile(checkpointfilename)){
		if(printtostdout == 0){
			printf("Adding to OutputFile: %s\n", outFileName);
		}
		outfile = fopen(outFileName, "a");

		LoadCheckpoint(checkpointfilename);
		stepCount = CheckpointTimestep;
		StartCount = CheckpointTimestep;
	}
	// new simulation
	else{
		if(printtostdout == 0){
			printf("Writing to OutputFile: %s\n", outFileName);
		}
		outfile = fopen(outFileName, "w");
		LoadCoord();
	}

	// initialize memory
	InitHostMem();

	// read PSF and PRM files
	ReadPsf_CHARMM();
	ReadPrm_CHARMM();

	if(shake){
		build_constraints();
	}

	GenExcl_minimal_new();
	InitMem();
	InitDeviceConstants();  // initialize various precomputed device constants...

	if(!restart){
		InitVelocity(); //for velocity reassignment prm.x is required...
		                //masses need to be initialized.
		cudaMemset(f4d, 0, f4size);
	}


	printf("-------------------------\nTotalM : %7.2f MB\n-----"
	       "--------------------\n",
	       ((float)device_memory_usage) / MEGA_BYTE);

	printf("Constant\t\tActual\tCurr\tPerf\n");
	printf("--------------------------------------------------\n");
	printf("MAXTYP\t\t\t%d\t"EXPAND(MAXTYP)"\t"EXPAND(MAXTYP)"\n", Num_NBTypes);
	printf("MAX_BOND_TYPE\t\t%d\t"EXPAND(MAX_BOND_TYPE)
	       "\t"EXPAND(MAX_BOND_TYPE)"\n",
	       Num_BondTypes);

	printf("MAX_ANGLE_TYPE\t\t%d\t"EXPAND(MAX_ANGLE_TYPE)
	       "\t"EXPAND(MAX_ANGLE_TYPE)"\n",
	       Num_AngleTypes);

	printf("MAX_DIHED_TYPE\t\t%d(*)\t"EXPAND(MAX_DIHED_TYPE)
	       "\t"EXPAND(MAX_DIHED_TYPE)"\n",
	       Num_DihedTypes);

	printf("--------------------------------------------------\n");
	printf("(*) - Includes Folds\n");
	printf("--------------------------------------------------\n");

	Check_Static_Parameters(1);

#ifdef PME_CALC
	// if PME, generate the lattice for charges
	CallLatticeBuild();
#endif

#ifdef PROFILING
	cpu_singlestep = ((double)clock());
#endif

	// generate the list of atoms for nonbonded force calculation
	// build cells
	CallCellBuild();

	/* NO IDEA WHAT ANY OF THIS DOES
	print_mass_charge_type();
	capture_position(2,StartCount);
	// 0 - write xyz file, 2-write position and charges
	check_capture_nbcell();
	TakeSnapshot(0);
	*/

	// build nonbond list
	BuildNBGPU();

#ifdef PROFILING
	cpu2 += ((double)clock() - cpu_singlestep) / CLOCKS_PER_SEC;
	cpu_singlestep = ((double)clock());
#endif

	// compute forces on atoms - step 0
	ComputeAccelGPU();

#ifdef PCONSTANT
	ScanVirial();
#endif

#ifdef PROFILING
	cpu2 += ((double)clock() - cpu_singlestep) / CLOCKS_PER_SEC;
#endif

//------------------Open Trajectory File and Write Headers-------------------//
	// PrintLevel > 1: print trajectory
	if(PrintLevel > 1){
		fprintf(logfile, "Open trajectory file and write headers...\n");

		//extract trajectory file prefix and suffix
		sprintf(trajFileNamePrefix, "%s", trajFileName);

		//truncate the trajectory filename prefix to the last '.' after the last '/'
		tempptr1 = strrchr(trajFileNamePrefix, '.');
		tempptr2 = strrchr(trajFileNamePrefix, '/');
		if (tempptr1 > tempptr2){
			*tempptr1 = '\0'; // terminate string with null character
		}

		printf("-------------------------------------" \
		       "-----------------------------------------\n");
		printf("Trajectory prefix :%s\n", trajFileNamePrefix);

		sprintf(trajFileName, "%s_dev%d.dcd", trajFileNamePrefix, devicenum);

		f = ffopen(trajFileName, "wb");

		/* initalize charmm trajectory data */
		charmm_trj_init(&charmm_trj_obj, f);

		memset(charmm_trj_title, ' ', 80 * sizeof(char));

		charmm_trj_obj.nframes = StepLimit / TrjFreq;
		charmm_trj_obj.initial_step_index = TrjFreq;
		charmm_trj_obj.nsteps = StepLimit;
		charmm_trj_obj.output_frequency = TrjFreq;
		charmm_trj_obj.ntitles = 1;
		charmm_trj_obj.natoms = nAtom;
		charmm_trj_obj.ndof = 1;
		charmm_trj_obj.qcrys = 1;
		charmm_trj_obj.qdim4 = 0;
		charmm_trj_obj.qcg = 0;
		charmm_trj_obj.version = 34;
		charmm_trj_obj.time_step = DeltaT / 4.888821E-2;

		sprintf(charmm_trj_title, "%s", "Generated by FENZI");
		charmm_trj_obj.titles = charmm_trj_title;

			/*
#ifdef USE_CONSFIX
		charmm_trj_obj.nfixed = 0;
		for(i = 0; i < nAtom; i++){
			if(seg_typeid[i] < 0){
				(charmm_trj_obj.nfixed)++;
			}
		}

		charmm_trj_obj.freeind = (uint32_t*)malloc((nAtom - charmm_trj_obj.nfixed) *
		                                         sizeof(charmm_trj_obj.freeind[0]));

		int j = 0;
		for(i = 1; i <= nAtom; i++, j++){
			if(seg_typeid[i] >= 0){
				charmm_trj_obj.freeind[j] = i;
			}
		}
#endif
			*/


		charmm_trj_write_header(&charmm_trj_obj);
		charmm_trj_write_titles(&charmm_trj_obj);
		charmm_trj_write_natoms(&charmm_trj_obj);
		charmm_trj_write_freelist(&charmm_trj_obj);

		charmm_trj_obj.X = (float*)malloc(nAtom * sizeof(charmm_trj_obj.X[0]));
		charmm_trj_obj.Y = (float*)malloc(nAtom * sizeof(charmm_trj_obj.Y[0]));
		charmm_trj_obj.Z = (float*)malloc(nAtom * sizeof(charmm_trj_obj.Z[0]));
		// charmm_trj_obj.C = (float *)malloc (nAtom*sizeof (charmm_trj_obj.C[0]));

		/* DONE -- initialize charmm trajectory data */
		/*
		for(i=0; i<nAtom; ++i) {
				charmm_trj_obj.X[i] = r[i].x;
				charmm_trj_obj.Y[i] = r[i].y;
				charmm_trj_obj.Z[i] = r[i].z;
				// charmm_trj_obj.C[i] = r[i].w;
		}

		charmm_trj_obj.celltmp[0] = (double)Region.x;
		charmm_trj_obj.celltmp[2] = (double)Region.y;
		charmm_trj_obj.celltmp[5] = (double)Region.z;

		charmm_trj_write_frame (&charmm_trj_obj);
		*/
	}

//------------------End Open Trajectory File and Write Headers---------------//

//------------------Print Energies Step 0------------------------------------//
	// PrintLevel > 0: PrintLevel is set in .in file; print energies step 0
	if(PrintLevel > 0){

		fprintf(logfile, "Print energies at step 0...\n");
		if(MinimizeStepLimit > 0){
			printf("\n-------------------------\nMinimizing Structure via"
			       " conjugated gradient...\n-------------------------\n");
			fprintf(logfile, "Minimizing Structure...\n");
		}
		else{
			printf("\n-------------------------\nContinuing dynamics...\n"
			       "-------------------------\n");
			fprintf(logfile, "Continuing Dynamics...\n");
			StartCount++;
		}
		print_output_headers();
		EvalProps();

	}

//------------------End Print Energies Step 0---------------------------------//

//------------------Minimize Structure----------------------------------------//
	for(stepCount = 0; stepCount < MinimizeStepLimit; stepCount++){

		fprintf(logfile, "Minimize structure...\n");
		MinimizeStep();

		if(!(stepCount % 100) && (stepCount > 0)){
			if(minimization_method == 1){
				cudaMemcpy(&cgfac, cgfacd, sizeof(float), cudaMemcpyDeviceToHost);
				cgfac *= (cgrate * 100.0f);
				cudaMemcpy(cgfacd, &cgfac, sizeof(float), cudaMemcpyHostToDevice);
			}
			else if(minimization_method == 2){
				cudaMemcpy(&sdfac, sdfacd, sizeof(float), cudaMemcpyDeviceToHost);
				sdfac *= (sdrate * 100.0f);
				cudaMemcpy(sdfacd, &sdfac, sizeof(float), cudaMemcpyHostToDevice);
			}

			printf("CGFAC = %e, SDFAC = %e\n", cgfac, sdfac);
		}

		if(!minimization_method){
			break;
		}

	}

	if(MinimizeStepLimit > 0){
		char tempfilename[100];
		fprintf(logfile, "Tune minimization parameters for further"
		        " minimization...\n\n");
		sprintf(tempfilename, "%s_min%d.xyz", filename_prefix, stepCount);
		Write_xyz(tempfilename, 1);
		stepCount = StartCount;
		StartCount++;
		InitVelocity();

		ComputeAccelGPU();
		printf("\n-------------------------\nContinuing dynamics...\n"
		       "-------------------------\n");
		print_output_headers();
		EvalProps();
	}

	//------------------Continue Dynamics--------------------------//
	double timer_start = clock();
	for(stepCount = StartCount; stepCount <= StepLimit; stepCount++){

#ifdef PROFILING
		tic();
#endif

#ifdef USE_NPT
// shake + constant pressure and constat temperature with extended Hamiltonian
		SingleStep_npt();
#else
//constat temparature with velocity scaling [and shake]
//constant energy [and shake];
		SingleStep();
#endif

#ifdef PROFILING
		cpu_singlestep =  toc();
		cpu2 += cpu_singlestep;
#endif

//------------------Reset Velocity-------------------------------------------//
		if(stepCount % VAssign == 0){
			//reset velocity for konstant temperature simulations
#ifdef PCONSTANT
			cudaMemcpy(r, r4d, f4size, cudaMemcpyDeviceToHost);
			cudaMemcpy(&Region, boxLengthd, sizeof(float4), cudaMemcpyDeviceToHost);
			cudaMemcpy(&RegionVeloc, boxVelocd, sizeof(float4),
			           cudaMemcpyDeviceToHost);
#endif
			fprintf(logfile, "Reset velocity at step %d\n", stepCount);
			InitVelocity();
		}

//------------------End Reset Velocity---------------------------------------//

//------------------Checkpoint Saving and Reloading--------------------------//
		if((stepCount % StepStore == 0) && (PrintLevel > 0)){

			fprintf(logfile, "Checkpoint coordiantes at step %d\n", stepCount);
			cudaMemcpy(r, r4d, f4size, cudaMemcpyDeviceToHost);
			cudaMemcpy(rv, v4d, f4size, cudaMemcpyDeviceToHost);

#ifdef PCONSTANT
			cudaMemcpy(&Region, boxLengthd, sizeof(float4), cudaMemcpyDeviceToHost);
			cudaMemcpy(&RegionVeloc, boxVelocd, sizeof(float4),
			           cudaMemcpyDeviceToHost);
#endif

/*
			if (keepcheckpoint){
				sprintf(restartfilename, "%s_%d_dev%d.rst", filename_prefix,
				        stepCount, devicenum);
				SaveCheckpoint(restartfilename);
			}
*/
			SaveCheckpoint(checkpointfilename);
			LoadCheckpoint(checkpointfilename);

			cudaMemcpy(r4d, r, f4size, cudaMemcpyHostToDevice);
			cudaMemcpy(v4d, rv, f4size, cudaMemcpyHostToDevice);

#ifdef PCONSTANT
			cudaMemcpy(boxLengthd, &Region, sizeof(float4), cudaMemcpyHostToDevice);
			cudaMemcpy(boxVelocd, &RegionVeloc, sizeof(float4),
			           cudaMemcpyHostToDevice);
#endif

		}

//------------------End Checkpoint Saving and Reloading----------------------//

//------------------Trajectory Saving----------------------------------------//
		if((PrintLevel > 1) && (stepCount % TrjFreq == 0)){
		//print out coordinates to traj file...

			for(i = 0; i < nAtom; i++) {
				charmm_trj_obj.X[i] = r[i].x;
				charmm_trj_obj.Y[i] = r[i].y;
				charmm_trj_obj.Z[i] = r[i].z;
				// charmm_trj_obj.C[i] = r[i].w;
			}

			charmm_trj_obj.celltmp[0] = (double)Region.x;
			charmm_trj_obj.celltmp[2] = (double)Region.y;
			charmm_trj_obj.celltmp[5] = (double)Region.z;

			charmm_trj_write_frame (&charmm_trj_obj);
		}

//------------------End Trajectory Saving-------------------------------------//

//------------------Print Energies--------------------------------------------//
		if((PrintLevel > 0) && (stepCount%StepAvg == 0)){

#ifdef PROFILING
			cpu1 = clock();
#endif

			EvalProps();

#ifdef PROFILING
			cudaThreadSynchronize();
			profile_times[EVALPROPS] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif

		}

//------------------End Print Energies---------------------------------------//

	} //for(stepCount=StartCount; stepCount<=StepLimit; stepCount++)

//------------------End Dynamics---------------------------------------------//

//------------------Summarize Overall Performance and Close Files------------//

	double timer_lapse = ((double)clock() - timer_start) / CLOCKS_PER_SEC;
	//fprintf(outfile, "Execution time (s) = %g\n" , cpu2 - cpu1);
	double timeperstep = timer_lapse / (StepLimit - StartCount);
	fprintf(logfile, "Printed time =  %12.6g, Avg time/step = %12.6g(ms),"
	        " Ns per day = %12.6g\n",
	        timer_lapse, 1000 * timeperstep, (86.4 / timeperstep) * DeltaT);
	printf("Printed time =  %12.6g, Avg time/step = %12.6g(ms),"
	       " Ns per day = %12.6g\n",
	       timer_lapse, 1000 * timeperstep, (86.4 / timeperstep) * DeltaT);

	if((StartCount >= StepLimit) && (restart)){
		printf("\n\nDid you mean to remove the previously saved checkpoint"
		       " file %s?\n",
		       checkpointfilename);
	}

#ifdef PROFILING
	printf("Total GPU Time(s),%12.6g\n\n", cpu2);
	printf("Bond,%12.6g\n", profile_times[BOND]);
	printf("Angle,%12.6g\n", profile_times[ANGLE]);
	printf("Dihed,%12.6g\n", profile_times[DIHED]);
	printf("NonBondForce,%12.6g\n", profile_times[NONBOND]);
	printf("NonBondList,%12.6g\n", profile_times[NBBUILD]);
	printf("CellBuild,%12.6g\n", profile_times[CELLBUILD]);
	printf("CellUpdate,%12.6g\n", profile_times[CELLUPDATE]);
	printf("CellClean,%12.6g\n", profile_times[CELLCLEAN]);
	printf("ChargeSpread,%12.6g\n", profile_times[CHARGESPREAD]);
	printf("BCMultiply,%12.6g\n", profile_times[BCMULTIPLY]);
	printf("PMEForce,%12.6g\n", profile_times[PMEFORCE]);
	printf("LatticeBuild,%12.6g\n", profile_times[LATTICEBUILD]);
	printf("HalfKick,%12.6g\n", profile_times[HALFKICK]);
	printf("CoordsUpdate,%12.6g\n", profile_times[COORDSUPDATE]);
	printf("LatticeUpdate,%12.6g\n", profile_times[LATTICEUPDATE]);
	printf("Reduce,%12.6g\n", profile_times[REDUCE]);
	printf("Shake,%12.6g\n", profile_times[CONSTRAINTS]);
	printf("CUFFT,%12.6g\n", profile_times[CUDAFFT]);
	printf("Evalprops,%12.6g\n", profile_times[EVALPROPS]);

	double kernel_times = 0.0f;
	kernel_times = profile_times[HALFKICK] + profile_times[COORDSUPDATE];
	kernel_times += profile_times[NBBUILD] + profile_times[LATTICEUPDATE];
	kernel_times += profile_times[BOND] + profile_times[ANGLE];
	kernel_times += profile_times[DIHED] + profile_times[NONBOND];
	kernel_times += profile_times[CHARGESPREAD] + profile_times[PMEFORCE];
	kernel_times += profile_times[CUDAFFT] + profile_times[BCMULTIPLY];
	kernel_times += profile_times[CELLUPDATE] + profile_times[CELLCLEAN];
	kernel_times += profile_times[REDUCE] + profile_times[CONSTRAINTS];
	kernel_times += profile_times[CELLBUILD];

	printf("\n\nOther,%g\n", (cpu2 - kernel_times));
#endif

	fclose(outfile);
	fclose(logfile);

	if(PrintLevel > 1){
		free(charmm_trj_obj.X);
		free(charmm_trj_obj.Y);
		free(charmm_trj_obj.Z);
		// free(charmm_trj_obj.C);

			/*
#ifdef USE_CONSFIX
		free(charmm_trj_obj.freeind);
#endif
			*/

		charmm_trj_free(&charmm_trj_obj);
	}

	//ffclose(f);

	exit(0);
}
