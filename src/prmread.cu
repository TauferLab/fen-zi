/******************************************************************************/
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
//function to perform case insensitive string comparison...
int cicompare(char* input, char* compstr){
//----------------------------------------------------------------------------
	int i = 0;
	//printf("%s\t%s", input, compstr);
	while(input[i] && compstr[i]){
		if(toupper(input[i]) == toupper(compstr[i])){
			i++;
		}

		if((int)toupper(input[i])> (int)toupper(compstr[i])){
			return 1;
		}// {printf("\t%d\t1\n",i+1); return 1;}

		if((int)toupper(input[i]) < (int)toupper(compstr[i])){
			return -1;
		}// {printf("\t%d\t-1\n",i+1); return -1;}
	}

	if((input[i]) && (!compstr[i])){
		return 1;
	}//{printf("\t%d\t1\n",i+1); return 1;}

	if((!input[i]) && (compstr[i])){
		return -1;
	}//{printf("\t%d\t-1\n",i+1); return -1;}

	if((!input[i]) && (!compstr[i])){
		return 0;
	}//{printf("\t%d\t0\n",i+1); return 0;}

	//return a value out of the range -1, 0, 1,
	//to indicate the control path is not valid...
	return -2;
}

//----------------------------------------------------------------------------
int addtype(char* curr_type){
//add the type of atoms present to types_present array...
//if already present do nothing
//----------------------------------------------------------------------------
	int i = 0;
	while(types_present[i][0]){
		if(!strcmp(types_present[i], curr_type)){
			return i;
		}
		i++;
	}
	sprintf(types_present[i], "%s", curr_type);
	return i;
}

//----------------------------------------------------------------------------
int typetonum_present(char* curr_type){
//----------------------------------------------------------------------------
	int i =- 1;
	//printf("%s\t", curr_type);
	while((strcmp(types_present[++i], curr_type) != 0) && (types_present[i][0]));

	if(!strcmp(types_present[i], curr_type)){
		return i;
	}

	//not sure if 0 is an atom type.. so using -1
	return -1;
}

//----------------------------------------------------------------------------
void InitHostMem(){
//----------------------------------------------------------------------------
	//Allocate various arrays to read data on the host side...

	// calculate number of GPU thread blocks to launch
	// round up to nearest integer if nAtom is not divisible by BlockSize

	int NBlocks;

	NBlocks = nAtom / BlockSize; //;MAX_BLOCK_SIZE
	if((nAtom % BlockSize) != 0){
		NBlocks++; //MAX_BLOCK_SIZE
	}

	DynadimBlock.x = BlockSize; //MAX_BLOCK_SIZE;
	DynadimBlock.y = 1;
	DynadimBlock.z = 1;

	DynadimGrid.x = NBlocks;
	DynadimGrid.y = 1;
	DynadimGrid.z = 1;

	//Dimension of Blocks and Grid for Nonbond Kernel...
	NBdimBlock.x = DynadimBlock.x;
	NBdimBlock.y = DynadimBlock.y;
	NBdimBlock.z = DynadimBlock.z;

	NBdimGrid.x = NBPART * DynadimGrid.x;
	NBdimGrid.y = DynadimGrid.y;
	NBdimGrid.z = DynadimGrid.z;

	//Dimension of Blocks and Grid for Bonded Kernel...
	BondeddimBlock.x = DynadimBlock.x;
	BondeddimBlock.y = DynadimBlock.y;
	BondeddimBlock.z = DynadimBlock.z;

	BondeddimGrid.x = BPART * DynadimGrid.x;
	BondeddimGrid.y = DynadimGrid.y;
	BondeddimGrid.z = DynadimGrid.z;


/*
	if (NBlocks > 512){
		DynadimGrid.y = (NBlocks / 512);
		if ((NBlocks%512) != 0) DynadimGrid.y++;

		DynadimGrid.x = (NBlocks / DynadimGrid.y);
		if ((NBlocks%DynadimGrid.y) != 0) DynadimGrid.x++;
	}
*/

	int j;

	printf("GPUgrid.x = %d, GPUgrid.y = %d, GPUblock.x = %d\n",
	       DynadimGrid.x, DynadimGrid.y, DynadimBlock.x);
#ifdef PME_CALC
	printf("FFT GridSize = %d x %d x %d\n", fftx, ffty, fftz);

///BCMultiply block and grid sizes///
	BCMuldimBlock.x = BCMULTIPLY_BLOCKSIZE;
	BCMuldimBlock.y = 1;
	BCMuldimBlock.z = 1;

	NBlocks = fftx * ffty * fftz / BCMuldimBlock.x;

	BCMuldimGrid.x = NBlocks;
	BCMuldimGrid.y = 1;
	BCMuldimGrid.z = 1;

	if((fftx * ffty * fftz % BCMuldimBlock.x) !=0 ){
		BCMuldimBlock.x = fftx;

		BCMuldimGrid.x = ffty;
		BCMuldimGrid.y = fftz;
	}

//printf("PME Threadblock (x, y, z): (%d, %d, %d);"
//       " PME Threadgrid (x, y, z): (%d, %d, %d)\n",
//       BCMuldimBlock.x, BCMuldimBlock.y, BCMuldimBlock.z,
//       BCMuldimGrid.x, BCMuldimGrid.y, BCMuldimGrid.z);

// Lattice Update block and grid sizes//
	LatUpdimBlock.x = LATTICEUPDATE_BLOCKSIZE;
	LatUpdimBlock.y	= 1;
	LatUpdimBlock.z = 1;

	NBlocks = fftx * ffty * fftz / LatUpdimBlock.x;
	LatUpdimGrid.x = NBlocks;
	LatUpdimGrid.y = 1;
	LatUpdimGrid.z = 1;

	if((fftx * ffty * fftz % LatUpdimBlock.x) !=0 ){
		LatUpdimBlock.x = fftx;

		LatUpdimGrid.x = ffty;
		LatUpdimGrid.y = fftz;
	}
///PME FORCE block and grid sizes///
	PMEForceBlock.x = BlockSize; //PMEFORCE_BLOCKSIZE;
	PMEForceBlock.y = 1;
	PMEForceBlock.z = 1;

	NBlocks = nAtom / PMEForceBlock.x;
	if((nAtom % PMEForceBlock.x) != 0){
		NBlocks++;
	}

	PMEForceGrid.x = NBlocks;
	PMEForceGrid.y = 1;
	PMEForceGrid.z = 1;

///CHARGE spread block and grid sizes///
	ChargeSpreadBlock.x = CHARGESPREAD_BLOCKSIZE;
	ChargeSpreadBlock.y = 1;
	ChargeSpreadBlock.z = 1;

	NBlocks = fftx * ffty * fftz / ChargeSpreadBlock.x;
	if((fftx * ffty * fftz % ChargeSpreadBlock.x) != 0){
		NBlocks++;
	}

	ChargeSpreadGrid.x = NBlocks;
	ChargeSpreadGrid.y = 1;
	ChargeSpreadGrid.z = 1;

#endif
	//initialize all memories...
	//least integral of number of blocks greater than nAtoms..
	WorkgroupSize = DynadimGrid.x * DynadimGrid.y * DynadimBlock.x;
	fsize = WorkgroupSize * sizeof(float);
	f2size = WorkgroupSize * sizeof(float2);
	f4size = WorkgroupSize * sizeof(float4);
	f3size = WorkgroupSize * sizeof(float3);
	isize = WorkgroupSize * sizeof(int);
	uisize = WorkgroupSize * sizeof(unsigned int);
	i2size = WorkgroupSize * sizeof(unsigned long long);
	i4size = WorkgroupSize * sizeof(int4);

	//Allocate memory and initialize lists...
	excllist = (int*)malloc(isize * (EXCL_COUNT_PERATOM + 1));
	ewlist = (int*)malloc(isize * (EXCL_COUNT_PERATOM + 1));
	excl_bitvec = (unsigned long long*)malloc(i2size);
	excl_bitvec_offset = (char*)malloc(sizeof(unsigned char) * WorkgroupSize);
	bonds_index = (int*)malloc(isize * (2 * BOND_COUNT_PERATOM + 1));
	//index of all angles per atom...
	angles_index = (int*)malloc(isize * (ANGLE_COUNT_PERATOM + 1));
	//index of all dihedrals per atom...
	dihedrals_index = (int*)malloc(isize * (DIHED_COUNT_PERATOM + 1));

	for(j = 0; j < WorkgroupSize; j++){
		bonds_index[j] = 0;
		angles_index[j] = 0;
		dihedrals_index[j] = 0;
		type[j] = 0;
		excllist[j] = 0;
		ewlist[j] = 0;

		if(j >= nAtom){
			r[j].x = 0;
			r[j].y = 0;
			r[j].z = 0;
		}
	}

	memset(angleprm, 0, MAX_ANGLE_TYPE * sizeof(float2));
	memset(bondprm, 0, MAX_BOND_TYPE * sizeof(float2));

#ifdef UREY_BRADLEY
	//Urey_Bradley list with only one other atom and a hash per entry...
	ureyb_index = (int*)malloc(isize * (2 * ANGLE_COUNT_PERATOM + 1));
	for(j = 0; j < WorkgroupSize; j++){
		ureyb_index[j] = 0;
	}
#endif

#ifdef IMPROPER
	impropers_index = (int*)malloc(isize * (DIHED_COUNT_PERATOM + 1));
	for(j = 0; j < WorkgroupSize; j++){
		impropers_index[j] = 0;
	}
#endif

	if(restraints){
		for(j = nAtom; j < WorkgroupSize; j++){
			seg_name[j][0] = 0; //initialize the segids
		}
	}

	if(shake){
		constraints = (int2*)malloc(MAX_CLUSTERS * (CLUSTER_SIZE + 1) *
		                            sizeof(int2));

		memset (constraints, 0, MAX_CLUSTERS * (CLUSTER_SIZE + 1) * sizeof(int2));
	}

	printf("Allocated Host Memory\n");
}

//----------------------------------------------------------------------------
void ReadXYZ(){
//----------------------------------------------------------------------------
	FILE* xyzcoordfile;
	char line[9999];
	float min_x = 0.0f;
	float min_y = 0.0f;
	float min_z = 0.0f;

	int atomid;
	char tempstring[99];

	if(!(xyzcoordfile = fopen(FileName, "r"))){
		printf("Error reading %s...\n", FileName);
		exit(-1);
	}

	fscanf(xyzcoordfile, "%d", &nAtom);
	fgets(line, 9999, xyzcoordfile); //Read the comments in the second line..
	fgets(line, 9999, xyzcoordfile);
	//printf("%s\n", line);

	for(atomid = 0; atomid < nAtom; atomid++){
		fscanf(xyzcoordfile, "%s %f %f %f", tempstring,
		      &r[atomid].x, &r[atomid].y, &r[atomid].z);

		//printf("%d = %f\t%f\t%f\n", atomid,
		//       r[atomid-1].x, r[atomid-1].y, r[atomid-1].z);

		fgets(line, 9999, xyzcoordfile);

		min_x = (r[atomid].x < min_x) ? r[atomid].x : min_x;
		min_y = (r[atomid].y < min_y) ? r[atomid].y : min_y;
		min_z = (r[atomid].z < min_z) ? r[atomid].z : min_z;
	}

	fclose(xyzcoordfile);

	printf("%d \t coordinates successfully read."
	       " Maximum allowed number of atoms = " EXPAND(NMAX) "\n", nAtom);

	printf("%f %f %f\n", min_x, min_y, min_z);

	return;
}

//----------------------------------------------------------------------------
void ReadCrd(){
//----------------------------------------------------------------------------
	FILE* crdfile;
	char line[9999];
	float min_x = 0.0f;
	float min_y = 0.0f;
	float min_z = 0.0f;

	//char tempstr[9999];
	int atomid;//, tempnum;
	char delims[] = " \t\r\n";
	char* tempstring;
	char tempstring1[99];
	int i;
	if(!(crdfile = fopen(FileName, "r"))){
		printf("Error reading %s...\n", FileName);
		exit(-1);
	}

	fgets(line, 9999, crdfile);
	while((line[0] == '*') || (line[0] == 0)){
		fgets(line, 9999, crdfile);
	}

	tempstring = strtok(line, delims);
	nAtom = atoi(tempstring);
	// printf("%d\n", nAtom);

	for(i = 0; i < nAtom; i++){
		fscanf(crdfile, "%d", &atomid);
		fscanf(crdfile, "%s %s %s",  tempstring1, tempstring1, tempstring1);
		fscanf(crdfile, "%f %f %f", &r[atomid-1].x,
		       &r[atomid-1].y, &r[atomid-1].z);
		//printf("%d = %f\t%f\t%f\n", atomid,
		//       r[atomid-1].x, r[atomid-1].y, r[atomid-1].z);

		fscanf(crdfile, "%s", seg_name[i]);

		if(restraints){
			segid[i] = -1;
			if(cicompare(seg_name[i], segname0) == 0){
				segid[i] = 0;
			}

			if(cicompare(seg_name[i], segname1) == 0){
				segid[i] = 1;
			}
		}

		fgets(line, 9999, crdfile);

		min_x = (r[atomid - 1].x < min_x) ? r[atomid - 1].x : min_x;
		min_y = (r[atomid - 1].y < min_y) ? r[atomid - 1].y : min_y;
		min_z = (r[atomid - 1].z < min_z) ? r[atomid - 1].z : min_z;
	}

	fclose(crdfile);
	/*
	for (i=0; i<nAtom; i++){
	r[i].x -= min_x;
	r[i].y -= min_y;
	r[i].z -= min_z;
	}
	*/
	/*
	FILE *segid_file = fopen("segid.txt", "w");;
	for (i=0; i<nAtom; i++)
	fprintf(segid_file, "%d %s %d\n", i, seg_name[i], segid[i]);
	fclose(segid_file);
	*/

	printf("%d \t coordinates successfully read."
	       " Maximum allowed number of atoms = " EXPAND(NMAX) "\n", nAtom);

	printf("%f %f %f\n", min_x, min_y, min_z);
	return;
}

//----------------------------------------------------------------------------
void ReadPdb(){
//----------------------------------------------------------------------------
	FILE* pdbfile;

	char line[9999];
	//char tempstr[9999];
	char delims[] = " \t\r\n";
	char* temparray[99];
	int i;
	float min_x = 0.0f;
	float min_y = 0.0f;
	float min_z = 0.0f;

	if(!(pdbfile = fopen(FileName, "r"))){
		printf("Error reading %s...\n", FileName);
		exit(-1);
	}

	temparray[0] = "";
	nAtom = 0;
	while(fgets(line, 9999, pdbfile) && (cicompare(temparray[0], "END")!=0)){

		i = 0;
		temparray[i] = strtok(line, delims);

		if(cicompare(temparray[0], "REMARK") == 0){
			continue;
		}

		while(temparray[i]){
			temparray[++i] = strtok(NULL, delims);
		}

		if(cicompare(temparray[0], "ATOM") == 0){
			//atomid = atoi(temparray[1]);
			r[nAtom].x = (float)atof(temparray[5]);
			r[nAtom].y = (float)atof(temparray[6]);
			r[nAtom].z = (float)atof(temparray[7]);

			min_x = (r[nAtom].x < min_x) ? r[nAtom].x : min_x;
			min_y = (r[nAtom].y < min_y) ? r[nAtom].y : min_y;
			min_z = (r[nAtom].z < min_z) ? r[nAtom].z : min_z;
			// MT: read molecular id for each atom from pdb file
			molid[nAtom] = atoi(temparray[4]);

			nAtom++;

		}
	}//while (!feof(pdbfile)||!strcmp(line, "END"))

	/*
	fscanf(pdbfile, "%s", line);
	while (!feof(pdbfile)||!strcmp(line, "END")){
	if (!strcmp(line, "ATOM")){//read the corresponding coordinates...
	fscanf(pdbfile,"%d %s %s %d",&atomid, tempstr, tempstr, &tempnum);
	fscanf(pdbfile, "%f %f %f", &r[atomid-1].x, &r[atomid-1].y, &r[atomid-1].z);
	nAtom++;
	fprintf(debug_file, "%d\t%d\t%f\t%f\t%f\n", atomid,
	        nAtom, r[atomid-1].x, r[atomid-1].y, r[atomid-1].z);
	}
	//next line...
	fgets(line, 9999, pdbfile);
	fscanf(pdbfile, "%s", line);
	}
	*/
	//fclose(debug_file);

	//center the coordinates around (0,0,0)
	fclose(pdbfile);

	//for (i=0; i<nAtom; i++){
		//r[i].x -= min_x;
		//r[i].y -= min_y;
		//r[i].z -= min_z;
	//}

	printf("nAtoms = %d\n", nAtom);
	return;
}

//------------------------------------------------------------------------------
void ReadInput(char* input_filename){
//------------------------------------------------------------------------------
	int j;
	//float temp;
	char line[9999];
	char checkpoint_format[512]="";
	char shake_preference[512]="";
	char* descriptor;
	char delims[] = " \t\n\r";
	float temp;

	if(!(conffile = fopen(input_filename, "r"))){
		printf("File %s not found\n", input_filename);
		exit(-1);
	}
	TrjFreq = -1;
	while(fgets(line, 9999, conffile)){

		descriptor = strtok(line, delims);

		if(descriptor == NULL){
			continue;
		}
		if(descriptor[0] == '#'){
			//ignore comments...
			continue;
		}

		if(cicompare(descriptor, "coordinates") == 0){
			sprintf(FileName, "%s", strtok(NULL, delims));
			continue;
		}
		if(cicompare(descriptor, "structure") == 0){
			sprintf(psfFileName, "%s", strtok(NULL, delims));
			continue;
		}
		if(cicompare(descriptor, "parameters") == 0){
			sprintf(prmFileName, "%s", strtok(NULL, delims));
			continue;
		}
		if(cicompare(descriptor, "topology") == 0){
			sprintf(topFileName, "%s", strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "temperature")){
			InitTemp = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "thermostat")){
			Tau = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "minimizesteps")){
			MinimizeStepLimit = (int)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "conjgradfac")){
			cgfac = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "steepdesfac")){
			sdfac = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "conjgradrate")){
			cgrate = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "steepdesrate")){
			sdrate = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "startstep")){
			StartCount = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "timestep")){
			DeltaT = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "dynasteps")){
			StepLimit = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "KAPPa")){
			KAPPa  = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "INBFRQ")){
			INBFRQ  = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "tempresetTiming")){
			VAssign = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "outputTiming")){
			StepAvg = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "checkpointTiming")){
			StepStore = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "cuton")){
			Cuton = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "cutoff")){
			Cutoff = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "pairlistdist")){
			CutMax = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "gpublocksize")){
			BlockSize = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "trajfrequencyTiming")){
			TrjFreq = (int)atoi(strtok(NULL, delims));
			continue;
		}

		else if(!cicompare(descriptor, "cellbasisvector1")){
			temp = (float)atof(strtok(NULL, delims));
			//read into temp and assign to Region to
			//suppress "variable unused" warnings..
			Region.x = temp;
			printf("Region.x %f \n", Region.x);
			temp = (float)atof(strtok(NULL, delims));
			temp = (float)atof(strtok(NULL, delims));
			continue;
		}

		else if(!cicompare(descriptor, "cellbasisvector2")){
			temp = (float)atof(strtok(NULL, delims));
			temp = (float)atof(strtok(NULL, delims));
			Region.y = temp;
			printf("Region.y %f \n", Region.y);
			temp = (float) atof(strtok(NULL, delims));
			continue;
		}

		else if(!cicompare(descriptor, "cellbasisvector3")){
			temp = (float)atof(strtok(NULL, delims));
			temp = (float)atof(strtok(NULL, delims));
			temp = (float)atof(strtok(NULL, delims));
			Region.z = temp;
			printf("Region.z %f \n", Region.z);
			continue;
		}

#ifdef PME_CALC
		else if(!cicompare(descriptor, "pmegridsizex")){
			fftx = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "pmegridsizey")){
			ffty = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "pmegridsizez")){
			fftz = (int)atoi(strtok(NULL, delims));
			continue;
		}
#endif

#ifdef PCONSTANT
		else if(!cicompare(descriptor, "pmass")){
			pmass_cubic = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "pref")){
			pRef = (float)atof(strtok(NULL, delims));
			continue;
		}
#endif

		else if(!cicompare(descriptor, "consharm")){
			sprintf(segname0, "%s", strtok(NULL, " \t"));
			sprintf(segname1, "%s", strtok(NULL, delims));
			restraints = 1;
		}

		else if(!cicompare(descriptor, "consharmdist")){
			consharmdist.x = (float)atof(strtok(NULL, delims));
			consharmdist.y = (float)atof(strtok(NULL, delims));
			consharmdist.z = (float)atof(strtok(NULL, delims));
			continue;
		}

		else if(!cicompare(descriptor, "consharmfc")){
			consharmfc = (float)atof(strtok(NULL, delims));
			continue;
		}

		else if(!cicompare(descriptor, "shake")){
			sprintf(shake_preference, "%s", strtok(NULL, delims));
			shaketol = (float)atof(strtok(NULL, delims)); //shake tolerance...
			continue;
		}

#ifdef USE_CONSFIX
		else if(!cicompare(descriptor, "consfix")){
			// printf("CONSFIX: \n");
			j = 0;
			// char *tempstring;
			consfix = 1;
			consfix_nseg = (int)atoi(strtok(NULL, delims));

			while(j < consfix_nseg){
				sprintf(consfix_segname[j], "%s", strtok(NULL, delims));
				// printf("CONSFIX: Segmentid %d  Segmentname %s\n",
				//j, consfix_segname[j]);
				j++;
			}

			/*
			TO DO - make this part fo the code robust !!!
			the code has tocheck that the number od segment are matching the list
                      if (j < 2) {
			  printf("ERROR: constfix needs at least two segments \n");
			  exit(-1);
			}
			*/
			continue;
		}
#endif

		else if(!cicompare(descriptor, "hfac")){
			hfac = (float)atof(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "printlevel")){
			PrintLevel = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "printtostdout")){
			printtostdout = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "keepcheckpoints")){
			keepcheckpoint = (int)atoi(strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "checkpointformat")){
			sprintf(checkpoint_format, "%s", strtok(NULL, delims));
			continue;
		}
		else if(!cicompare(descriptor, "seed")){
			seed = atof(strtok(NULL, " \t\n"));
			continue;
		}
		else if(!cicompare(descriptor, "outputtrajectory")){
			sprintf(trajFileName, "%s", strtok(NULL, delims));
			continue;
		}

		if((descriptor[0] != 0) && (!feof(conffile))){
			printf("Ignoring unrecognized token, \"%s\", start line with \"#\" to"
			       " comment and supress this warning...\n", descriptor);
			continue;
		}

	}//while(fgets(line, 9999, conffile))

	LoadCoord = &ReadCrd;
	int i = (int)strlen(FileName);
	while(FileName[i] != '.'){
		i--;
	}

	if(!cicompare(FileName + i + 1, "pdb")){
		printf("Loading PDB coordinates\n");
		LoadCoord = &ReadPdb;
	}
	else if(!cicompare(FileName + i + 1, "crd")){
		printf("Loading CRD coordinates\n");
		//LoadCoord = &ReadCrd;
	}
	else if(!cicompare(FileName+ i + 1, "xyz")){
		printf("Loading XYZ coordinates\n");
		LoadCoord = &ReadXYZ;
	}

	if(cicompare(shake_preference, "on") == 0){
		shake = 1;
	}
	else if(cicompare(shake_preference, "off") == 0){
		shake = 0;
	}
	else{
		printf("Unrecognized token %s for value \"shake\"(must be either \"on\", "
		       "or \"off\"), defaulting to \"off\"\n", shake_preference);
	}

	if (TrjFreq == -1){ //this means that no trajectory frequency was given
	   TrjFreq = StepAvg;
	   printf("option trajfrequencyTiming not specified.\n"
	          "defaulting to outputTiming value of %d.\n", StepAvg);
	}

	if(printtostdout){
		printout = &print_to_stdout;
	}
	else{
		printout = &print_to_file;
	}

	if((checkpoint_format[0] == 'a') ||
	   (checkpoint_format[0] == 'A') ||
	   (checkpoint_format[0] == '1')){

		LoadCheckpoint = &LoadCheckpointAscii;
		SaveCheckpoint = &SaveCheckpointAscii;
	}
	else if((checkpoint_format[0] == 'b') ||
	        (checkpoint_format[0] == 'B') ||
	        (checkpoint_format[0] == '2')){

		LoadCheckpoint = &LoadCheckpointBinary;
		SaveCheckpoint = &SaveCheckpointBinary;
	}
	else if(checkpoint_format[0] != 0){
		printf("Checkpoint format \"%s\" is undefined. It must be \"Ascii\" or "
		       "\"Binary\".\n", checkpoint_format);
		exit(-1);
	}

	printf("Coordinates : %s\n", FileName);
	printf("%d", strlen(FileName));
	printf("Structure : %s\n", psfFileName);
	printf("PRM : %s\n", prmFileName);
	printf("TOP : %s\n", topFileName);

	printf("Region.x = %f, Region.y = %f, Region.z = %f\n",
	       Region.x, Region.y, Region.z);

	//set the values needed for simulation...
	RegionH.x = Region.x / 2;
	RegionH.y = Region.y / 2;
	RegionH.z = Region.z / 2;

	/* Computes basic parameters */
	DeltaTH = 0.5f * DeltaT;

	CutCheck = 0.5f * (CutMax - Cutoff);

	float rc3 = Cutoff * Cutoff * Cutoff;
	float rc6 = rc3 * rc3;
	float rc12 = rc6 * rc6;
	rc7 = rc6 * Cutoff;
	rc13 = rc12 * Cutoff;

	// initialize atomic params to NaN as a code to indicate they are not set
	for(j = 0; j < MAXTYP; j++) {
		prm[j].x = NAN;
		prm[j].y = NAN;
		prm[j].z = NAN;
		prm[j].w = NAN;


#if (NBXMOD==5)
		prm1_4[j].x = 0.0f;
		prm1_4[j].y = 0.0f;
#endif
	}

	// initialize atomic eps and sigma
	// printf("NBFIX: %d\n", num_types_present);
	for(int i = 0;i < MAX_NBFIX_TYPE; i++){
		nbfixprm[i].x = NAN;
		nbfixprm[i].y = NAN;
	}
	/* Debugging
        printf("NBFIX: init %d\n", num_types_present);
	for (int i=0;i<MAX_NBFIX_TYPE; i++){
            if ((i % 16) == 0) printf("\n");
            printf("%f %f ", nbfixprm[i].x, nbfixprm[i].y);
        }
	printf("\n");
	*/

	for(j = 0; j < MAX_DIHED_TYPE; j++){
		dihedral_type_count[j] = 0;
		wildcardstatus[j] = 1;
	}

	fclose(conffile);

	//allocate host memory for position, velocity, acceleration...
	r = (float4*)malloc(sizeof(float4) * NMAX);
	rv = (float4*)malloc(sizeof(float4) * NMAX);
	// MT setting
	for(i = 0; i < nAtom; i++) {
		rv[i].x=0.0f;
		rv[i].y=0.0f;
		rv[i].z=0.0f;
		rv[i].w=1.0f;
	}

	f4h = (float4*)malloc(sizeof(float4) * NMAX);
	f4h_bonded = (float4*)malloc(BPART * sizeof(float4) * NMAX);
	f4h_nonbond = (float4*)malloc(NBPART * sizeof(float4) * NMAX);
#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
	f4h_nonbond0 = (float4*)malloc(NBPART * sizeof(float4) * NMAX);
	f4h_nonbond1 = (float4*)malloc(NBPART * sizeof(float4) * NMAX);
#endif
#endif

	// npt
	p1h = (float4*)malloc(sizeof(float4) * NMAX);
	p2h = (float4*)malloc(sizeof(float4) * NMAX);
 	p3h = (float4*)malloc(sizeof(float4) * NMAX);
 	p4h = (float4*)malloc(sizeof(float4) * NMAX);
 	r1h = (float4*)malloc(sizeof(float4) * NMAX);

	//initialize stepCount;
	stepCount = StartCount;

	return;
}

//------------------------------------------------------------------------------
int present_in_dihedrals_set(int w, int x, int y, int z){
//dihedrals set is the set of dihedrals present in the system...
//------------------------------------------------------------------------------
	int i;
	int c1;
	int c2;

	for(i = 0; i < dihedrals_set[0]; i++){
		//symmetric dihedrals.. (w,x,y,z)=(z,y,x,w)
		c1 = ((w == dihedrals_set[4 * i + 1]) &&
		      (x == dihedrals_set[4 * i + 2]) &&
		      (y == dihedrals_set[4 * i + 3]) &&
		      (z == dihedrals_set[4 * i + 4]));

		c2 = ((z == dihedrals_set[4 * i + 1]) &&
		      (y == dihedrals_set[4 * i + 2]) &&
		      (x == dihedrals_set[4 * i + 3]) &&
		      (w == dihedrals_set[4 * i + 4]));

		if(c1 || c2){
			return 1;
		}
	}

	return 0;
}

//------------------------------------------------------------------------------
int add_retrieve_dihedrals_set_type(int w,
                                    int x,
                                    int y,
                                    int z,
                                    char add_dihedral){
//------------------------------------------------------------------------------
	int i;

	char c1;
	char c2;

	for(i = 0; i < dihedrals_set_type[0]; i++){
		//symmetric dihedrals.. (w,x,y,z)=(z,y,x,w)
		c1 = (((w == dihedrals_set_type[4 * i + 1]) || (w == -1)) &&
		      (x == dihedrals_set_type[4 * i + 2]) &&
		      (y == dihedrals_set_type[4 * i + 3]) &&
		      ((z == dihedrals_set_type[4 * i + 4]) || (z == -1)));

		c2 = (((z == dihedrals_set_type[4 * i + 1]) || (z == -1)) &&
		      (y == dihedrals_set_type[4 * i + 2]) &&
		      (x == dihedrals_set_type[4 * i + 3]) &&
		      ((w == dihedrals_set_type[4 * i + 4]) || (w == -1)));

		//c1 = ((w==dihedrals_set_type[4*i + 1])&&(x==dihedrals_set_type[4*i + 2])&&
		//(y==dihedrals_set_type[4*i + 3])&&(z==dihedrals_set_type[4*i + 4]));
		//c2 = ((z==dihedrals_set_type[4*i + 1])&&(y==dihedrals_set_type[4*i + 2])&&
		//(x==dihedrals_set_type[4*i + 3])&&(w==dihedrals_set_type[4*i + 4]));

		if(c1 || c2){
			return i;
		}
	}

	//add the dihedral type only if not already present in the set,...
	if(add_dihedral){
		dihedrals_set_type[4 * dihedrals_set_type[0] + 1] = w;
		dihedrals_set_type[4 * dihedrals_set_type[0] + 2] = x;
		dihedrals_set_type[4 * dihedrals_set_type[0] + 3] = y;
		dihedrals_set_type[4 * dihedrals_set_type[0] + 4] = z;

		dihedrals_set_type[0]++;

		if(dihedrals_set_type[0] >= MAX_DIHED_TYPE){
			printf("\nError! Actual number of Dihed Types has exceeded the allocated"
			       " number of Dihed Types of %d, \n Recompile the code with a "
			       "larger number...", dihedrals_set_type[0]);
			exit(0);
		}

		return(dihedrals_set_type[0] - 1);
	}

	return -1;
}

#ifdef IMPROPER
//------------------------------------------------------------------------------
int add_retrieve_impropers_set_type(int w,
                                    int x,
                                    int y,
                                    int z,
                                    char add_improper){
//------------------------------------------------------------------------------
	int i;
	char c1;
	char c2;

	for(i = 0; i < impropers_set_type[0]; i++){
		c1 = ((w == impropers_set_type[4 * i + 1]) &&
		      ((x == impropers_set_type[4 * i + 2]) || (x == -1)) &&
		      ((y == impropers_set_type[4 * i + 3]) || (y == -1)) &&
		      (z == impropers_set_type[4 * i + 4]));

		c2 = ((z == impropers_set_type[4 * i + 1]) &&
		      ((y == impropers_set_type[4 * i + 2]) || (y == -1)) &&
		      ((x == impropers_set_type[4 * i + 3]) || (x == -1)) &&
		      (w == impropers_set_type[4 * i + 4]));

		if(c1 || c2){
			return i;
		}
	}

	if((add_improper) && (x != -1) && (y != -1)){
		impropers_set_type[4 * impropers_set_type[0] + 1] = w;
		impropers_set_type[4 * impropers_set_type[0] + 2] = x;
		impropers_set_type[4 * impropers_set_type[0] + 3] = y;
		impropers_set_type[4 * impropers_set_type[0] + 4] = z;

		impropers_set_type[0]++;
		return(impropers_set_type[0] - 1);
	}

	return -1;
}
#endif
//------------------------------------------------------------------------------
int add_retrieve_angles_set_type(int x, int y, int z, char add_angle){
//------------------------------------------------------------------------------
	int i;
	int c1;
	int c2;

	for(i = 0; i < angles_set_type[0]; i++){
		c1 = ((x == angles_set_type[3 * i + 1]) &&
		      (y == angles_set_type[3 * i + 2]) &&
		      (z == angles_set_type[3 * i + 3]));

		c2 = ((z == angles_set_type[3 * i + 1]) &&
		      (y == angles_set_type[3 * i + 2]) &&
		      (x == angles_set_type[3 * i + 3]));

		if(c1 || c2){
			return i;
		}
	}

	//add the angle type only if not already present in the set,...
	if(add_angle){
		angles_set_type[3 * angles_set_type[0] + 1] = x;
		angles_set_type[3 * angles_set_type[0] + 2] = y;
		angles_set_type[3 * angles_set_type[0] + 3] = z;

		angles_set_type[0]++;

		if(angles_set_type[0] >= MAX_ANGLE_TYPE){
			printf("\nError! Actual number of Angle Types has exceeded the allocated"
			       " number of Angle Types of %d, \n Recompile the code with a "
			       "larger number...", angles_set_type[0]);
			exit(0);
		}

		return(angles_set_type[0] - 1);
	}

	return -1;
}

#ifdef UREY_BRADLEY
//----------------------------------------------------------------------------
int add_retrieve_ureyb_set_type(int x, int y, int z, char add_ureyb){
//----------------------------------------------------------------------------
	int i;
	int c1;
	int c2;

	for(i = 0; i < ureyb_set_type[0]; i++){

		c1 = ((x == ureyb_set_type[3 * i + 1]) &&
		      (y == ureyb_set_type[3 * i + 2]) &&
		      (z == ureyb_set_type[3 * i + 3]));

		c2 = ((z == ureyb_set_type[3 * i + 1]) &&
		      (y == ureyb_set_type[3 * i + 2]) &&
		      (x == ureyb_set_type[3 * i + 3]));

		//c1 = ((x==ureyb_set_type[2*i + 1])&&(z==ureyb_set_type[2*i + 2]));
		//c2 = ((z==ureyb_set_type[2*i + 1])&&(x==ureyb_set_type[2*i + 2]));

		if(c1 || c2){
			return i;
		}
	}

	if(add_ureyb){
		ureyb_set_type[3 * ureyb_set_type[0] + 1] = x;
		ureyb_set_type[3 * ureyb_set_type[0] + 2] = y;
		ureyb_set_type[3 * ureyb_set_type[0] + 3] = z;

		ureyb_set_type[0]++;
		return(ureyb_set_type[0] - 1);
	}

	return -1;
}
#endif

//------------------------------------------------------------------------------
int add_retrieve_bonds_set_type(int x, int y, char add_bond){
//------------------------------------------------------------------------------
	int i;
	int c1;
	int c2;

	for(i = 0; i < bonds_set_type[0]; i++){
		c1 = ((x == bonds_set_type[2 * i + 1]) && (y == bonds_set_type[2 * i + 2]));
		c2 = ((y == bonds_set_type[2 * i + 1]) && (x == bonds_set_type[2 * i + 2]));

		if(c1 || c2){
			return i;
		}
	}

	if(add_bond){
		bonds_set_type[2 * bonds_set_type[0] + 1] = x;
		bonds_set_type[2 * bonds_set_type[0] + 2] = y;

		bonds_set_type[0]++;

		if(bonds_set_type[0] >= MAX_BOND_TYPE){
			printf("\nError! Actual number of Bond Types has exceeded the allocated"
			       " number of Bond Types of %d,\n Recompile the code with a"
			       " larger number...", bonds_set_type[0]);
			exit(0);
		}

		return(bonds_set_type[0] - 1);
	}

	return -1;
}

//------------------------------------------------------------------------------
int add_retrieve_nbfix_set_type(int x, int y, char add_nbfix){
//----------------------------------------------------------------------------
	int i;
	int c1;
	int c2;

	for(i = 0; i < nbfix_set_type[0]; i++){
		c1 = ((x == nbfix_set_type[2 * i + 1]) && (y == nbfix_set_type[2 * i + 2]));
		c2 = ((y == nbfix_set_type[2 * i + 1]) && (x == nbfix_set_type[2 * i + 2]));

		if(c1 || c2){
			return i;
		}
	}

	if(add_nbfix){
		nbfix_set_type[2 * nbfix_set_type[0] + 1] = x;
		nbfix_set_type[2 * nbfix_set_type[0] + 2] = y;

		nbfix_set_type[0]++;

		if(nbfix_set_type[0] >= MAX_BOND_TYPE){
			printf("\nError! Actual number of Bond Types has exceeded the allocated"
			       " number of Bond Types of %d,\n Recompile the code with a "
			       "larger number...", nbfix_set_type[0]);
			exit(0);
		}

		return(nbfix_set_type[0] - 1);
	}

	return -1;
}

//----------------------------------------------------------------------------
void add_to_constraints(int x, int y){
//adds a pair of atoms to the list of constraints...
//Since we dont know the number of clusters ahead of time, the data structure
//is built in row major order...
//The data structure is then transposed before copying to device memory,
//for contiguous memory access
//----------------------------------------------------------------------------
	int j;
	int i;
	int n;

	char add_to_cluster = 0;
	//printf("nClusters=%d,", nClusters);

	//iterate through the clusters, in reverse to save time...
	for(j = nClusters - 1; j >= 0; j--){

		//printf("j=%d,", j);

		//number of constraints in this clusters...
		n = constraints[(CLUSTER_SIZE + 1) * j].x;

		//printf("n=%d,", n);
		add_to_cluster = 0;

		for(i = 1; i <= n; i++){
			if((constraints[(CLUSTER_SIZE + 1) * j + i].x == x) ||
			   (constraints[(CLUSTER_SIZE + 1) * j + i].y == x) ||
			   (constraints[(CLUSTER_SIZE + 1) * j + i].x == y) ||
			   (constraints[(CLUSTER_SIZE + 1) * j + i].y == y)){

				//the constraint belongs in this cluster
				add_to_cluster = 1;
			}

			if(add_to_cluster == 1){
				break;
			}
		} //for (i=0; i<n; i++)

		if(add_to_cluster == 1){
			n = n + 1;
			if(n > CLUSTER_SIZE){
				printf("CLUSTER_SIZE reached, please recompile with a larger value\n");
				exit(0);
			}

			constraints[(CLUSTER_SIZE + 1) * j].x = n;
			constraints[(CLUSTER_SIZE + 1) * j + n].x = x;
			constraints[(CLUSTER_SIZE + 1) * j + n].y = y;
			return; //added the atoms to constraints.. just return..
		}
	} //for (j=0; j<nClusters; j++)

	//if not a part of any clusters start a new cluster....
	nClusters++;

	if(nClusters >= MAX_CLUSTERS){
		printf("MAX_CLUSTERS for constraints reached, please recompile with a"
		       " larger value\n");
		exit(0);
	}

	constraints[(CLUSTER_SIZE + 1) * (nClusters - 1)].x = 1;
	constraints[(CLUSTER_SIZE + 1) * (nClusters - 1) + 1].x = x;
	constraints[(CLUSTER_SIZE + 1) * (nClusters - 1) + 1].y = y;

}

//----------------------------------------------------------------------------
int get_localid(int* local_atoms, int atomid){
//----------------------------------------------------------------------------
	int i;

	for(i = 1; i <= local_atoms[0]; i++){
		if(atomid == local_atoms[i]){
			return(i - 1);
		}
	}

	local_atoms[0]++;
	local_atoms[local_atoms[0]] = atomid;
	return(local_atoms[0] - 1);
}

//----------------------------------------------------------------------------
void add_atom_to_cluster(int clusterid, int key, int total_clusters){
//----------------------------------------------------------------------------
	int i;

	for(i = 1; i <= atoms_in_cluster[clusterid]; i++){
		if(atoms_in_cluster[total_clusters * i + clusterid] == key){
			return;
		}
	}

	atoms_in_cluster[clusterid]++;
	if(atoms_in_cluster[clusterid] > ATOMS_IN_CLUSTER){
		printf("Atoms in Cluster exceeded " EXPAND(ATOMS_IN_CLUSTER)
		       ".. Please recompile...\n");
		exit(0);
	}

	atoms_in_cluster[total_clusters *
	                 atoms_in_cluster[clusterid] +
	                 clusterid] = key;

	return;
}

//----------------------------------------------------------------------------
void build_constraints(){
//transposes the data structure and assigns local atom ids..
//for use by the device..
//----------------------------------------------------------------------------
	int i;
	int j;
	int n;
	int h;

	int atomid0;
	int atomid1;
	int localid;

	int key;
	//char assigned_constraint=0;

	int local_atoms[CLUSTER_SIZE + 1];

	cluster_blocksize.x = BlockSize;
	cluster_blocksize.y = 1;
	cluster_blocksize.z = 1;

	printf("building %d H-bond constraint clusters\n", nClusters);

	////Allocate and Assign Shake Parameters......
	num_cluster_blocks.x = nClusters / cluster_blocksize.x;

	if(num_cluster_blocks.x * cluster_blocksize.x != nClusters){
		num_cluster_blocks.x++;
	}

	num_cluster_blocks.y = 1;
	num_cluster_blocks.z = 1;

	//reassign the number of clusters to be integral multiple of blocksize
	int nClusters_new = num_cluster_blocks.x * cluster_blocksize.x;

	int2* constraints_temp; //temporary data structure to hold the constraints...
	constraints_temp = (int2*)malloc(nClusters_new *
	                                 (CLUSTER_SIZE + 1) * sizeof(int2));

	memset(constraints_temp, 0, nClusters_new *
	       (CLUSTER_SIZE + 1) * sizeof(int2));

	atoms_in_cluster = (int*)malloc(nClusters_new *
	                                (ATOMS_IN_CLUSTER + 1) * sizeof(int));

	memset(atoms_in_cluster, 0, nClusters_new *
	       (ATOMS_IN_CLUSTER + 1) * sizeof(int));

	constraintsprm = (float2*)malloc(nClusters_new *
	                                 (CLUSTER_SIZE) * sizeof(float2));

	memset(constraintsprm, 0, nClusters_new * CLUSTER_SIZE * sizeof(float2));

	constraints_by_atom = (unsigned char*)malloc(nAtom *
	                                             (CONSTRAINTS_PER_ATOM + 1) *
	                                             sizeof(unsigned char));

	memset (constraints_by_atom, 0, nAtom *
	        CONSTRAINTS_PER_ATOM * sizeof(unsigned char));

//transpose and copy the data to the temporary data structure...
//and assign local atom ids...
	for(j = 0; j < nClusters; j++){

		constraints_temp[nClusters_new * 0 + j].x =
		                                      constraints[(CLUSTER_SIZE + 1) * j].x;

		n = constraints_temp[nClusters_new * 0 + j].x;

		num_constraints += n;

		local_atoms[0] = 0;

		for(i = 1; i <= n; i++){
			atomid0 = constraints[(CLUSTER_SIZE + 1) * j + i].x;

			localid = get_localid(local_atoms, atomid0);
			key = (localid << 24) | (atomid0);
			constraints_temp[nClusters_new * i + j].x = key;
			//adds the current atom to the cluster if it isn't already added..
			add_atom_to_cluster(j, key, nClusters_new);

			atomid1 = constraints[(CLUSTER_SIZE + 1) * j + i].y;
			localid = get_localid(local_atoms, atomid1);
			key = (localid << 24) | (atomid1);
			constraints_temp[nClusters_new * i + j].y = key;
			//adds the current atom to the cluster if it isn't already added..
			add_atom_to_cluster(j, key, nClusters_new);

			//1 to add, 0 to retrieve...
			h=add_retrieve_bonds_set_type(type[atomid0], type[atomid1], 0);

			constraintsprm[nClusters_new * (i - 1) + j].x = 0.0f;
			constraintsprm[nClusters_new * (i - 1) + j].y = bondprm[h].y;

			if(constraintsprm[nClusters_new * (i - 1) + j].y < 0.1f){
				printf("constraint distance for %s-%s is too small or not set... \n",
				       atom_type[atomid0], atom_type[atomid1]);
				exit(0);
			}

			constraints_by_atom[nAtom * 0 + atomid0]++;
			if(constraints_by_atom[nAtom * 0 + atomid0] > CONSTRAINTS_PER_ATOM){
				printf("Constraints per atom exceeded " EXPAND(CONSTRAINTS_PER_ATOM)
				       "... please recompile\n");
				exit(0);
			}
			//signbit is 0;
			int index = nAtom * constraints_by_atom[nAtom * 0 + atomid0] + atomid0;
			constraints_by_atom[index] = (i - 1) | (0 << 7);

			constraints_by_atom[nAtom * 0 + atomid1]++;

			if(constraints_by_atom[nAtom * 0 + atomid0] > CONSTRAINTS_PER_ATOM){
				printf("Constraints per atom exceeded " EXPAND(CONSTRAINTS_PER_ATOM)
				       "... please recompile\n");
				exit(0);
			}
			//signbit is 1;
			index = nAtom * constraints_by_atom[nAtom * 0 + atomid1] + atomid1;
			constraints_by_atom[index] = (i - 1) | (1 << 7);
		}
	}

	nClusters = nClusters_new;
	constraints = (int2*)realloc(constraints, nClusters *
	                             (CLUSTER_SIZE + 1) * sizeof(int2));
	memcpy(constraints, constraints_temp, nClusters *
	       (CLUSTER_SIZE + 1) * sizeof(int2));

	free(constraints_temp);

	printf("built %d H-bond constraint clusters\n", nClusters);

}

/*----------------------------------------------------------------------------*/
void ReadPsf_CHARMM(){
/*------------------------------------------------------------------------------
read coords, charges, parameters - psf file
------------------------------------------------------------------------------*/
	int i;
	int j;
	int w;
	int x;
	int y;
	int z;
	int id;
	int h;

	char line[9999];
	char curr_type[10];
	int read_case;
	char assigned_id = 0;
	char rtf_types[10000][10];
	//double e[3], seed;

	//---------- populate all the atom types and their corresponding ids ---
	//clear out type_2_name list
	for(i = 0; i < MAXTYP; i++){
		sprintf(types_present[i], "");
		sprintf(atom_type[i], "");
	}

	for(i = 0; i < 10000; i++){
		sprintf(rtf_types[i], "");
	}

	if(!(topfile = fopen(topFileName, "r"))){
		fprintf(logfile, "Warning! Topology file %s not found... Assigning numeric"
		        " type id based on the order of appearance...\n", topFileName);
		assigned_id = 1;
	}
	else{
		fprintf(logfile, "Reading %s...\n", topFileName);

		read_case = 1;
		//get to part where atom type is stored under "MASS"...
		while(read_case){
			fscanf(topfile, "%s", line);

			if(!strcmp(line, "MASS")){
				read_case=0;
			}
		}

		read_case = 1;
		fscanf(topfile, "%d", &id);//read the first atom type..
		fscanf(topfile, "%s", rtf_types[id]);
		fgets(line, 9999, topfile);

		while((read_case) && (strcmp(line, "END") != 0)){

			fscanf(topfile, "%s", line);

			if(!strcmp(line, "MASS")){//if line=="MASS"...
				fscanf(topfile, "%d", &id);
				fscanf(topfile, "%s", rtf_types[id]);
				fgets(line, 9999, topfile); //ignore the rest and continue...
				continue;
			}
			else if(line[0] == '!'){//check for all the comment types here...
				fgets(line, 9999, topfile);
				continue;
			}//ignore comments...
			//else, we've reached a different section...
			read_case = 0;
		}//finished reading the types file.... now read the prm file...

		fclose(topfile);
	}
//------- finished populating all the atom types and their corresponding ids ---

//---------- read psf file -----------------------------------------------------

	// read topology, masses, charges from psf file
	if(!(psffile = fopen(psfFileName,"r"))){
		printf("File %s not found\n", psfFileName);
		exit(-1);
	}

	// read first 2 lines, get number of lines in title
	fgets(line, 9999, psffile);
	fgets(line, 9999, psffile);
	fscanf(psffile, "%d %s", &j, line);

	// read past title
	for(i = 0; i <= j; i++){
		fgets(line, 9999, psffile);
	}

	// read number of atoms
	fscanf(psffile, " %d %s", &j, line);

	// exit if number of atoms in psf doesn't match
	if(j != nAtom){
	  printf("Number of atoms in psf (%d) does not match number of atoms "
		       "in coord file (%d).\n",j,nAtom);
	  exit(0);
	}

	for(i = 0; i < nAtom; i++){
		// read molid and atom type from psf
		// fscanf(psffile, "%s %s %s %s %s", line, line, line, line, atom_type[i]);
		fscanf(psffile, "%s %s", line, seg_type[i]);
		fscanf(psffile, "%d", &molid[i]);
		fscanf(psffile, "%s %s", line, atom_type[i]);

		if(assigned_id == 1){ //no rtf file...
			fscanf(psffile, "%s", curr_type);
			// printf("ReadPSF: %d %s \n", i, curr_type);
		}

		if(assigned_id==0){ //with rtf file...
			fscanf(psffile, "%d", &id);
			sprintf(curr_type, "%s", rtf_types[id]);
			atom_typeid[i] = id;
			// printf("ReadPSF: %d %s \n", id, rtf_types[id]);
		}

		j = addtype(curr_type);
		num_types_present = j;

		// fprintf(logfile, "TYPE_PRESENT: %d\n", num_types_present);
		if(num_types_present >= MAX_NBFIX_TYPE){
			printf("\nError! Actual number of Bond Types has exceeded the allocated"
			       " number of Bond Types of %d,\n Recompile the code with a "
			       "larger number...", num_types_present);
			exit(0);
		}

		// printf("ReadPSF: %d \n", j);

		// only read mass for this type if it has not been read before
		// read charge for this atom either way
		if(__myisnan(prm[j].w)){
			fscanf(psffile, "%f %f", &r[i].w, &prm[j].x);

			if(!shake){
				//used for Water Hydrogens
				if((strcmp(atom_type[i], "H1") == 0) ||
				   (strcmp(atom_type[i], "H2") == 0)){

					prm[j].x = hfac * prm[j].x;
				}

				//used for Methanol Hydrogens
				if((strcmp(atom_type[i], "HG1") == 0) ||
				   (strcmp(atom_type[i], "HB1") == 0) ||
				   (strcmp(atom_type[i], "HB2") == 0) ||
				   (strcmp(atom_type[i], "HB3") == 0)){

					prm[j].x = hfac*prm[j].x;
				}
			}
			// printf("id %d mass %f \n", j, prm[j].x);
			prm[j].w = 0.0f;
		}
		else{
			fscanf(psffile, "%f", &r[i].w);
		}

		//set the atom typeid...
		type[i] = j;
		// printf("set the atom typeid %d \n", type[i]);

		if(restraints){
			if(segid[i] == 0){
				mass_segid0 += prm[j].x;
			}

			if(segid[i] == 1){
				mass_segid1 += prm[j].x;
			}
		}

		eEwself += (r[i].w * r[i].w);

		fgets(line, 9999, psffile);
	}//for(i=0; i<nAtom; i++)

	num_types_present++;
	printf("TYPE_PRESENT: %d\n", num_types_present);

	// npt
	// Populate molmass with x mass atom; y mass molecule; z number atoms per
	//molecule; w start count
	int mtid; // molecule type id
	int k = 0;
	while(k < nAtom){
		mtid = molid[k];
		int mtstart = k;
		int mtstop = mtstart;
		float mtmass = prm[type[k]].x;
		molmass[k].x = prm[type[k]].x;
		molmass[k].w = mtstart;

		// printf("Atom %d Molecule %d Mass %f \n", k, mtid, molmass[k].x)
		do{
			k++;
			if(k == nAtom){
				break;
			}

			if(molid[k] == mtid){
				mtmass += prm[type[k]].x;
				molmass[k].x = prm[type[k]].x;
				molmass[k].w = mtstart;
			}
		}while((molid[k] == mtid));
		mtstop = k;

		for(int j = mtstart; j < mtstop; j++){
			molmass[j].y = mtmass;
			molmass[j].z = mtstop-mtstart;
			// printf("Atom %d Molecule %d Mass atom %f Mass molecule %f %f %f \n",
			//j, molid[j], molmass[j].x, molmass[j].y, molmass[j].z, molmass[j].w);
		}
	} //while

#ifdef USE_CONSFIX
	// consfix
	// populate seg_typeid
	// initialize the seg_typeid[i] = i for atom i
	for(i = 0; i < nAtom; i++){
		seg_typeid[i] = atom_typeid[i]; // i;
	}

	printf("Set up consfix array\n");
	// if the atom i is in the segment j then seg_typeid[i] = -10 * j
	if(consfix == 1){
		// printf("CONSFIX: Set up seg_typeid %d \n", consfix_nseg);
		j = 0; // number segments

		while(j <  consfix_nseg){
			i = 0; // number of atoms
			printf("CONSFIX: Segment to find %s \n", consfix_segname[j]);
			int k = 0;

			while(i < nAtom){

				// printf(" %s ", seg_type[i]);
				if(cicompare(consfix_segname[j], seg_type[i]) == 0){
					seg_typeid[i] = -10 * (j + 1);
					k++;
				}

				i++;
			} // while (i < nAtom)

			consfix_segcount[j] = k;

			j++;
		} // while (j <  consfix_nseg)
	} // if (consfix == 1)

	nAtomwoseg = nAtom;
	for(i = 0; i < consfix_nseg; i++) {
		nAtomwoseg -= consfix_segcount[i];
	}

	printf("CONSFIX: Number of amtoms wo segment %d %d \n",
	       consfix_nseg, nAtomwoseg);

	// adjust if consfix is define by setting the charge == 0
	// for atoms in any of the listed segments
	// coding: seg_typeid[i] = i for atoms that are NOT in any segment
	// coding: seg_typeid[i] = -10 * j  for atoms that are in segment j
	// for (i=0; i<nAtom; i++) {
          // printf("CONSFIX: %d %d %f - ", i, seg_typeid[i], r[i].w);
          // r[i].w = r[i].w * (seg_typeid[i] > -1);
          // printf("%d %f\n", (seg_typeid[i] > -1), r[i].w);
	// }

#endif

  // use rv.w for consfix info rather than mantaining an addtional array
	for(i = 0; i < nAtom; i++){
#ifdef CONSFIX
		rv[i].w = (seg_typeid[i] > -1) * 1.0f;
#else
    rv[i].w = 1.0f;
#endif
	}



#ifdef PME_CALC
	//eEwself = -1*CC*beta*eEwself/sqrt(PI);
	eEwself = -1 * CC * KAPPa * eEwself / sqrt(PI);
#endif
	//printf("EwSelf = %f\n", eEwself);

///////////////////////////Read and Assign Bonds////////////////////////////
	// read number of bonds
	fscanf(psffile, "%d %s %s", &j, line, line);
	Bond_Count = j;
	printf("Num Bonds = %d, ", Bond_Count);
	if(shake){
		printf("Building Constraints... ");
	}

	bonds_set_type[0] = 0;

	// subtract 1 from atom indices in psf since atoms here are
	//numbered 0 to nAtom-1
	for(i = 0; i < j; i++){
		fscanf(psffile, "%d %d", &x, &y);

		//add the atoms to bond, only if it is not a part of the constraints list...
		//if the mass is less than 1.5f, and shake on, then add the bond to
		//constraints...
		if((shake) && ((prm[type[x - 1]].x <= 1.5f) ||
		               (prm[type[y - 1]].x <= 1.5f))){

			add_to_constraints(x - 1, y - 1);
		}

		//1 to add, 0 to retrieve...
		h = add_retrieve_bonds_set_type(type[x - 1], type[y - 1], 1);

		//because h starts from 0
		Num_Hash_Bonds = (h + 1 > Num_Hash_Bonds) ? h + 1 : Num_Hash_Bonds;
		//h = HashBond_Manual(type[x-1], type[y-1]);

		//hash value to reference the parameter array...
		//bonds_set[MAX_BOND_COUNT*0 + (i+1)] = h;
		//bonds_set[MAX_BOND_COUNT*1 + (i+1)] = x-1;
		//bonds_set[MAX_BOND_COUNT*2 + (i+1)] = y-1;

		bonds_index[WorkgroupSize * (2 * bonds_index[x - 1] + 1) + x - 1] = h;
		bonds_index[WorkgroupSize * (2 * bonds_index[x - 1] + 2) + x - 1] = y - 1;

		bonds_index[WorkgroupSize * (2 * bonds_index[y - 1] + 1) + y - 1] = h;
		bonds_index[WorkgroupSize * (2 * bonds_index[y - 1] + 2) + y - 1] = x - 1;

		bonds_index[x - 1]++;
		bonds_index[y - 1]++;

	}
	//bonds_set[0] = j;

///////////////////////////Read and Assign Angles////////////////////////////
	// read number of angles
	fscanf(psffile, "%d %s %s", &j, line, line);
	Angle_Count = j;
	printf("Num Angles = %d, ", Angle_Count);

	angles_set = (int*)malloc(sizeof(int) * (4 * Angle_Count + 1));
	angles_set_type[0] = 0;
#ifdef UREY_BRADLEY
	ureyb_set_type[0] = 0;
#endif

	for(i = 0; i < j; i++){
		fscanf(psffile, "%d %d %d", &x, &y, &z);

		//1 to add, 0 to retrieve
		h = add_retrieve_angles_set_type(type[x - 1], type[y - 1], type[z - 1], 1);

		//because h starts from 0
		Num_Hash_Angles = (h + 1 > Num_Hash_Angles) ? h + 1 : Num_Hash_Angles;

		//h = HashAngle_Manual(type[x-1], type[y-1], type[z-1]);

		//hash value to reference the parameter array...
		angles_set[Angle_Count * 0 + (i + 1)] = h;
		angles_set[Angle_Count * 1 + (i + 1)] = x - 1;
		angles_set[Angle_Count * 2 + (i + 1)] = y - 1;
		angles_set[Angle_Count * 3 + (i + 1)] = z - 1;

		angles_index[WorkgroupSize * (angles_index[x - 1] + 1) + x - 1] = i;
		angles_index[WorkgroupSize * (angles_index[y - 1] + 1) + y - 1] = i;
		angles_index[WorkgroupSize * (angles_index[z - 1] + 1) + z - 1] = i;

		angles_index[x - 1]++;
		angles_index[y - 1]++;
		angles_index[z - 1]++;
	}
	angles_set[0] = j;

///////////////////////////Read and Assign Dihedrals////////////////////////////
	// read number of dihedrals
	fscanf(psffile, "%d %s %s", &j, line, line);
	Dihed_Count = j;
	printf("Num Diheds = %d, ", Dihed_Count);

	dihedrals_set = (int*)malloc(sizeof(int) * (5 * Dihed_Count + 1));
	dihedrals_set_type[0] = 0;

	for(i = 0; i < j; i++){
		fscanf(psffile, "%d %d %d %d", &x, &y, &z, &w);

		//add this dihedral type to dihedral_set_type if it is not already added...
		//1 to add, 0 to retrieve
		h = add_retrieve_dihedrals_set_type(type[x - 1], type[y - 1],
		                                    type[z - 1], type[w - 1], 1);

		//h = HashDihed_Manual(type[x-1], type[y-1], type[z-1], type[w-1]);

		//hash value to reference the parameter array...
		dihedrals_set[Dihed_Count * 0 + (i + 1)] = h;
		dihedrals_set[Dihed_Count * 1 + (i + 1)] = x - 1;
		dihedrals_set[Dihed_Count * 2 + (i + 1)] = y - 1;
		dihedrals_set[Dihed_Count * 3 + (i + 1)] = z - 1;
		dihedrals_set[Dihed_Count * 4 + (i + 1)] = w - 1;

		//add this dihedral index to corresponding atoms' dihedrals_index
		dihedrals_index[WorkgroupSize * (dihedrals_index[x - 1] + 1) + x - 1] = i;
		dihedrals_index[WorkgroupSize * (dihedrals_index[y - 1] + 1) + y - 1] = i;
		dihedrals_index[WorkgroupSize * (dihedrals_index[z - 1] + 1) + z - 1] = i;
		dihedrals_index[WorkgroupSize * (dihedrals_index[w - 1] + 1) + w - 1] = i;

		dihedrals_index[x - 1]++;
		dihedrals_index[y - 1]++;
		dihedrals_index[z - 1]++;
		dihedrals_index[w - 1]++;
	}
	dihedrals_set[0] = j;

	// read past improper entries
	fscanf(psffile, "%d %s %s", &j, line, line);

///////////////////////////Read and Assign Impropers////////////////////////////
#ifdef IMPROPER
	//initialize improper paramters...
	Improper_Count = j;
	printf("Num Impropers = %d\n", Improper_Count);

	impropers_set = (int*)malloc(sizeof(int) * (5 * Improper_Count + 1));
	impropers_set_type[0] = 0;
#endif

	for(i = 0; i < j; i++){
		fscanf(psffile, "%d %d %d %d", &x, &y, &z, &w);

#ifdef IMPROPER
		//add this improper type to impropers_set_type if it is not already added...
		//1 to add, 0 to retrieve
		h = add_retrieve_impropers_set_type(type[x - 1], type[y - 1],
		                                    type[z - 1], type[w - 1], 1);

		//hash value to reference the parameter array...
		impropers_set[Improper_Count * 0 + (i + 1)] = h;
		impropers_set[Improper_Count * 1 + (i + 1)] = x - 1;
		impropers_set[Improper_Count * 2 + (i + 1)] = y - 1;
		impropers_set[Improper_Count * 3 + (i + 1)] = z - 1;
		impropers_set[Improper_Count * 4 + (i + 1)] = w - 1;

		//add this improper index to corresponding atoms' impropers_index
		impropers_index[WorkgroupSize * (impropers_index[x - 1] + 1) + x - 1] = i;
		impropers_index[WorkgroupSize * (impropers_index[y - 1] + 1) + y - 1] = i;
		impropers_index[WorkgroupSize * (impropers_index[z - 1] + 1) + z - 1] = i;
		impropers_index[WorkgroupSize * (impropers_index[w - 1] + 1) + w - 1] = i;

		impropers_index[x - 1]++;
		impropers_index[y - 1]++;
		impropers_index[z - 1]++;
		impropers_index[w - 1]++;
#endif
	}

#ifdef IMPROPER
	impropers_set[0] = j;
#endif

	// read past donor entries
	fscanf(psffile, "%d %s %s", &j, line, line);

	for(i = 0; i < j; i++){
		fscanf(psffile, "%d %d", &x, &y);
	}

	// read past acceptor entries
	fscanf(psffile, "%d %s %s", &j, line, line);

	for(i = 0; i < j; i++){
		fscanf(psffile, "%d %d", &x, &y);
	}

	// read past nonbond entries
	fscanf(psffile, "%d %s", &j, line, line);

	for(i = 0; i < nAtom; i++){
		fscanf(psffile, "%d", &x);
	}

	// read past ngrp entries
	fscanf(psffile, "%d %s %s %s", &j, line, line, line);

	for(i = 0; i < j; i++){
		fscanf(psffile, "%d %d %d", &x, &y, &z);
	}

	// read molecule assignments
	fscanf(psffile, "%d %s", &j, line);

	for(i = 0; i < nAtom; i++){
		fscanf(psffile, "%d", &x);
		molid[i] = x - 1;
	}

	// if end of file is already reached, then something is wrong so exit
	if(feof(psffile)){
		fprintf(logfile, "Warning!! Unexpected end of file %s\n", psfFileName);
		//exit(-1);
	}

	// stop reading here, cross-terms etc. not needed
	fclose(psffile);
}

//------------------------------------------------------------------------------
int add_specific_dihedrals_set_type_to_prm(int w,
                                           int x,
                                           int y,
                                           int z,
                                           float a,
                                           int n,
                                           float c){
//------------------------------------------------------------------------------
	int h;
	int dcount;
	int k;
	int ntemp;

	float atemp;
	float ctemp;

	h = add_retrieve_dihedrals_set_type(w, x, y, z, 0);//1 to add, 0 to retrieve

	if(wildcardstatus[h] == 1){
		wildcardstatus[h] = 0;
		dcount = 0;
		//reset the ndihedprm list and populate with the explicit types..
		dihedral_type_count[h] = 0;
	}

	dcount = dihedral_type_count[h];

	//insert the dihedral in ascending order of n...

	for(k = 0; k < dcount; k++){

		if(dihedral_prm[h + k * MAX_DIHED_TYPE].n > n){
			atemp = dihedral_prm[h + k * MAX_DIHED_TYPE].x;
			ntemp = dihedral_prm[h + k * MAX_DIHED_TYPE].n;
			ctemp = dihedral_prm[h + k * MAX_DIHED_TYPE].d;

			dihedral_prm[h + k * MAX_DIHED_TYPE].x = a;
			dihedral_prm[h + k * MAX_DIHED_TYPE].n = n;
			dihedral_prm[h + k * MAX_DIHED_TYPE].d = c;

			a = atemp;
			n = ntemp;
			c = ctemp;
		}
	}

	//finished inserting..

	dihedral_prm[h + dcount * MAX_DIHED_TYPE].x = a;
	dihedral_prm[h + dcount * MAX_DIHED_TYPE].n = n;
	dihedral_prm[h + dcount * MAX_DIHED_TYPE].d = c;
	dihedral_type_count[h]++;

	fprintf(logfile, "%s\t%s\t%s\t%s\t", types_present[w], types_present[x],
	        types_present[y], types_present[z]);
	fprintf(logfile, "%f %d %f\t!%d %d %d %d\t hash=%d,n=%d\n", a, n, c, w, x,
	        y, z, h,dihedral_type_count[h]);

	return 1;
}

//------------------------------------------------------------------------------
int add_wildcard_dihedrals_set_type_to_prm(int x,
                                           int y,
                                           float a,
                                           int n,
                                           float c){
//------------------------------------------------------------------------------
	int i;
	int c1;
	int c2;
	int num_added = 0;
	int x1;
	int y1;
	int w1;
	int z1;
	int h;
	int dcount;

	int k;
	int ntemp;

	float atemp;
	float ctemp;

	for(i = 0; i < dihedrals_set_type[0]; i++){

		w1 = dihedrals_set_type[4 * i + 1];
		x1 = dihedrals_set_type[4 * i + 2];
		y1 = dihedrals_set_type[4 * i + 3];
		z1 = dihedrals_set_type[4 * i + 4];

		c1 = ((x1 == x) && (y1 == y));
		c2 = ((x1 == y) && (y1 == x));

		//this dihedral type's middle atoms are present in the system...
		if(c1 || c2){

			//1 to add, 0 to retrieve
			h = add_retrieve_dihedrals_set_type(w1, x1, y1, z1, 0);

			//only if the dihedral prm set is empty or already
			//populated by wildcard entries...
			if(wildcardstatus[h] == 1){

				dcount = dihedral_type_count[h];

				//insert the dihedral in ascending order of n...

				for(k = 0; k < dcount; k++){

					if(dihedral_prm[h + k * MAX_DIHED_TYPE].n > n){
						atemp = dihedral_prm[h + k * MAX_DIHED_TYPE].x;
						ntemp = dihedral_prm[h + k * MAX_DIHED_TYPE].n;
						ctemp = dihedral_prm[h + k * MAX_DIHED_TYPE].d;

						dihedral_prm[h + k * MAX_DIHED_TYPE].x = a;
						dihedral_prm[h + k * MAX_DIHED_TYPE].n = n;
						dihedral_prm[h + k * MAX_DIHED_TYPE].d = c;

						a = atemp;
						n = ntemp;
						c = ctemp;
					}
				}

				//finished inserting..

				dihedral_prm[h + dcount * MAX_DIHED_TYPE].x = a;
				dihedral_prm[h + dcount * MAX_DIHED_TYPE].n = n;
				dihedral_prm[h + dcount * MAX_DIHED_TYPE].d = c;

				dihedral_type_count[h]++;

				fprintf(logfile, "%s(X)\t%s\t%s\t%s(X)\t", types_present[w1],
				        types_present[x1], types_present[y1], types_present[z1]);
				fprintf(logfile, "%f %d %f\t!%d %d %d %d\t hash=%d,n=%d\n", a, n, c, w1,
				        x1, y1, z1, h,dihedral_type_count[h]);

				num_added++;
			}
		}//if (c1||c2){
	}//for(i=0; i<dihedrals_set_type[0]; i++)

	return num_added;
}

#ifdef IMPROPER
//------------------------------------------------------------------------------
int add_specific_impropers_set_type_to_prm(int w,
                                           int x,
                                           int y,
                                           int z,
                                           float a,
                                           float b,
                                           float c){
//------------------------------------------------------------------------------
	int h;//n

	h = add_retrieve_impropers_set_type(w, x, y, z, 0);//1 to add, 0 to retrieve

	//wildcardstatus[h]=0;

	improper_prm[h].x = a; //a/RAD2DEG;
	improper_prm[h].y = c;

	fprintf(logfile, "%d %d %d %d %f %f %f\t\t ! hash=%d\t\t\t\t", w, x, y, z, a,
	        b, c, h);
	fprintf(logfile, "%s\t%s\t%s\t%s\n", types_present[w], types_present[x],
	        types_present[y], types_present[z]);
	//printf("%d, ", h);
	return 1;
}

//------------------------------------------------------------------------------
int add_wildcard_impropers_set_type_to_prm(int w,
                                           int z,
                                           float a,
                                           float b,
                                           float c){
//------------------------------------------------------------------------------
	int i;
	int h;
	int w1;
	int x1;
	int y1;
	int z1;
	int num_added = 0;

	char c1;
	char c2;

	for(i = 0; i < impropers_set_type[0]; i++){

		w1 = impropers_set_type[4 * i + 1];
		x1 = impropers_set_type[4 * i + 2];
		y1 = impropers_set_type[4 * i + 3];
		z1 = impropers_set_type[4 * i + 4];

		c1 = (w1 == w) && (z1 == z);
		c2 = (w1 == z) && (z1 == w);

		if(c1 || c2){
			//1 to add, 0 to retrieve
			h = add_retrieve_impropers_set_type(w1, x1, y1, z1, 0);

			improper_prm[h].x = a; //a/RAD2DEG;
			improper_prm[h].y = c;

			fprintf(logfile, "%d %d %d %d %f %f %f\t\t ! hash=%d\t\t\t\t", w1, x1, y1,
			        z1, a, b, c, h);
			fprintf(logfile, "%s\t%s\t%s\t%s\n", types_present[w1], types_present[x1],
			        types_present[y1], types_present[z1]);

			num_added++;
		}//if ((w1==w)&&(z1==z))
	}//for(i=0; i<impropers_set_type[0]; i++)

	return num_added;
}
#endif

/*----------------------------------------------------------------------------*/
void ReadPrm_CHARMM(){
/*----------------------------------------------------------------------------*/

	//read coords, charges, parameters from prm file
	int j;
	int w;
	int x;
	int y;
	int z;
	int h;
	int n;

	float a;
	float b;
	float c;

	char line[9999];
	char temp_line[9999];
	char* descriptor;
	char delims[] = " \t\n\r";
	//char curr_type1[10], curr_type2[10], curr_type3[10], curr_type4[10];
	char* curr_type1;
	char* curr_type2;
	char* curr_type3;
	char* curr_type4;
	char* temp_str;

	int read_case = 0;
	float temp;
	float two_pow_sixth = (float)pow(2.0f, 1.0f / 6.0f);

	if(!(prmfile = fopen(prmFileName,"r"))){
		printf("File %s not found\n", prmFileName);
		exit(-1);
	}

	fprintf(logfile, "Reading %s...\n", prmFileName);

	while(fgets(line, 9999, prmfile)){
		//while (strcmp(line, "END")!=0){
		sprintf(temp_line, "%s", line);
		//printf("%s", line);
		descriptor = strtok(temp_line, delims);

		//first check if descriptor is NULL before cheking the first char..
		if(descriptor == NULL){
			continue;
		}

		//ignore comments...
		if((descriptor[0] == '!') ||
		   (descriptor[0] == '*') ||
		   (descriptor[0] == '\n')){

			continue;
		}

		if(strcmp(descriptor, "BONDS") == 0){
			read_case = 1;
			fprintf(logfile, "BONDS\n");
			//fgets(line, 9999, prmfile);//ignore the rest of the line...
			continue;
		}

		if(strcmp(descriptor, "ANGLES") == 0){
			//if (!strcmp(line, "ANGLES\n")) {
			read_case = 2;
			fprintf(logfile, "ANGLES\n");
			//fgets(line, 9999, prmfile);
			continue;
		}

		if(strcmp(descriptor, "DIHEDRALS") == 0){
			//if (!strcmp(line, "DIHEDRALS\n")) {
			read_case = 3;
			fprintf(logfile, "DIHEDRALS\n");
			//fgets(line, 9999, prmfile);
			continue;
		}

		if(strcmp(descriptor, "NONBONDED") == 0){
			//if (!strcmp(line, "NONBONDED\n")) {
			read_case = 4;
			fprintf(logfile, "NONBONDED\n");
			//fgets(line, 9999, prmfile);
			continue;
		}

		if(strcmp(descriptor, "NBFIX") == 0){
			//if (!strcmp(line, "NBFIX\n")){
			read_case = 5;
			fprintf(logfile, "NBFIX\n");
			//printf("%s...\n", line);
			//fgets(line, 9999, prmfile);
			continue;
		}

		if(strcmp(descriptor, "HBOND") == 0){
			//if (!strcmp(line, "HBOND\n")){//unused...
			read_case = 8;
			fprintf(logfile, "HBOND\n");
			//printf("%s...\n", line);
			//fgets(line, 9999, prmfile);
			continue;
		}

		if(strcmp(descriptor, "CMAP") == 0){
			//if (!strcmp(line, "CMAP\n")){//unused...
			read_case = 6;
			//printf("%s...\n", line);
			//fgets(line, 9999, prmfile);
			continue;
		}

		if(strcmp(descriptor, "IMPROPER") == 0){
			//if (!strcmp(line, "IMPROPER\n")){//unused...
			read_case = 7;
#ifdef IMPROPER
			fprintf(logfile, "IMPROPER\n");
#endif
			//fgets(line, 9999, prmfile);
			continue;
		}

		//else.. proceed with the current read_case...

		//sprintf(curr_type1, "%s", line);//copy the line information to type1..
		//because the first symbol of the text from file has already been read

		//Read BONDS...
		if(read_case == 1){
			//printf("%s...\n", line);
			curr_type1 = strtok(line, delims);
			curr_type2 = strtok(NULL, delims);
			//fscanf(prmfile, "%s", curr_type2);
			x = typetonum_present(curr_type1);
			y = typetonum_present(curr_type2);

			//1 to add, 0 to retrieve...
			if((h = add_retrieve_bonds_set_type(x, y, 0)) != -1){

				a = (float)atof(strtok(NULL, delims));
				b = (float)atof(strtok(NULL, delims));
				//fscanf(prmfile, "%f %f", &a, &b);

				//h = HashBond_Manual(x, y);

				bondprm[h].x = (a = 2 * a); // force constant
				bondprm[h].y = b; // equilibrium distance

				Num_BondTypes++;
				fprintf(logfile, "%d %d %f %f\t\t hash = %d\t\t",x, y, a, b, h);
				fprintf(logfile, "%s %s\n", curr_type1, curr_type2);
			}
			//fgets(line, 9999, prmfile);
			continue;
		}

		//Read ANGLES...
		if(read_case == 2){

			curr_type1 = strtok(line, delims);
			curr_type2 = strtok(NULL, delims);
			curr_type3 = strtok(NULL, delims);

			x = typetonum_present(curr_type1);
			y = typetonum_present(curr_type2);
			z = typetonum_present(curr_type3);

			//1 to add, 0 to retrieve
			if((h = add_retrieve_angles_set_type(x, y, z, 0)) != -1){

				a = (float)atof(strtok(NULL, delims));
				b = (float)atof(strtok(NULL, delims));

				//h = HashAngle_Manual(x, y, z);

				angleprm[h].x = a; // force constant
				angleprm[h].y = b / RAD2DEG; // ** convert to radians!! **
				// equilibrium angle

				Num_AngleTypes++;
				fprintf(logfile, "%d %d %d %f %f\t hashAngle = %d\t\t",
				        x, y, z, a, b, h);

#ifdef UREY_BRADLEY

				temp_str = strtok(NULL, delims);

				if((temp_str != NULL) && (atof(temp_str) != 0.0)){

					a = (float)atof(temp_str);
					b = (float)atof(strtok(NULL, delims));

					h = add_retrieve_ureyb_set_type(x, y, z, 1);
					ureybprm[h].x = a;
					ureybprm[h].y = b;
					Num_UreyBTypes++;

					fprintf(logfile, "%f %f\t hashUreyB = %d\t\t", a, b, h);
				}
#endif
				fprintf(logfile, "!%s %s %s\n", curr_type1, curr_type2, curr_type3);
			}

			continue;
		}

		//Read DIHED...
		if(read_case == 3){

			curr_type1 = strtok(line, delims);
			curr_type2 = strtok(NULL, delims);
			curr_type3 = strtok(NULL, delims);
			curr_type4 = strtok(NULL, delims);

			w = typetonum_present(curr_type1);
			x = typetonum_present(curr_type2);
			y = typetonum_present(curr_type3);
			z = typetonum_present(curr_type4);

			//1 to add, 0 to retrieve
			if((w != -1) &&
			   (x != -1) &&
			   (y != -1) &&
			   (z != -1) &&
			   (add_retrieve_dihedrals_set_type(w, x, y, z, 0) != -1)){

				a = (float)atof(strtok(NULL, delims));
				n = (int)atof(strtok(NULL, delims));
				// CHARMM 36
				c = (float)atof(strtok(NULL, delims));

				if((c == 0.0f) || (c == 180.0f)){
					c36 = 0;
				}
				else{
					c36 = 1;
				}

				c = c / RAD2DEG;

				//return 0 if overwritten or 1 if added new..
				Num_DihedTypes += add_specific_dihedrals_set_type_to_prm(w, x, y, z,
				                                                         a, n, c);
			}

			//wildcard entries with an "X" .. make sure 2nd and 3rd
			//atom types are present in the system...
			//1 to add, 0 to retrieve
			if((cicompare(curr_type1, "X") == 0) &&
			   (cicompare(curr_type4, "X") == 0) &&
			   (x != -1) &&
			   (y != -1) &&
			   (add_retrieve_dihedrals_set_type(-1, x, y, -1, 0) != -1)){

				a = (float)atof(strtok(NULL, delims));
				n = (int)atof(strtok(NULL, delims));
				// CHARMM 36
				c = (float)atof(strtok(NULL, delims));

				if((c == 0.0f) || (c == 180.0f)){
					c36 = 0;
				}
				else{
					c36 = 1;
				}

				c = c / RAD2DEG;

				// the number of wildcard entries added is returned;
				Num_DihedTypes += add_wildcard_dihedrals_set_type_to_prm(x, y, a, n, c);
			}
			continue;
		}//if (read_case==3){//Read DIHED...

		//Read NONBOND...
		if(read_case == 4){

			curr_type1 = strtok(line, delims);
			if((w = typetonum_present(curr_type1)) != -1){
				temp = (float)atof(strtok(NULL, delims));
				a = (float)atof(strtok(NULL, delims));
				b = (float)atof(strtok(NULL, delims));
				//fscanf(prmfile, "%f %f %f %c", &temp, &a, &b);

				//not excluded
				if(temp == 0.0){
					prm[w].y = (a = abs(a));
					prm[w].z = (b = b / two_pow_sixth);
				}
				//excluded
				else if(temp == 1.0){
					prm[w].y = (a = 0.0);
					prm[w].z = (b = 0.0);
				}

				fprintf(logfile, "%d %f %f %f\t", w, temp, a, b);

#if (NBXMOD==5)
				//copy the existing to 1-4 parameters.. rewrite
				//with special 1-4 parameters if present
				prm1_4[w].x = a;
				prm1_4[w].y = b;

				float d;
				descriptor = strtok(NULL, delims);
				if(descriptor != NULL){
					//not commented and is numeric..
					if((descriptor[0] != '!') &&
					   ((descriptor[0] == '0') || (descriptor[0] == '1'))){

						temp = (float)atof(descriptor);
						c = (float)atof(strtok(NULL, delims));
						d = (float)atof(strtok(NULL, delims));

						if(temp == 0.0){
							prm1_4[w].x = (c = abs(c));
							prm1_4[w].y = (d = d / two_pow_sixth);
						}

						if(temp == 1.0){
							prm1_4[w].x = (c = 0.0);
							prm1_4[w].y = (d = 0.0);
						}

						fprintf(logfile, "%f %f\t", c, d);
					}//if (descriptor[0]!='!')
					else if(descriptor[0] != '!'){
						printf("Unrecognized prm format.. check line \"%s\"", temp_line);
						exit(0);
					}
				}//if (descriptor!=NULL)
#endif
				Num_NBTypes++;
				fprintf(logfile, "!\t\t%s \n", curr_type1);
			}

			//printf("%s %d %s %d\n", curr_type1, w, curr_type2, x);
			//fgets(line, 9999, prmfile);
			continue;
		}//Read NONBOND...

		//Read NBFIX...
		if(read_case==5){
			//printf("%s...\n", line);
			curr_type1 = strtok(line, delims);
			curr_type2 = strtok(NULL, delims);
			//fscanf(prmfile, "%s", curr_type2);
			x = typetonum_present(curr_type1);
			y = typetonum_present(curr_type2);
			// printf("NBFIX: %s %s\n", curr_type1, curr_type2);
			// printf("NBFIX: %d %d\n", x, y);

			// if ((h = add_retrieve_nbfix_set_type(x, y, 1))!=-1){
			//1 to add, 0 to retrieve...
			a = (float)atof(strtok(NULL, delims));
			b = (float)atof(strtok(NULL, delims));
			//fscanf(prmfile, "%f %f", &a, &b);

			// mantain matrix symmetry
			//h = HashBond_Manual(x, y);
			h = x * num_types_present + y;

			// __my_fmul(4.0f, parameter.x);
			// __my_fmul(parameter.y,parameter.y);
			// nbfixprm[h].x = a;   // force constant
			// nbfixprm[h].y = b;   // equilibrium distance
			nbfixprm[h].x = 4.0f * abs(a);   // force constant
			// equilibrium distance
			nbfixprm[h].y = (b / two_pow_sixth) * (b / two_pow_sixth);

			h = y * num_types_present + x;

			// nbfixprm[h].x = a;   // force constant
			// nbfixprm[h].y = b;   // equilibrium distance
			nbfixprm[h].x = 4.0f * abs(a);   // force constant
			// equilibrium distance
			nbfixprm[h].y = (b / two_pow_sixth) * (b / two_pow_sixth);

			// Num_BondTypes++;
			fprintf(logfile, "%d %d %f %f\t\t hash = %d\t\t\n",x, y, a, b, h);
			fprintf(logfile, "%s %s\n", curr_type1, curr_type2);
			// printf("NBFIX: %d %d %f %f\t\t hash = %d\t\t\n",x, y, a, b, h);
			// printf("NBFIX: %s %s\n", curr_type1, curr_type2);
			// }

			//fgets(line, 9999, prmfile);
			continue;
		}

#ifdef IMPROPER
		//Read IMPROPER...
		if(read_case == 7){

			curr_type1 = strtok(line, delims);
			curr_type2 = strtok(NULL, delims);
			curr_type3 = strtok(NULL, delims);
			curr_type4 = strtok(NULL, delims);

			w = typetonum_present(curr_type1);
			x = typetonum_present(curr_type2);
			y = typetonum_present(curr_type3);
			z = typetonum_present(curr_type4);

			//1 to add, 0 to retrieve
			if((w != -1) &&
			   (x != -1) &&
			   (y != -1) &&
			   (z != -1) &&
			   (add_retrieve_impropers_set_type(w, x, y, z, 0) != -1)){

				a = (float)atof(strtok(NULL, delims));
				//middle column is not used for improper entries...
				b = (float)atof(strtok(NULL, delims));
				c = (float)atof(strtok(NULL, delims));

				//return 0 if overwritten or 1 if added new..
				Num_ImprTypes += add_specific_impropers_set_type_to_prm(w, x, y, z,
				                                                        a, b, c);
			}

			//wildcard entries with an "X" .. make sure 1st and 4th
			//atom types are present in the system...
			//1 to add, 0 to retrieve
			if((cicompare(curr_type2, "X") == 0) &&
			   (cicompare(curr_type3, "X") == 0) &&
			   (w != -1) &&
			   (z != -1) &&
			   (add_retrieve_impropers_set_type(w, -1, -1, z, 0) != -1)){

				a = (float)atof(strtok(NULL, delims));
				b = (float)atof(strtok(NULL, delims));
				c = (float)atof(strtok(NULL, delims));

				// the number of wildcard entries added is returned;
				Num_ImprTypes += add_wildcard_impropers_set_type_to_prm(w, z, a, b, c);
			}
			continue;

		}//Read IMPROPER...

#endif //#ifdef IMPROPER

		if((read_case > 4) || (read_case == 0)){
			//fgets(line, 9999, prmfile);
			continue;
		}

	}//while(strcmp(line, "END")!=0)

	if(Num_Hash_Bonds != Num_BondTypes){
		printf("WARNING!!! Number of Bond types in prm (%d) does not match that "
		       "in psf (%d). Energy values may be incorrect!!!\n",
		       Num_BondTypes, Num_Hash_Bonds);
	}
	if(Num_Hash_Angles != Num_AngleTypes){
		printf("WARNING!!! Number of Angle types in prm (%d) does not match that "
		       "in psf (%d). Energy values may be incorrect!!!\n",
		       Num_AngleTypes, Num_Hash_Angles);
	}

	fprintf(logfile, "NONBONDED=%d\tBONDS=%d\tANGLES=%d\tUREY-BRADLEY=%d\t"
	        "DIHEDRALS(including folds)=%d\tIMPROPERS=%d\n",
	        Num_NBTypes, Num_BondTypes, Num_AngleTypes, Num_UreyBTypes,
	        Num_DihedTypes, Num_ImprTypes);

	// close prm file
	fclose(prmfile);

	// set charges from nAtom to WorkgroupSize to NaN as a code
	// so they are not accumulated in acceleration
	for(j = nAtom; j < WorkgroupSize; j++){
		r[j].w = NAN;
	}

	//fclose(logfile);
/*
	//////////////
	//Check for any unset parameter values...
	//////////////
	for (j=0; j<bonds_set_type[0]; j++){
		id1 = bonds_set_type[2*i + 1];
		id2 = bonds_set_type[2*i + 2];
		h = add_retrieve_bonds_set_type(id1, id2, 0);
		if (__my_isnan(bondprm[h].x)||__my_isnan(bondprm[h].y)){
			printf("Bond Parameter for bond type %s - %s is not properly set."
			" Make sure the prm file is in the CHARMM format",
			type_2_name[id1], type_2_name[id2]);
			incomplete_parameter = 1;
		}

	for (j=0; j<angles_set_type[0]; j++){
		id1 = angles_set_type[2*i + 1];
		id2 = angles_set_type[2*i + 2];
		id3 = angles_set_type[2*i + 3];
		h = add_retrieve_angles_set_type(id1, id2, id3, 0);
		if (__my_isnan(angleprm[h].x)||__my_isnan(angleprm[h].y)){
			printf("Angle Parameter for angle type %s - %s - %s is not properly set."
			" Make sure the prm file is in the CHARMM format", \
				type_2_name[id1], type_2_name[id2], type_2_name[id3]);
				incomplete_parameter = 1;
		}


	for (j=0; j<dihedrals_set_type[0]; j++){
		id1 = dihedrals_set_type[2*i + 1];
		id2 = dihedrals_set_type[2*i + 2];
		id3 = dihedrals_set_type[2*i + 3];
		id4 = dihedrals_set_type[2*i + 4];
		h = add_retrieve_dihedrals_set_type(id1, id2, id3, id4, 0);
		for (k =0; k< dihedral_type_count[h]; k++){
			if (__my_isnan(dihedral_prm[h+k*MAX_DIHED_TYPE].x)
				||__my_isnan(dihedral_prm[h+k*MAX_DIHED_TYPE].n)
				||__my_isnan(dihedral_prm[h+k*MAX_DIHED_TYPE].d)){
				printf("Dihedral Parameter for dihed type %s - %s - %s - %s and"
				" multiplicity %d is not properly set. Make sure the prm file is"
				" in the CHARMM format", \
				type_2_name[id1], type_2_name[id2], type_2_name[id3],
				type_2_name[id4], k);
			incomplete_parameter = 1;
			}
		}
	}

	//////////////
	//end check parameter values...
	//////////////
	*/

        // Feed nbfix paramters still missing
        /*
	printf("NBFIX: before %d\n", num_types_present);
	for (int i=0;i<num_types_present; i++){
	  for (int j=0;j<num_types_present; j++){
	    printf("%f %f ", nbfixprm[i*num_types_present+j].x,"
			" nbfixprm[i*num_types_present+j].y);
	  }
	  printf("\n");
	}
       */

	// initialize atomic eps and sigma  is NAN
	printf("NBFIX: %d\n", num_types_present);
	for(int i = 0; i < num_types_present; i++){
		for(int j = 0; j < num_types_present; j++){
			if(__myisnan(nbfixprm[i * num_types_present + j].x)){
				nbfixprm[i * num_types_present + j].x = 4.0f * sqrtf(__my_fmul(prm[i].y,
				                                                             prm[j].y));
			}
			if(__myisnan(nbfixprm[i*num_types_present+j].y)){
				nbfixprm[i * num_types_present + j].y = __my_fadd(prm[i].z, prm[j].z) *
				                                        __my_fadd(prm[i].z, prm[j].z);
			}
		}
	}

	/*
	printf("NBFIX: after %d\n", num_types_present);
	for (int i=0;i<num_types_present; i++){
	  for (int j=0;j<num_types_present; j++){
	    printf("%f %f ", nbfixprm[i*num_types_present+j].x,
			nbfixprm[i*num_types_present+j].y);
	  }
	  printf("\n");
	}
	*/

	return;
}

//----------------------------------------------------------------------------
char Search_excl_binary_host(int* excl, int x, int y, char start, char end){
//----------------------------------------------------------------------------
//performs a binary search on the exclusion list ...
//----------------------------------------------------------------------------
	int mid;
	int neighbor_id;

	for(mid = 1; mid <= excl[x]; mid++){
		printf("%d\t", excl[mid * WorkgroupSize + x]);
	}
	printf("Searching for %d\n", y);
	do{
		mid = start + (end - start) / 2;
		printf("%d %d %d -\t%d\n", (int)start, (int)mid, (int)end,
		       excl[mid * WorkgroupSize + x]);

		neighbor_id = excl[mid * WorkgroupSize + x];

		if(y > neighbor_id){
			start = mid + 1;
		}
		else{
			end = mid - 1;
		}
	}while((neighbor_id != y) && (start <= end));

	if(neighbor_id == y){
		return 1;
	}
	else{
		return 0;
	}
}

//----------------------------------------------------------------------------
int SearchExcl(int x, int y){
//------------------------------------------------------------------------------
//search exclusion list for a given atom x and return the position of
//the first occurrence of a given value y
//additionally, it is assumed that the bonds atoms are
//already added to the exclusion list..
//so no need to check for the empty list condition
//------------------------------------------------------------------------------
	int j;
	int r;
	int done;

	done = 0;
	r = -1;

	for(j = 1; j <= excllist[x] && !done; j++){
		if(excllist[WorkgroupSize * j + x] == y){
			done = 1;
			r = j;
		}
	}

	return r;
}

//----------------------------------------------------------------------------
void SearchInsertExcl(int x, int y){
//----------------------------------------------------------------------------
//Searches and inserts atom id y, in the exclusion list of x ordered by atom id
//if y already exists due to cyclic redundancy of bonds, or angles,
//nothing is done...
//if not the atom id y, it is inserted in appropriate position...
//additionally, it is assumed that the bonds atoms are already
//added to the exclusion list..
//so no need to check for the empty list condition
//----------------------------------------------------------------------------

	int temp1;
	int temp2;
	int j;

	char insert = 0;
	char add_element = 0;

	//if y is greater than all the elements in the list..
	if((y > excllist[WorkgroupSize * excllist[x] + x]) || (excllist[x] == 0)){
		add_element = 1;
		temp1 = y;
	}

	for(j = 1; j <= excllist[x] && !add_element; j++){

		if(y == excllist[WorkgroupSize * j + x]){
			return;
		}

		if(insert){
			temp2 = excllist[WorkgroupSize * j + x];
			excllist[WorkgroupSize * j + x] = temp1;
			temp1 = temp2;
		}

		if((y < excllist[WorkgroupSize * j + x]) && (!insert)){
			insert = 1;
			temp1 = excllist[WorkgroupSize * j + x];
			excllist[WorkgroupSize * j + x] = y;
		}
	}//for (j=1; j<=excllist[x]; j++)

	if(insert || add_element){
		excllist[x]++;
		excllist[WorkgroupSize * excllist[x] + x] = temp1;
	}

	return;
}

//----------------------------------------------------------------------------
void checkexcllistorder(){
//----------------------------------------------------------------------------
	for(int i = 0; i < nAtom; i++){
		for(int j = 2; j <= excllist[i]; j++){
			if(excllist[j * WorkgroupSize + i] <=
			   excllist[(j - 1) * WorkgroupSize + i]){

				printf("Error in order of %d\n", i);
			}
		}
	}

	return;
}

//----------------------------------------------------------------------------
int Generate_ExclBitVector(){
//----------------------------------------------------------------------------
	int i;
	int j;
	int base_atom;

	unsigned long long one = 1;
	unsigned long long temp_vec;

	int excl_overflow = 0;
	int diff;

	for(i = 0; i < nAtom; i++){
		excl_bitvec[i] = 0;
		excl_bitvec_offset[i] = 0;
		if(excllist[WorkgroupSize * 0 + i] > 0){
			base_atom = min(i, excllist[WorkgroupSize * 1 + i]);
			excl_bitvec_offset[i] = i - base_atom;
		}

		for(j = 1; j <= excllist[WorkgroupSize * 0 + i]; j++){
			diff = (excllist[WorkgroupSize * j + i] - base_atom);
			if(diff >= 64){
				fprintf(logfile, "Exclusion overflow in atom %d: atoms %d and %d "
				        "are more than 64 apart...\n", i, base_atom,
				        excllist[WorkgroupSize * j + i]);

				excl_overflow = 1;
			}

			temp_vec = one << diff;
			excl_bitvec[i] |= temp_vec;
		}
	}
	/*
	if (excl_error){
		printf("Exclusion Errors in the system, Please increase the exclusion"
		       " bitvector size or perform binary search on exclusion list...");
		exit(0);
	}
	*/

	return excl_overflow;
}

//----------------------------------------------------------------------------
void GenExcl_minimal_new(){
//----------------------------------------------------------------------------
//generate nonbond exclusion lists from bond, angle, and dihedral lists
//----------------------------------------------------------------------------

	int i;
	int j;
	int n;
	int n1;
	int n2;
	int n3;
	int n4;
	int ai;
	int di;//n2,

	for(i = 0; i < nAtom; i++){

		// //start by adding the atom to its own exclusion list...
		//excllist[i]++;
		//excllist[WorkgroupSize*excllist[i] + i] = i;

		 //add atoms in the bond list...
		for(j = 0; j < bonds_index[i]; j++){
			n = bonds_index[WorkgroupSize * (2 * j + 2) + i];

			SearchInsertExcl(i, n);
			//excllist[i]++;
			//excllist[WorkgroupSize*excllist[i] + i] = n;
		}
		//printf("%d ", i);

	}//for(i=0; i<nAtom; i++) {

	//add the 1-3 angle atoms to each other's exclusion list
	for(ai = 0; ai < angles_set[0]; ai++){

			//angles_set[Angle_Count*0 + (ai + 1)] is the hash value...
			n1 = angles_set[Angle_Count * 1 + (ai + 1)];
			//n2 = angles_set[Angle_Count*2 + (ai + 1)];
			n3 = angles_set[Angle_Count * 3 + (ai + 1)];

			SearchInsertExcl(n1, n3);
			SearchInsertExcl(n3, n1);

		/*
			//search the exclusion list in case of cyclic redundancies...
			if ((SearchExcl(n1,n3) == -1)){
				excllist[n1]++;
				excllist[WorkgroupSize*excllist[n1] + n1] = n3;
			}

			if ((SearchExcl(n3,n1) == -1)){
				excllist[n3]++;
				excllist[WorkgroupSize*excllist[n3] + n3] = n1;
			}
		*/

	}//for(ai=0; ai<angles_set[0]; ai++){

	//copy the exclusion list with bonds and angles into ewlist...
	memcpy(ewlist, excllist, isize * (EXCL_COUNT_PERATOM + 1));

//#if (NBXMOD>=4)
	//add the 1-4 dihedral atoms to each other's exclusion list
	//search the exclusion list in case of cyclic redundancies...
	for(di = 0; di < dihedrals_set[0]; di++){

		//dihedrals_set[Dihed_Count*0 + (di + 1)] is the hash value...
		n1 = dihedrals_set[Dihed_Count * 1 + (di + 1)];
		n2 = dihedrals_set[Dihed_Count * 2 + (di + 1)];
		n3 = dihedrals_set[Dihed_Count * 3 + (di + 1)];
		n4 = dihedrals_set[Dihed_Count * 4 + (di + 1)];

		SearchInsertExcl(n1, n4);
		SearchInsertExcl(n4, n1);

		/*
		//search the exclusion list in case of cyclic redundancies...
		if ((SearchExcl(n1,n4) == -1)){
		excllist[n1]++;
		excllist[WorkgroupSize*excllist[n1] + n1] = n4;
		}

		if ((SearchExcl(n4,n1) == -1)){
		excllist[n4]++;
		excllist[WorkgroupSize*excllist[n4] + n4] = n1;
		}
		*/

	}//for(di=0; j<dihedrals_set[0]; di++){
//#endif

#ifdef UREY_BRADLEY
	int h;
	//build the Urey-Bradley list...
	for(ai = 0; ai < angles_set[0]; ai++){

		//angles_set[Angle_Count*0 + (ai+1)] is the hash value...
		n1 = angles_set[Angle_Count * 1 + (ai + 1)];
		n2 = angles_set[Angle_Count * 2 + (ai + 1)];
		n3 = angles_set[Angle_Count * 3 + (ai + 1)];

		if((h = add_retrieve_ureyb_set_type(type[n1],
		                                    type[n2], type[n3], 0)) != -1){

			ureyb_index[(2 * ureyb_index[n1] + 1) * WorkgroupSize + n1] = h;
			ureyb_index[(2 * ureyb_index[n1] + 2) * WorkgroupSize + n1] = n3;

			ureyb_index[(2 * ureyb_index[n3] + 1) * WorkgroupSize + n3] = h;
			ureyb_index[(2 * ureyb_index[n3] + 2) * WorkgroupSize + n3] = n1;

			ureyb_index[n1]++;
			ureyb_index[n3]++;
		}
	}
#endif
	int excl_overflow	= Generate_ExclBitVector();
	//capture_Excllist();
	//checkexcllistorder();
	//int x = 12771, y=12776;
	//printf("Answer = %d",
	//       (int)Search_excl_binary_host(excllist, x, y, 2, excllist[x]-1));
	//exit(0);

	if(excl_overflow){
		printf("The System has Exclusion overflow...see logfile for more "
		       "information.. Switching to Exclusion list.\n");
		NBBuild_Kernel = &nbbuild_excllist;
	}
	else{
		NBBuild_Kernel = &nbbuild_exclbitvec;
    }

	return;
}
