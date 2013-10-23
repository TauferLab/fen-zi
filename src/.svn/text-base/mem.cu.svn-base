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
#include "globals.h"


void CudaAllocateCopy(void **globalVarPointer,
                      void *hostPointer,
                      size_t size,
                      const char* var_name){

	printf("%s : %7.2f MB\n", var_name, ((float)size) / (1024 * 1024));
	cudaMalloc(globalVarPointer, size);

	if(hostPointer != NULL){
		cudaMemcpy(*globalVarPointer, hostPointer, size, cudaMemcpyHostToDevice);
	}

	checkCUDAError(var_name);
	device_memory_usage += (int)size;
	return;
}

void CudaAllocate(void **globalVarPointer,
                  size_t size,
                  const char* Variable_Name){

	printf("%s : %7.2f MB\n", Variable_Name, ((float)size) / (1024 * 1024));
	cudaMalloc(globalVarPointer, size);

	checkCUDAError(Variable_Name);
	//if( cudaPeekAtLastError() != cudaSuccess){
	//	printf("%s\n", cudaGetErrorString( cudaPeekAtLastError() ) );
	//}
	device_memory_usage += (int)size;
	return;
}

void InitMem(){
/*------------------------------------------------------------------------------
	allocate device and host memory and copy data to device
------------------------------------------------------------------------------*/
	printf("-----------------------\n Device Memory Usage\n"
	       "-----------------------\n");
	//determine amount of dynamic shared memory per block for positions
	shSize = BlockSize * sizeof(float4);

	//allocate memory on device
	CudaAllocateCopy((void**)&r4d, r, f4size, "Positn");
	CudaAllocateCopy((void**)&r4shaked, r, f4size, "Positn");
	CudaAllocate((void**)&roldd, f4size, "PrePos");

#ifdef CELL5
	CellSize.x = 5.0f;
	CellSize.y = 5.0f;
	CellSize.z = 5.0f;
#else
	CellSize.x = CutMax;
	CellSize.y = CutMax;
	CellSize.z = CutMax;
#endif

	NumCells.x = ((int)ceil(Region.x / CellSize.x));
	NumCells.y = ((int)ceil(Region.y / CellSize.y));
	NumCells.z = ((int)ceil(Region.z / CellSize.z));

	NumCells.w = NumCells.x * NumCells.y * NumCells.z;

#ifdef PCONSTANT
	MaxNumCells.x = NumCells.x + 2;
	MaxNumCells.y = NumCells.y + 2;
	MaxNumCells.z = NumCells.z + 2;
	MaxNumCells.w = MaxNumCells.x * MaxNumCells.y * MaxNumCells.z;
#endif

#ifdef PCONSTANT  //Allocate different amount of cell memory if PCONSTANT
	CudaAllocate((void**)&num_nonbond, MaxNumCells.w * sizeof(int), "NumNnB");
	cudaMemset(num_nonbond, 0, MaxNumCells.w * sizeof(int));

	CudaAllocate((void**)&cell_nonbond,
	             MaxNumCells.w * CELL_ATOMS * sizeof(unsigned int),
	             "CellNB");
#else
	CudaAllocate((void**)&num_nonbond, NumCells.w * sizeof(int), "NumNnB");
	cudaMemset(num_nonbond, 0, NumCells.w * sizeof(int));

	CudaAllocate((void**)&cell_nonbond,
	             NumCells.w * CELL_ATOMS * sizeof(unsigned int),
	             "CellNB");
#endif

	cudaMemcpyToSymbol(CELL_X, &CellSize.x, sizeof(float));
	cudaMemcpyToSymbol(CELL_Y, &CellSize.y, sizeof(float));
	cudaMemcpyToSymbol(CELL_Z, &CellSize.z, sizeof(float));

#ifdef PCONSTANT
	//use texture memory if PCONSTANT or else use constant memory...
	CudaAllocateCopy((void**)&numcellsd, &NumCells, sizeof(int4), "NumCel");
	cudaBindTexture(0, texnumcells, numcellsd, sizeof(int4));

	cudaMemcpyToSymbol(max_numcells, &MaxNumCells, sizeof(MaxNumCells));
#else
	cudaMemcpyToSymbol(num_cells, &NumCells, sizeof(NumCells));
	cudaMemcpyToSymbol(total_cells, &NumCells.w, sizeof(int));

	//cudaMemcpyToSymbol(num_cells_x, &NumCells.x, sizeof(int));
	//cudaMemcpyToSymbol(num_cells_y, &NumCells.y, sizeof(int));
	//cudaMemcpyToSymbol(num_cells_z, &NumCells.z, sizeof(int));
#endif

	//NBBuild_dimBlock and dimGrid used to build nonbond list with shared memory
	NBBuild_dimBlock.x = CELL_BLOCKSIZE;
	NBBuild_dimBlock.y = 1;
	NBBuild_dimBlock.z = 1;

	//divide the components among x and y dimensions
	//to stay well within the maximum limit...
	NBBuild_dimGrid.x = NumCells.x * NumCells.y;
	NBBuild_dimGrid.y = NumCells.z;
	NBBuild_dimGrid.z = 1;

		if(NBBuild_dimBlock.x > 1024) { // 512){
		printf("Number of threads for NB Build has been reached"
		       ", please reset the number of threads...");
		exit(-1);
	}

	if((NBBuild_dimGrid.x >= 65535) || (NBBuild_dimGrid.x >= 65535)){
		printf("Maximum Grid dimension for NBCELL_dimGrid is reached,"
		       " please reset the number of threads...");
		exit(-1);
	}

#ifdef PCONSTANT
	//Cellclean dimBlock and dimGrid is used for the Cell clean...
	CellClean_dimBlock.x = MaxNumCells.x;
	CellClean_dimBlock.y = 1;
	CellClean_dimBlock.z = 1;

	CellClean_dimGrid.x = MaxNumCells.y;
	CellClean_dimGrid.y = MaxNumCells.z;
	CellClean_dimGrid.z = 1;
#else
	CellClean_dimBlock.x = NumCells.x;
	CellClean_dimBlock.y = 1;
	CellClean_dimBlock.z = 1;

	CellClean_dimGrid.x = NumCells.y;
	CellClean_dimGrid.y = NumCells.z;
	CellClean_dimGrid.z = 1;
#endif

	if(CellClean_dimBlock.x > 512){
		printf("Maximum number of threads for CellClean has been reached,"
		       " please reset the number of threads...");
		exit(-1);
	}

	if((CellClean_dimGrid.x >= 65535) || (CellClean_dimGrid.x >= 65535)){
		printf("Maximum Grid dimension for CellClean_dimGrid has been reached,"
		       " please reset the number of threads...");
		exit(-1);
	}


//////////////////////////////////////////////
#ifdef PME_CALC
//////////////////////////////////////////////

	Total_PME_Lattices = fftx * ffty * fftz;

	//copy spline values to device
	CudaAllocate((void**)&M4d,
	             sizeof(float) * (NUM_SPLINE_SAMPLES + 1),
	             "M4DSpl");

	CudaAllocate((void**)&dM4d,
	             sizeof(float) * (NUM_SPLINE_SAMPLES + 1),
	             "dM4DSp");

	CudaAllocate((void**)&r4d_scaled, f4size, "ScaleR");
	CudaAllocate((void**)&prev_r4d_scaled, f4size, "PrevSR");
	CudaAllocate((void**)&disp, i4size, "DispIn");

	ePME32 = (float*)malloc(fsize);
	ePME = (double*)malloc(fsize);
	CudaAllocate((void**)&ePMEd, fsize, "enPMEd");

	eEwexcl32 = (float*)malloc(NBPART * fsize);
	eEwexcl = (double*)malloc(NBPART * fsize);
	CudaAllocate((void**)&eEwexcld, NBPART * fsize, "enEwex");

	//numLh = (int*) malloc(Total_PME_Lattices*sizeof(int));
	CudaAllocate((void**)&Qd,
	             sizeof(cufftComplex) * Total_PME_Lattices,
	             "QdCFFT");

	CudaAllocate((void**)&numL, Total_PME_Lattices * sizeof(int), "LatNum");

	CudaAllocate((void**)&cellL,
	             Total_PME_Lattices * MAXLATTICE_NEIGHBORS * sizeof(int),
	             "LatLst");
	//spline values are copied in InitDeviceConstants();

	cudaMemcpyToSymbol(FFTX, &fftx, sizeof(fftx));
	cudaMemcpyToSymbol(FFTY, &ffty, sizeof(ffty));
	cudaMemcpyToSymbol(FFTZ, &fftz, sizeof(fftz));

	float fftxh = ((float)fftx) / 2;
	float fftyh = ((float)ffty) / 2;
	float fftzh = ((float)fftz) / 2;
	cudaMemcpyToSymbol(FFTXH, &fftxh, sizeof(fftxh));
	cudaMemcpyToSymbol(FFTYH, &fftyh, sizeof(fftyh));
	cudaMemcpyToSymbol(FFTZH, &fftzh, sizeof(fftzh));

	float4 fftfrac;
	fftfrac.x = ((float)fftx) / Region.x;
	fftfrac.y = ((float)ffty) / Region.y;
	fftfrac.z = ((float)fftz) / Region.z;
	fftfrac.w = fftfrac.x * fftfrac.y * fftfrac.z;

#ifdef PCONSTANT
	CudaAllocateCopy((void**)&fracd, &fftfrac, sizeof(float4), "Frcxyz");
	cudaBindTexture(0, texfrac, fracd, sizeof(float4));
#else
	cudaMemcpyToSymbol(frac, &fftfrac, sizeof(float4));
#endif

	cudaMemcpyToSymbol(TOTAL_PME_LATTICES, &Total_PME_Lattices, sizeof(int));

	//Create a plan for 3D FFT
	cufftPlan3d( &plan, fftz, ffty, fftx, CUFFT_C2C);

//////////////////////////////////////////////
#endif//#ifdef PME_CALC
//////////////////////////////////////////////


#ifdef USE_NPT
	CudaAllocateCopy((void**)&p1d, p1h, f4size, "P1momento");
	CudaAllocateCopy((void**)&p2d, p2h, f4size, "P2momento");
	CudaAllocateCopy((void**)&p3d, p3h, f4size, "P3momento");
	CudaAllocateCopy((void**)&p4d, p4h, f4size, "P4momento");
	CudaAllocateCopy((void**)&r1d, r1h, f4size, "Coord temp 1");
#endif

	CudaAllocateCopy((void**)&v4d, rv, f4size, "Veloct");
	cudaMemcpy(v4d, rv, f4size, cudaMemcpyHostToDevice);
	//cudaMemset(v4d, 0, f4size);

	CudaAllocateCopy((void**)&f4d, f4h, f4size, "ForceD");
	cudaMemset(f4d, 0, f4size);

	CudaAllocateCopy((void**)&f4d_bonded, f4h_bonded, BPART * f4size, "ForceD");
	cudaMemset(f4d_bonded, 0, f4size);

	CudaAllocateCopy((void**)&f4d_nonbond,
	                 f4h_nonbond,
	                 NBPART * f4size,
	                 "ForceD");

	cudaMemset(f4d_nonbond, 0, f4size);

#ifdef USE_CONSFIX
#ifdef DEBUG_CONSFIX
	CudaAllocateCopy((void**)&f4d_nonbond0,
	                 f4h_nonbond0,
	                 NBPART * f4size,
	                 "ForceD0");

	cudaMemset(f4d_nonbond0, 0, f4size);

	CudaAllocateCopy((void**)&f4d_nonbond1,
	                 f4h_nonbond1,
	                 NBPART * f4size,
	                 "ForceD1");

	cudaMemset(f4d_nonbond1, 0, f4size);
#endif
#endif

	CudaAllocateCopy((void**)&min_r4d, r, f4size, "MinPos");

	CudaAllocate((void**)&prev_f4d, f4size, "PrvFrc");
	cudaMemset(prev_f4d, 0, f4size);

	CudaAllocate((void**)&Htd, f3size, "HtdMin");
	cudaMemset(Htd, 0, f3size);

	CudaAllocate((void**)&nupdated, sizeof(int), "Nupdat");

	ebnd = (float*)malloc(BPART * fsize);
	CudaAllocate((void**)&ebndd, BPART * fsize, "enBond");
	cudaMemset(ebndd, 0, BPART * fsize);

	eang = (float*)malloc(BPART * fsize);
	CudaAllocate((void**)&eangd, BPART * fsize, "enAngl");
	cudaMemset(eangd, 0, BPART * fsize);

#ifdef UREY_BRADLEY
	eureyb = (float*)malloc(BPART * fsize);
	CudaAllocate((void**)&eureybd, BPART * fsize, "eUreyB");
	cudaMemset(eureybd, 0, BPART * fsize);
#endif

	edihed = (float*)malloc(BPART * fsize);
	CudaAllocate((void**)&edihedd, BPART * fsize, "enDihe");
	cudaMemset(edihedd, 0, BPART * fsize);

#ifdef IMPROPER
	eimprop = (float*)malloc(BPART * fsize);
	CudaAllocate((void**)&eimpropd, BPART * fsize, "enImpr");
	cudaMemset(eimpropd, 0, BPART * fsize);
#endif

	evdw = (float*)malloc(NBPART * fsize);
	CudaAllocate((void**)&evdwd, NBPART * fsize, "enVDWd");
	cudaMemset(evdwd, 0, NBPART * fsize);

	eelec = (float*)malloc(NBPART * fsize);
	CudaAllocate((void**)&eelecd, NBPART * fsize, "enElec");
	cudaMemset(eelecd, 0, BPART * fsize);

	//sum = (float *)malloc(fsize);
	//CudaAllocate((void**)&sumd, sizeof(float)*nAtom, "Sumd12");

	//========= Diagnose Stuff================
#ifdef DIAGNOSE
	CudaAllocate((void**)&diagnose_d, fsize * NUM_DIAG, "Diagns");
	printf("Diagnose Debug Memory Allocated\n");
#endif

#ifdef ANGLE_DEBUG
	CudaAllocate((void**)&angles_debugd, fsize * NUM_ANGLE_DEBUG, "AngDbg");
	printf("Angle Debug Memory Allocated\n");
#endif

#ifdef BOND_DEBUG
	CudaAllocate((void**)&bonds_debugd, fsize * NUM_BOND_DEBUG, "BndDbg");
	printf("Bond Debug Memory Allocated\n");
#endif

#ifdef DIHED_DEBUG
	CudaAllocate((void**)&dihedrals_debugd, fsize * NUM_DIHED_DEBUG, "DihDbg");
	printf("Dihedrals Debug Memory Allocated\n");
#endif

#ifdef DEBUG_NONB
	CudaAllocate((void**)&debug_nonbd, fsize * NUM_DEBUG_NONB, "NnbDbg");
#endif

#ifdef NUM_DEBUG_CELL
	CudaAllocate((void**)&debug_celld, fsize * NUM_DEBUG_CELL, "CelDbg");
#endif

#ifdef DEBUG_PME
	CudaAllocate((void**)&pme_debug_d,
	             Total_PME_Lattices * NUM_DEBUG_PME * sizeof(float),
	             "PMEDbg");
#endif

#ifdef DEBUG_NBLIST
	CudaAllocate((void**)&debug_nblistd,
	             MAX_NBCELL * NumCells_Maximum * NUM_DEBUG_NBLIST * sizeof(float),
	             "NbLDbg");
#endif

#ifdef DEBUG_CELL_BUILD
	CudaAllocate((void**)&debug_celld, fsize * DEBUG_CELL_BUILD, "CelDbg");
#endif

	//========= End Diagnose Stuff============

/*-------------------Dihedral Parameters----------------------*/
	CudaAllocateCopy((void**)&dihedrals_indexd,
	                 dihedrals_index,
	                 isize * (DIHED_COUNT_PERATOM + 1),
	                 "DihIdx");

	CudaAllocateCopy((void**)&dihedral_type_countd,
	                 dihedral_type_count,
	                 sizeof(unsigned char) * MAX_DIHED_TYPE,
	                "DiType");

	CudaAllocateCopy((void**)&dihedrals_setd,
	                 dihedrals_set,
	                 sizeof(int) * (5 * Dihed_Count + 1),
	                 "DihSet");

	CudaAllocateCopy((void**)&dihedral_prmd,
	                 dihedral_prm,
	                 sizeof(dihedralParameter) * MAX_DIHED_TYPE * MAXDPRM,
	                 "DiPrmd");

	cudaMemcpyToSymbol(Dihed_Countd, &Dihed_Count, sizeof(Dihed_Count));

#ifdef IMPROPER
	cudaMemcpyToSymbol(Improper_Countd, &Improper_Count, sizeof(Improper_Count));
	//printf("\n%f", ((float) device_memory_usage)/MEGA_BYTE); exit(1);
	CudaAllocateCopy((void**)&impropers_indexd,
	                 impropers_index,
	                 isize * (DIHED_COUNT_PERATOM + 1),
	                 "ImpIdx");

	CudaAllocateCopy((void**)&impropers_setd,
	                 impropers_set,
	                 sizeof(int) * (5 * Improper_Count + 1),
	                 "Impset");

	CudaAllocateCopy((void**)&improper_prmd,
	                 improper_prm,
	                 sizeof(float2) * MAX_DIHED_TYPE,
	                 "ImpPrm");
#endif
/*-------------------Angle Parameters----------------------*/
	CudaAllocateCopy((void**)&angles_indexd,
	                 angles_index,
	                 isize * (ANGLE_COUNT_PERATOM + 1),
	                 "AngIdx");

	CudaAllocateCopy((void**)&angles_setd,
	                 angles_set,
	                 sizeof(int) * (4 * Angle_Count + 1),
	                 "AngSet");

	/*
	CudaAllocateCopy((void**)&angleprmd,
	                 angleprm,
	                 sizeof(float2)*Num_AngleTypes,
	                 "AngPrm");
	*/
	CudaAllocateCopy((void**)&angleprmd,
	                 angleprm,
	                 sizeof(float2) * Num_Hash_Angles,
	                 "AngPrm");

	cudaMemcpyToSymbol(Angle_Countd, &Angle_Count, sizeof(Angle_Count));

#ifdef UREY_BRADLEY
	CudaAllocateCopy((void**)&ureyb_indexd,
	                 ureyb_index,
	                 isize * (2 * ANGLE_COUNT_PERATOM + 1),
	                 "UryIdx");
	/*
	CudaAllocateCopy((void**)&ureybprmd,
	                 ureybprm,
	                 sizeof(float2) * MAX_ANGLE_TYPE,
	                 "UryPrm");
	*/
	CudaAllocateCopy((void**)&ureybprmd,
	                 ureybprm,
	                 sizeof(float2) * Num_UreyBTypes,
	                 "UryPrm");
#endif

/*-------------------Bond Parameters----------------------*/
	CudaAllocateCopy((void**)&bonds_indexd,
	                 bonds_index,
	                 isize * (2 * BOND_COUNT_PERATOM + 1),
	                 "BndIdx");
	/*
	CudaAllocateCopy((void**)&bonds_setd,
	                 bonds_set,
	                 sizeof(int) * (3 * MAX_BOND_COUNT + 1),
	                 "");

	CudaAllocateCopy((void**)&bondprmd,
	                 bondprm,
	                 sizeof(float2) * Num_BondTypes,
	                 "BndPrm");
	*/
	CudaAllocateCopy((void**)&bondprmd,
	                 bondprm,
	                 sizeof(float2) * Num_Hash_Bonds,
	                 "BndPrm");

/*-------------------Non-Bond Parameters----------------------*/
	CudaAllocate((void**)&nblistd, uisize * MAXNB, "NBList");

	CudaAllocateCopy((void**)&excllistd,
	                 excllist,
	                 isize * (EXCL_COUNT_PERATOM + 1),
	                 "ExlLst");

	CudaAllocateCopy((void**)&excl_bitvecd, excl_bitvec, i2size, "Exlbit");

	CudaAllocateCopy((void**)&excl_bitvec_offsetd,
	                 excl_bitvec_offset,
	                 WorkgroupSize,
	                 "ExlOfs");

#ifdef PME_CALC
	CudaAllocateCopy((void**)&ewlistd,
	                 ewlist,
	                 isize * (EXCL_COUNT_PERATOM + 1),
	                 "ExlLst");
#endif

	CudaAllocateCopy((void**)&prmd, prm, sizeof(float4)*MAXTYP, "AtmPrm");

	CudaAllocateCopy((void**)&nbfixprmd,
	                 nbfixprm,
	                 sizeof(float2) * MAX_NBFIX_TYPE,
	                 "NBFIXAtmPrm");

/*-------------------Molecule Paramters----------------------*/
	CudaAllocateCopy((void**)&typed, type, isize, "AtmTyp");
	CudaAllocateCopy((void**)&molidd, molid, sizeof(float) * NMAX, "MolTyp");
	CudaAllocateCopy((void**)&molmassd,
	                 molmass,
	                 sizeof(float4) * NMAX,
	                 "MolMass");

#ifdef USE_CONSFIX
	/*
	for (int i=0; i<nAtom; i++) {
	  printf("CONSFIX: atom %d ID %d  \n", i, seg_typeid[i]);
	}
	*/
	CudaAllocateCopy((void**)&seg_typeidd,
	                 seg_typeid,
	                 sizeof(int) * NMAX,
	                 "SegTypeID");

	//CudaAllocateCopy((void**)&molidd, molid, isize, "Moleid");
#endif

/*-------------------Minimization----------------------*/
	CudaAllocateCopy((void**)&sdfacd, &sdfac, sizeof(float), "minimz");
	CudaAllocateCopy((void**)&cgfacd, &cgfac, sizeof(float), "minimz");

#if (NBXMOD==5)
	//copy the 1-4 vdw parameters..
	CudaAllocateCopy((void**)&prm1_4d, prm1_4, sizeof(float2) * MAXTYP, "Prm1_4");
#endif

	//copy various constants to constant device memory

#ifdef VSHIFT
	float vdwC1 = -13.0f / pow(Cutoff, 12);
	float vdwC2 = - 7.0f / pow(Cutoff, 6);
	cudaMemcpyToSymbol(vdwC1d, &vdwC1, sizeof(vdwC1));
	cudaMemcpyToSymbol(vdwC2d, &vdwC2, sizeof(vdwC2));
#endif //VSHIFT

#ifdef VSWITCH
	Swcoeff1 = Cutoff * Cutoff;
	Swcoeff2 = Cutoff * Cutoff - 3 * Cuton * Cuton;
	Swcoeff3 = 1.0f / (Cutoff * Cutoff - Cuton * Cuton);
	Swcoeff3 = Swcoeff3 * Swcoeff3 * Swcoeff3;

	cudaMemcpyToSymbol(Swcoeff1d, &Swcoeff1, sizeof(Swcoeff1));
	cudaMemcpyToSymbol(Swcoeff2d, &Swcoeff2, sizeof(Swcoeff2));
	cudaMemcpyToSymbol(Swcoeff3d, &Swcoeff3, sizeof(Swcoeff3));
	cudaMemcpyToSymbol(Cutond, &Cuton, sizeof(Cuton));
#endif //VSWITCH

//////////////////memory allocation for RESTRAINTS///////////////
	if(restraints){
		CudaAllocateCopy((void**)&segidd, segid, nAtom * sizeof(char), "segIDd");

		CudaAllocate((void**)&com0d, nAtom * sizeof(float3), "CenMs1");
		cudaMemset(com0d, 0, nAtom * sizeof(float3));

		CudaAllocate((void**)&com1d, nAtom * sizeof(float3), "CenMs2");
		cudaMemset(com1d, 0, nAtom * sizeof(float3));

		CudaAllocateCopy((void**)&mass_segid0d,
		                 &mass_segid0,
		                 sizeof(float),
		                 "Seg1Ms");

		CudaAllocateCopy((void**)&mass_segid1d,
		                 &mass_segid1,
		                 sizeof(float),
		                 "Seg2Ms");

		cudaMemcpyToSymbol(consharmdistd, &consharmdist, sizeof(consharmdist));
		cudaMemcpyToSymbol(consharmfcd, &consharmfc, sizeof(consharmfc));
	}
//////////////////End memory allocation for RESTRAINTS///////////////

	if(shake){
		CudaAllocateCopy((void**)&constraintsd,
		                 constraints,
		                 nClusters * (CLUSTER_SIZE + 1) * sizeof(int2),
		                 "Cnstra");

		CudaAllocateCopy((void**)&constraints_by_atomd,
		                 constraints_by_atom,
		                 nAtom * (CONSTRAINTS_PER_ATOM + 1) * sizeof(unsigned char),
		                 "ConsAtm");

		CudaAllocateCopy((void**)&constraintsprmd,
		                 constraintsprm,
		                 nClusters * CLUSTER_SIZE * sizeof(float2),
		                 "Consprm");

		CudaAllocateCopy((void**)&atoms_in_clusterd,
		                 atoms_in_cluster,
		                 nClusters * (ATOMS_IN_CLUSTER + 1) * sizeof(int),
		                 "AtinCl");

		cudaMemcpyToSymbol(nClustersd, &nClusters, sizeof(nClusters));

		//shake tolerance...
		cudaMemcpyToSymbol(shaketold, &shaketol, sizeof(shaketol));
	}

#ifdef PCONSTANT
	CudaAllocate((void**)&kineticd, f4size, "KETnsr");
	cudaMemset(kineticd, 0, f4size);

	//virial = (float4 *)malloc(NBPART*f4size);
	CudaAllocate((void**)&viriald, NBPART * f4size, "Virial");
	cudaMemset(viriald, 0, NBPART * f4size);

	CudaAllocate((void**)&propertiesd, f4size, "Properties");
	cudaMemset(propertiesd, 0, f4size);

#ifdef PME_CALC
	float eEwself32 = eEwself;
	cudaMemcpyToSymbol(eEwselfd, &eEwself32, sizeof(eEwself32));
#endif //PME_CALC

	CudaAllocate((void**)&boxAcceld, sizeof(float4), "BoxAcc");
	cudaMemset(boxAcceld, 0, sizeof(float4));

	float4 ReciprocalBoxSquareRatio = {1.0f / (Region.x * Region.x),
	                                   1.0f / (Region.y * Region.y),
	                                   1.0f / (Region.z * Region.z),
	                                   0.0f};
	float tempsum = (ReciprocalBoxSquareRatio.x +
	                 ReciprocalBoxSquareRatio.y +
	                 ReciprocalBoxSquareRatio.z);

	ReciprocalBoxSquareRatio.x = ReciprocalBoxSquareRatio.x / tempsum;
	ReciprocalBoxSquareRatio.y = ReciprocalBoxSquareRatio.y / tempsum;
	ReciprocalBoxSquareRatio.z = ReciprocalBoxSquareRatio.z / tempsum;

	CudaAllocateCopy((void**)&ReciprocalBoxSquareRatiod,
	                 &ReciprocalBoxSquareRatio,
	                 sizeof(float4),
	                 "RcBxRt");

	cudaBindTexture(0, texRBSRat, ReciprocalBoxSquareRatiod, sizeof(float4));

	float4 boxLengthRatio = {1.0f, Region.y / Region.x,
	                         Region.z / Region.x, 0.0f};

	cudaMemcpyToSymbol(boxLengthRatiod, &boxLengthRatio, sizeof(boxLengthRatio));

	CudaAllocateCopy((void**)&boxVelocd, &RegionVeloc, sizeof(float4), "BoxVel");

	CudaAllocateCopy((void**)&boxHd, &RegionH, sizeof(float4), "BoxHlf");
	cudaBindTexture(0, texboxH, boxHd, sizeof(float4));

	CudaAllocateCopy((void**)&boxLengthd, &Region, sizeof(float4), "BoxLen");
	cudaBindTexture(0, texbox, boxLengthd, sizeof(float4));

	CudaAllocateCopy((void**)&prev_boxLengthd, &Region, sizeof(float4), "BoxLen");

	cudaMemcpyToSymbol(pmass_cubicd, &pmass_cubic, sizeof(pmass_cubic));
	cudaMemcpyToSymbol(pRefd, &pRef, sizeof(pRef));
#else //no PCONST
	cudaMemcpyToSymbol(boxH, &RegionH, sizeof(RegionH));
	cudaMemcpyToSymbol(box, &Region, sizeof(Region));
#endif

	cudaMemcpyToSymbol(beta, &KAPPa, sizeof(KAPPa));
	cudaMemcpyToSymbol(betaSqr, &KAPPaSqr, sizeof(KAPPaSqr));
	cudaMemcpyToSymbol(cutoffd, &Cutoff, sizeof(Cutoff));
	cudaMemcpyToSymbol(cutmaxd, &CutMax, sizeof(CutMax));
	cudaMemcpyToSymbol(cutcheckd, &CutCheck, sizeof(CutCheck));
	cudaMemcpyToSymbol(deltaTd, &DeltaT, sizeof(DeltaT));
	//cudaMemcpyToSymbol(initTempd, &CurrentTemp, sizeof(CurrentTemp));
	//cudaMemcpyToSymbol(currentTempd, &CurrentTemp, sizeof(CurrentTemp));
	cudaMemcpyToSymbol(natomd, &nAtom, sizeof(nAtom));
	cudaMemcpyToSymbol(numconstraintsd,
	                   &num_constraints,
	                   sizeof(num_constraints));

	cudaMemcpyToSymbol(num_types_presentd,
	                   &num_types_present,
	                   sizeof(num_types_present));

	cudaMemcpyToSymbol(cutoff7, &rc7, sizeof(rc7));
	cudaMemcpyToSymbol(cutoff13, &rc13, sizeof(rc13));
	cudaMemcpyToSymbol(WorkgroupSized, &WorkgroupSize, sizeof(int));

	// set nonbond list update flag to zero
	nupdate = 0;
	cudaMemcpy(nupdated, &nupdate, sizeof(int), cudaMemcpyHostToDevice);

	// bind texture reference to coordinate array in device memory
	cudaBindTexture(0, texcrd, r4d, f4size);
#ifdef PME_CALC
	cudaBindTexture(0, texsclcrd, r4d_scaled, f4size);
#endif

	cudaBindTexture(0, texprm, prmd, sizeof(float4) * MAXTYP);
	cudaBindTexture(0, texnbfixprm, nbfixprmd, sizeof(float2) * MAX_NBFIX_TYPE);
	cudaBindTexture(0, texmolid, molidd, sizeof(float) * NMAX);
	cudaBindTexture(0, texmolmass, molmassd, sizeof(float2) * NMAX);

#ifdef USE_CONSFIX
	cudaBindTexture(0, texsegtype, seg_typeidd, sizeof(int) * NMAX);
#endif

#if (NBXMOD==5)
	cudaBindTexture(0, texprm1_4, prm1_4d, sizeof(float2) * MAXTYP);
#endif

	cudaBindTexture(0, textype, typed, isize);

	checkCUDAError("DeviceMemoryAllocationFailure");
}
