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

//----------------------------------------------------------------------------
void CallCellBuild(){
//----------------------------------------------------------------------------

#ifdef PROFILING
	cpu1 = clock();
#endif

#ifdef PCONSTANT
	cudaMemset(num_nonbond, 0, MaxNumCells.w * sizeof(int));
#else
	cudaMemset(num_nonbond, 0, NumCells.w * sizeof(int));
#endif

#ifdef REPRO
    printf("Calling CellBuild\n");
#endif

	CellBuild<<<DynadimGrid, DynadimBlock>>>(r4d,
	                                         roldd,
	                                         cell_nonbond,
	                                         num_nonbond,
	                                         nblistd
#ifdef NUM_DEBUG_CELL
	                                       , debug_celld
#endif
	                                                    );

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[CELLBUILD] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	//checkCUDAError("Cell List");
	//captureCellList(0);
	//capture_position(0,0);
}

//----------------------------------------------------------------------------
void Check_Static_Parameters(char success){
//----------------------------------------------------------------------------

	printf("Checking Static Parameters\n");
	int i;
	int xi;
	int yi;
	int zi;

	float frac_numcell = 0.8f;
	float frac_nblist = 0.85f;

#ifdef PCONSTANT
	int4 Number_of_Cells = MaxNumCells;
#else
	int4 Number_of_Cells = NumCells;
#endif

	cudaMemset(num_nonbond, 0, Number_of_Cells.w * sizeof(int));
	CheckCellOccupancy<<<DynadimGrid, DynadimBlock>>>(r4d, num_nonbond);
	cudaThreadSynchronize();

	int* num_nonbond_host;
	num_nonbond_host = (int*)malloc(Number_of_Cells.w * sizeof(int));
	cudaMemcpy(num_nonbond_host, num_nonbond,
	           Number_of_Cells.w * sizeof(int), cudaMemcpyDeviceToHost);

	int max_occupancy = 0;
	int cell_id;
	for(i = 0; i < Number_of_Cells.w; i++){
		if(num_nonbond_host[i] > max_occupancy){
			max_occupancy = num_nonbond_host[i];
			cell_id = i;
		}
	}

	zi = cell_id / (Number_of_Cells.x * Number_of_Cells.y);
	yi = (cell_id % (Number_of_Cells.x * Number_of_Cells.y)) / Number_of_Cells.x;
	xi = (cell_id % (Number_of_Cells.x * Number_of_Cells.y)) % Number_of_Cells.x;

//---------------------------------------
/*
	FILE* file1 = fopen("c:\\userdata\\narayan\\debug_script\\file1.dat", "w");
	cudaMemcpy(num_nonbond_host, num_nonbond,
	           Number_of_Cells.w*sizeof(int), cudaMemcpyDeviceToHost);

	for (i=0; i<Number_of_Cells.w; i++){
		zi = i/(Number_of_Cells.x * Number_of_Cells.y);
		yi = (i%(Number_of_Cells.x * Number_of_Cells.y))/Number_of_Cells.x;
		xi = (i%(Number_of_Cells.x * Number_of_Cells.y))%Number_of_Cells.x;
		fprintf(file1, "%d %d %d\t\t%d\n", xi, yi, zi, num_nonbond_host[i]);
	}
	fclose(file1);
*/
//---------------------------------------


	printf("Cell Info:\tMax cell occupancy =\t%d(cell\t%d, %d, %d),"
	       "\tMax allowed = " EXPAND(CELL_ATOMS)",\tRecommended >= %d,\t"
	       "Stability = ", max_occupancy, xi, yi, zi,
	       (int)((float)max_occupancy / frac_numcell));

	if(max_occupancy >= CELL_ATOMS){
		printf("none\n\nPlease recompile with recommended maximum cell "
		       "capacity...");
		exit(0);
	}

	if(max_occupancy > CELL_ATOMS * frac_numcell){
		printf("Low\n");
	}
	else{
		printf("High\n");
	}


	if(CELL_ATOMS > CELL_BLOCKSIZE){
		printf("Pl. recompile with CELL_BLOCKSIZE > CELL_ATOMS\n");
		exit(0);
	}

	free(num_nonbond_host);

	CallCellBuild();

#ifdef REPRO
    printf("Calling CheckNonBondNum\n");
#endif

	CheckNonbondNum<<<DynadimGrid, DynadimBlock>>>(r4d,
	                                               nblistd,
	                                               cell_nonbond,
	                                               num_nonbond
#ifdef SEARCH_EXCLUSION_LIST
	                                             , excllistd
#endif
	                                                        );
	cudaThreadSynchronize();

	int* nblist_host;
	//copy only the number of nonbond neighbors.. not the neighbors themselves...
	nblist_host = (int*)malloc(isize);

	cudaMemcpy(nblist_host, nblistd, isize, cudaMemcpyDeviceToHost);

	int max_nonbond = 0;
	int atomid;
	for(i = 0; i < nAtom; i++){
		if(nblist_host[i] > max_nonbond){
			max_nonbond = nblist_host[i];
			atomid = i;
		}
	}

	printf("===============\n");
	printf("Nonbond Info:\tMax nonbond neighbors =\t%d(atomid\t%d),\t\tMax "
	       "allowed = " EXPAND(MAXNB) ",\tRecommended >= %d,\tStability = ",
	       max_nonbond, atomid, (int)((float)max_nonbond / frac_nblist));

	if(max_nonbond >= MAXNB){
		printf("none\n\nPlease recompile with recommended maximum nonbond "
		       "neighbors...\n");
		exit(0);
	}

	if(max_nonbond > frac_nblist * MAXNB){
		printf("Low\n");
	}
	else{
		printf("High\n");
	}

	free(nblist_host);
#ifdef PME_CALC
	float frac_lattice_num = 0.75f;

	cudaMemset(numL, 0, fftx * ffty * fftz * sizeof(int));

	CheckLatticeNum<<<DynadimGrid, DynadimBlock>>>(r4d, numL);
	cudaThreadSynchronize();

	int* numL_host;
	numL_host = (int*)malloc(fftx * ffty * fftz * sizeof(int));
	cudaMemcpy(numL_host, numL, fftx * ffty * fftz * sizeof(int),
	           cudaMemcpyDeviceToHost);

	int max_lattice_num = 0;
	int lattice_id;

	for(i = 0; i < fftx * ffty * fftz; i++){
		if(max_lattice_num < numL_host[i]){
			max_lattice_num = numL_host[i];
			lattice_id = i;
		}
	}

	zi = lattice_id / (fftx * ffty);
	yi = (lattice_id % (fftx * ffty)) / (fftx);
	xi = (lattice_id % (fftx * ffty)) % (fftx);

	printf("===============\n");
	printf("Lattice Info:\tMax lattice neighbors =\t%d(lattice\t%d, %d, %d),\tMax"
	       " allowed = " EXPAND(MAXLATTICE_NEIGHBORS) ",\tRecommended >= %d,"
	       "\tStability = ", max_lattice_num, xi, yi, zi,
	      (int)((float)max_lattice_num / frac_lattice_num));

	if(max_lattice_num >= MAXLATTICE_NEIGHBORS){
		printf("none\n\nPlease increase FFT dimensions or recompile with "
		       "recommended maximum lattice neighbors...\n");
		exit(0);
	}

	if(max_lattice_num >  frac_lattice_num * MAXLATTICE_NEIGHBORS){
		printf("Low\n");
	}
	else{
		printf("High\n");
	}

	free(numL_host);
#endif //PME_CALC

	if(!success){
		exit(-1);
	}

	return;
}

//----------------------------------------------------------------------------
void BuildNBGPU(){
//----------------------------------------------------------------------------
//build nonbond list on GPU
//----------------------------------------------------------------------------
	// set execution configuration, p = block size

	//CallCellBuild();

#ifdef PCONSTANT
	cudaMemcpy(&NumCells, numcellsd, sizeof(int4), cudaMemcpyDeviceToHost);

	//recompute the number of blocks sizes for NBBuild and CellClean,
	//incase the number of cells has changed...
	NBBuild_dimBlock.x = CELL_ATOMS;
	NBBuild_dimBlock.y = 1;
	NBBuild_dimBlock.z = 1;

	//divide the components among x and y dimensions to stay well
	//within the maximum limit...
	NBBuild_dimGrid.x = NumCells.x * NumCells.y;
	NBBuild_dimGrid.y = NumCells.z;
	NBBuild_dimGrid.z = 1;


	//CellClean_dimBlock.x = MaxNumCells.x; //NumCells.x; //
	//CellClean_dimBlock.y = 1;
	//CellClean_dimBlock.z = 1;

	//CellClean_dimGrid.x = MaxNumCells.y; //NumCells.y; //
	//CellClean_dimGrid.y = MaxNumCells.z; //NumCells.z; //
	//CellClean_dimGrid.z = 1;

#endif

#ifdef PROFILING
	cpu1 = clock();
#endif

	CellUpdate<<<DynadimGrid, DynadimBlock>>>(r4d,
	                                          roldd,
	                                          cell_nonbond,
	                                          num_nonbond
#ifdef PCONSTANT
	                                        , prev_boxLengthd
#endif
	                                                         );
	//captureCellList(0, 0, 0);

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[CELLUPDATE] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("CellUpdate");
	//TakeSnapshot(1);
	//captureCellList(0, 0, 1);


#ifdef PROFILING
	cpu1 = clock();
#endif

	CellClean<<<CellClean_dimGrid, CellClean_dimBlock>>>(r4d,
	                                                     cell_nonbond,
	                                                     num_nonbond);

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[CELLCLEAN] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif

	checkCUDAError("CellClean");
	//captureCellList(0, 1, 2);
	//capture_position(0,0);

#ifdef PROFILING
	cpu1 = clock();
#endif

#ifdef REPRO
    printf("Calling NBBuild_Kernel\n");
#endif

	NBBuild_Kernel<<<NBBuild_dimGrid, NBBuild_dimBlock>>>(r4d,
	                                                      roldd,
	                                                      nblistd,
	                                                      cell_nonbond,
	                                                      num_nonbond,
	                                                      excl_bitvecd,
	                                                      excl_bitvec_offsetd,
	                                                      excllistd
#ifdef PCONSTANT
	                                                    , prev_boxLengthd
#endif
#ifdef DEBUG_NBLIST
	                                                    , debug_nblistd
#endif
		                                                                 );



#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[NBBUILD] += (clock() - cpu1) / CLOCKS_PER_SEC;
	nblist_call_count++;
#endif
	checkCUDAError("NB list build");
	//TakeSnapshot(1);
	//capturenblist();
	//capture_position(0,0);

}


//////////////////////////////////////////
#ifdef PME_CALC
//////////////////////////////////////////
//==============================================================================
void CallLatticeBuild(){
//==============================================================================

	//Clear Lattice neighbor list...
	cudaMemset(numL, 0, fftx * ffty * fftz * sizeof(int));

#ifdef PROFILING
	cpu1 = clock();
#endif

	LatticeBuild<<<DynadimGrid, DynadimBlock>>>(r4d, r4d_scaled, cellL, numL
#ifdef DEBUG_PME
	                                          , pme_debug_d
#endif
	                                                       );

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[LATTICEBUILD]  += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("LatticeBuild");

//==============================================================================
	//captureCellList(0, 0, -1);
	//capture_position(0,0);
	//TakeSnapshot(1);
}
//////////////////////////////////////////
#endif
//////////////////////////////////////////

#ifdef PCONSTANT
//----------------------------------------------------------------------------
void ScanVirial(){
//----------------------------------------------------------------------------
//===================Compute Virial=============================================
#ifdef PROFILING
	cpu1 = clock();
#endif

#ifdef USE_NPT
	float scalarvirialT = 0.0f;
	float pressure = 0.0f;
	float kinEnergy = 0.0f;
	float temperature = 0.0f;
	float4 VirialTensor = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 kineticE = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 boxLength = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 properties = {0.0f, 0.0f, 0.0f, 0.0f};

#endif

	// #define DEBUG_NPT
	//***Important!!! The blocksize must be a power of 2 for this reduction
	//kernel to work.. otherwise the kernel code
	// must be modified to handle non powers of 2

	//---- Single Pass reduction using threadfence global synchronization... -----

	reduce_virial<<<nAtom / MAX_BLOCK_SIZE + 1, MAX_BLOCK_SIZE>>>(v4d,
	                                                              boxVelocd,
	                                                              boxAcceld,
	                                                              kineticd,
	                                                              viriald
#ifdef USE_NPT
	                                                            , boxLengthd,
	                                                              propertiesd
#endif
	                                                                         );
#ifdef USE_NPT

#ifdef DEBUG_NPT
	// on the host
	cudaMemcpy(&VirialTensor, viriald, sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(&boxLength, boxLengthd, sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(&kineticE, kineticd, sizeof(float4), cudaMemcpyDeviceToHost);
#endif
	// on the device
	cudaMemcpy(&properties, propertiesd, sizeof(float4), cudaMemcpyDeviceToHost);

#ifdef DEBUG_NPT
	// on the host
	scalarvirialT = VirialTensor.x + VirialTensor.y + VirialTensor.z;
	pressure = (68558.64100846418f / (3.0f * boxLength.x *
	                                  boxLength.y * boxLength.z)) *
	           (kineticE.x + kineticE.y + kineticE.z + scalarvirialT);

	kinEnergy = (kineticE.x + kineticE.y + kineticE.z) * 0.5f;

#ifdef CONSFIX
		temperature = 2 * kinEnergy /  RC / ( (3 * nAtomwoseg) - 6 - num_constraints);
#else
    // narayan version
		// temperature = 2 * kinEnergy / RC / (3.0f * nAtom - 3.0f - num_constraints);
		//patel version
		temperature = 2 * kinEnergy / RC / (3.0f * nAtom - 6.0f - num_constraints);
#endif

	printf("Host: scalarvirialT %f pressure host %f kinEnergy %f "
	       "temperature %f \n", scalarvirialT, pressure, kinEnergy, temperature);
#endif
	// printf("Device: temperature is incomplete \n");
	printf("Device: scalarvirialT %f pressure host %f kinEnergy %f "
	       "temperature %f \n", properties.x, properties.y,
	       properties.z, properties.w);
#endif

	//////-------- or enable the following two step reduction... ----------

	//reduce_virial_partial_sum<<<nAtom/MAX_BLOCK_SIZE + 1, MAX_BLOCK_SIZE>>>(v4d,
	//boxVelocd, kineticd, viriald);
	//reduce_virial_final_sum<<<1, MAX_BLOCK_SIZE>>>(kineticd, viriald);

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[REDUCE] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("Virial");
}
#endif

//----------------------------------------------------------------------------
void Scan_Energy(){
//----------------------------------------------------------------------------
	//***Important!!! The blocksize must be a power of 2 and one dimensional
	//grid for this reduction kernel to work..
	// otherwise the kernel code must be modified to handle non powers of 2 and
	//multiple dimensional grids for this to work...

	//////single pass reduction
	reduce_PE<<<nAtom / MAX_BLOCK_SIZE + 1, MAX_BLOCK_SIZE>>>(f4d,
	                                                          f4d_nonbond,
	                                                          f4d_bonded);

	//////-----or Multipass reduction-------

	//reduce_PE_partial_sum<<<nAtom/MAX_BLOCK_SIZE + 1, MAX_BLOCK_SIZE>>>(f4d);
	//reduce_PE_final_sum<<<1, MAX_BLOCK_SIZE>>>(f4d);

	checkCUDAError("Scan Energy");
}

//----------------------------------------------------------------------------
void MinimizeStep(){
//----------------------------------------------------------------------------

	static float Prev_Egy = 0;
	static float Curr_Egy = 0;
	float4 temp;

//---------------------------Save previous Scaled Positions---------------------
#ifdef PME_CALC
	//save the old positions after updating coordinates...
	cudaMemcpy(prev_r4d_scaled, r4d_scaled, f4size, cudaMemcpyDeviceToDevice);
#endif

	cudaMemset(nupdated, 0, sizeof(int));

//---------------------------Conjugated Gradient Descent------------------------
	//save last known minimized structure...
	cudaMemcpy(min_r4d, r4d, f4size, cudaMemcpyDeviceToDevice);

	if(minimization_method == 1){
		ConjugatedGradient<<<DynadimGrid, DynadimBlock>>>(r4d,
		                                                  roldd,
		                                                  nupdated,
		                                                  v4d,
		                                                  f4d,
		                                                  f4d_nonbond,
		                                                  f4d_bonded,
		                                                  prev_f4d,
		                                                  Htd,
		                                                  cgfacd
#ifdef PME_CALC
		                                                , r4d_scaled,
		                                                  prev_r4d_scaled,
		                                                  disp
#endif
		                                                      );
	}
	else if(minimization_method == 2){
		SteepestDescent<<<DynadimGrid, DynadimBlock>>>(r4d,
		                                               roldd,
		                                               nupdated,
		                                               v4d,
		                                               f4d,
		                                               f4d_nonbond,
		                                               f4d_bonded,
		                                               sdfacd
#ifdef PME_CALC
		                                             , r4d_scaled,
		                                               prev_r4d_scaled,
		                                               disp
#endif
		                                                   );
	}

//---------------------------Update Nonbond List--------------------------------
	cudaMemcpy(&nupdate, nupdated, sizeof(int), cudaMemcpyDeviceToHost);
	if(INBFRQ < 0) {
		 if(nupdate == 1){
			 BuildNBGPU();
		 }
	}
	else{
		if((stepCount % INBFRQ) == 0){
			BuildNBGPU();
		}
	}

#ifdef PME_CALC
//---------------------------Update Lattice-------------------------------------
	if((stepCount > 0) && (!(stepCount % 1000))){
		CallLatticeBuild();
	}
	else{
		LatticeUpdate<<<LatUpdimGrid, LatUpdimBlock>>>(r4d_scaled,
		                                               disp,
		                                               cellL,
		                                               numL
#ifdef DEBUG_PME
		                                             , pme_debug_d
#endif
		                                                          );

	cudaThreadSynchronize();
	checkCUDAError("Lattice Update");
	}// end if ((stepCount>0)&&(!(stepCount%1000)))

#endif


//------------------Compute Forces------------------------------------
	ComputeAccelGPU();
//------------------GPUReduce Potential Energy---------------------------

	Scan_Energy();

	Prev_Egy = Curr_Egy;
	cudaMemcpy(&temp, f4d, sizeof(float4), cudaMemcpyDeviceToHost);
	Curr_Egy = temp.w;

	if(!(stepCount % 10)){
		printf("Step = %d, Energy = %f\n", stepCount, Curr_Egy + eEwself);
	}

	//conjugatedgradient = 1;

	//restore previous positions and initialize accelerations...
	if((Curr_Egy > Prev_Egy) && (Prev_Egy != 0.0f)){
		Curr_Egy = Prev_Egy;
		cudaMemcpy(r4d, min_r4d, f4size, cudaMemcpyDeviceToDevice);
		cudaMemcpy(f4d, prev_f4d, f4size, cudaMemcpyDeviceToDevice);
		cudaMemset(f4d_nonbond, 0, NBPART * f4size);
		cudaMemset(f4d_bonded, 0, BPART * f4size);

		stepCount= stepCount - 1;

		switch(minimization_method){
			case 1:
				minimization_method = 2;
				printf("----------------------------\nSwitching to steepest descent "
				       "at step = %d\n----------------------------\n", stepCount);
				break;//switch to steepest descent...
			case 2:
				minimization_method = 0;
				printf("----------------------------\nExiting minimization at step"
				       " = %d\n----------------------------\n", stepCount);
				break;//quit minimization...
		}
	}

}

//------------------------------------------------------------------------------
void SingleStep(){
//------------------------------------------------------------------------------
//r & rv are propagated by DeltaT in time using the velocity-Verlet method.
//------------------------------------------------------------------------------
//---------------------------Halfkick the velocities----------------------------

#ifdef PROFILING
	cpu1 = clock();
#endif

	HalfKickGPU<<<DynadimGrid, DynadimBlock>>>(v4d, f4d, f4d_nonbond, f4d_bonded
#ifdef PCONSTANT
	                                         , boxVelocd, boxAcceld
#endif
		                                                             );

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[HALFKICK] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("Halfkick1");

//--------------------------Update Coordinate and Boxsizes----------------------

#ifdef PROFILING
	cpu1 = clock();
#endif

	//save the old positions after updating coordinates...
	if(shake){
		cudaMemcpy(r4shaked, r4d, f4size, cudaMemcpyDeviceToDevice);
	}

	UpdateCoords<<<DynadimGrid, DynadimBlock>>>(r4d, v4d
#ifdef PCONSTANT
	                                          , viriald
#endif
	                                                   );

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

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[CONSTRAINTS] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("Shake");

	//Write_xyz("xyz_after.dat", 0);
	//exit(0);

//--------------------------Save Previous Scaled Positions----------------------
#ifdef PROFILING
	cpu1 = clock();
#endif

#ifdef PME_CALC
	//save the old positions after updating coordinates...
	cudaMemcpy(prev_r4d_scaled, r4d_scaled, f4size, cudaMemcpyDeviceToDevice);
#endif
	cudaMemset(nupdated, 0, sizeof(int));

	CoordsUpdate<<<DynadimGrid, DynadimBlock>>>(r4d,
	                                            roldd,
	                                            nupdated
#ifdef PME_CALC
	                                          , r4d_scaled,
	                                            prev_r4d_scaled,
	                                            disp
#endif
#ifdef PCONSTANT
	                                          , boxLengthd,
	                                            boxHd,
	                                            boxVelocd,
	                                            boxAcceld,
	                                            numcellsd,
	                                            ReciprocalBoxSquareRatiod
#ifdef PME_CALC
	                                          , fracd
#endif
#endif
	                                                 );

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[COORDSUPDATE] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("Update coords, NB Update Check");

//-------------------------------------Update Nonbond List----------------------


	cudaMemcpy(&nupdate, nupdated, sizeof(int), cudaMemcpyDeviceToHost);
	if (INBFRQ < 0) {
		 if(nupdate == 1){
			 BuildNBGPU();
		 }
	} 
	else{
		if ((stepCount % INBFRQ) == 0){
			BuildNBGPU();
		}
	}

//-------------------------------------Update Lattice---------------------------
#ifdef PME_CALC
#ifndef CHARGESPREAD_ATOMIC
	if((stepCount > 0) && (!(stepCount % 1000))){
		CallLatticeBuild();
	}
	else{

#ifdef PROFILING
	cpu1 = clock();
#endif

	LatticeUpdate<<<LatUpdimGrid, LatUpdimBlock>>>(r4d_scaled, disp, cellL, numL
#ifdef DEBUG_PME
	                                             , pme_debug_d
#endif
	                                                          );

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[LATTICEUPDATE] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("Lattice Update");

	//TakeSnapshot(1);
	}// end if ((stepCount>0)&&(!(stepCount%1000)))

#endif
#endif//#ifdef PME_CALC

//-------------------------------------Compute Forces---------------------------

	ComputeAccelGPU();

//-------------------------------------Reduce Virial----------------------------
#ifdef PCONSTANT
	ScanVirial();
#endif

//----------------------------------------HalfKick------------------------------
#ifdef PROFILING
	cpu1 = clock();
#endif
	HalfKickGPU<<<DynadimGrid, DynadimBlock>>>(v4d, f4d, f4d_nonbond, f4d_bonded
#ifdef PCONSTANT
	                                         , boxVelocd, boxAcceld
#endif
	                                                               );

#ifdef PROFILING
	cudaThreadSynchronize();
	profile_times[HALFKICK] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
	checkCUDAError("Halfkick2");

//-------------------------------------Do RATTLE--------------------------------

	if(shake){
#ifdef PROFILING
		cpu1 = clock();
#endif
		solve_velocity_constraints<<<num_cluster_blocks,
		                             cluster_blocksize>>>(constraintsd,
		                                                  constraints_by_atomd,
		                                                  constraintsprmd,
		                                                  atoms_in_clusterd,
		                                                  v4d
#ifdef PCONSTANT
		                                                , viriald
#endif
		                                                         );
#ifdef PROFILING
		cudaThreadSynchronize();
		profile_times[CONSTRAINTS] += (clock() - cpu1) / CLOCKS_PER_SEC;
#endif
		checkCUDAError("Rattle");
	}

	//Write_xyz("xyz_after.dat", 0);
	//exit(0);

} //void SingleStep()

//------------------------------------------------------------------------------
void write_trj_header(FILE* xyzfile, int stepCount){
//------------------------------------------------------------------------------
	char header[80] = "CORD";
	//int tempval=0, ntitle = 1;
	//double timestep;
	//timestep = DeltaT;

	int nframes;

	//write timestep, size, number of timesteps etc..
	fwrite(header, 1, 4, xyzfile);

	if(StepStore > StepLimit - stepCount){
		nframes = (StepLimit - stepCount)/StepAvg;
	}
	else{
		nframes = StepStore / StepAvg;
	}

	printf("Num Frames = %d\n", nframes);

	fwrite(&nframes, sizeof(nframes), 1, xyzfile);
	fwrite(&nAtom, sizeof(nAtom), 1, xyzfile);
	//fwrite(&StartCount, sizeof(StartCount), 1, xyzfile);
	//fwrite(&StepAvg, sizeof(StepAvg), 1, xyzfile);

	//fwrite(&tempval, sizeof(int), 5, xyzfile);
	//fwrite(&tempval, sizeof(int), 1, xyzfile);
	//fwrite(&timestep, sizeof(timestep), 1, xyzfile);
	//fwrite(&tempval, sizeof(int), 9, xyzfile);

	//write title information...
	//sprintf(header, "DMPC, Lipid Membrane System in Water....DMPC, "
	//"Lipid Membrane System in Water...");
	//fwrite(&ntitle, sizeof(int), 1, xyzfile);
	//fwrite(&header, 1, 80, xyzfile);

	//fwrite(&nAtom, sizeof(int), 1, xyzfile);

	return;
}

/*----------------------------------------------------------------------------*/
void checkCUDAError(const char* msg){
/*------------------------------------------------------------------------------
	Get last CUDA error
------------------------------------------------------------------------------*/
	cudaError_t err = cudaGetLastError();

	if(cudaSuccess != err){
		//fprintf(stderr, "CUDA Error: %s\n");
		//fprintf(outfile, "CUDA Error: %s\n");

		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		fprintf(outfile,"Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));

		Check_Static_Parameters(-1);//exit(-1);
	}
}


/*----------------------------------------------------------------------------*/
void validate_dihedral(){
/*----------------------------------------------------------------------------*/
	//int ndihed;
	float ener = 0;

	float3 a;
	float3 b;//, ab;

	float invrasqr;
	float invrbsqr;
	float invrab;// gab;//phi,
	float cosphi;
	float cosphi2;
	float cosphi3;
	float cosphi4;
	float cosphi6;
	float cosNphi;

	float3 v1;
	float3 v2;
	float3 v3;

	int idx1;
	int idx2;
	int idx3;
	int idx4;

	float4 x1;
	float4 x2;
	float4 x3;
	float4 x4;

	int j;
	int k;
	int h;
	int nd;
	int m;
	int i;

	printf("Natoms = %d\n", nAtom);

///process list///////

	for(i = 0; i < nAtom; i++){
		for(m = 0; m < dihedrals_index[i]; m++){
			j = dihedrals_index[WorkgroupSize * (m + 1) + i];
			//ndihed = dihedlist[i];

			//printf("Atomid = %d, ndihed = %d\n", i, ndihed);

			//printf("Dihed %d, %d %d %d %d\n", j, idx1, idx2, idx3, idx4);

	///end process list////////

/*
	//process set/////////

	printf("Dihedreal Count %d\n", dihedrals_set[0]);
		for (j=0; j< dihedrals_set[0]; j++){
*/
			//hash computed later below...
			h = dihedrals_set[0 * Dihed_Count + (j + 1)];
			idx1 = dihedrals_set[1 * Dihed_Count + (j + 1)];
			idx2 = dihedrals_set[2 * Dihed_Count + (j + 1)];
			idx3 = dihedrals_set[3 * Dihed_Count + (j + 1)];
			idx4 = dihedrals_set[4 * Dihed_Count + (j + 1)];

			//printf("j=%d id1=%d id2=%d id3=%d id4=%d\n", j, idx1, idx2, idx3, idx4);
			x1 = r[idx1];
			x2 = r[idx2];
			x3 = r[idx3];
			x4 = r[idx4];

			v1.x = x1.x - x2.x;
			v1.y = x1.y - x2.y;
			v1.z = x1.z - x2.z;

			v2.x = x2.x - x3.x;
			v2.y = x2.y - x3.y;
			v2.z = x2.z - x3.z;

			v3.x = x4.x - x3.x;
			v3.y = x4.y - x3.y;
			v3.z = x4.z - x3.z;


			// ** nearest images!!! **
			v1.x = v1.x - copysign(RegionH.x, v1.x - RegionH.x) -
			       copysign(RegionH.x, v1.x + RegionH.x);

			v1.y = v1.y - copysign(RegionH.y, v1.y - RegionH.y) -
			       copysign(RegionH.y, v1.y + RegionH.y);

			v1.z = v1.z - copysign(RegionH.z, v1.z - RegionH.z) -
			       copysign(RegionH.z, v1.z + RegionH.z);

			v2.x = v2.x - copysign(RegionH.x, v2.x - RegionH.x) -
			       copysign(RegionH.x, v2.x + RegionH.x);

			v2.y = v2.y - copysign(RegionH.y, v2.y - RegionH.y) -
			       copysign(RegionH.y, v2.y + RegionH.y);

			v2.z = v2.z - copysign(RegionH.z, v2.z - RegionH.z) -
			       copysign(RegionH.z, v2.z + RegionH.z);

			v3.x = v3.x - copysign(RegionH.x, v3.x - RegionH.x) -
			       copysign(RegionH.x, v3.x + RegionH.x);

			v3.y = v3.y - copysign(RegionH.y, v3.y - RegionH.y) -
			       copysign(RegionH.y, v3.y + RegionH.y);

			v3.z = v3.z - copysign(RegionH.z, v3.z - RegionH.z) -
			       copysign(RegionH.z, v3.z + RegionH.z);

			dihedralParameter prm;

			//h = HashDihed_Manual(type[idx1], type[idx2], type[idx3], type[idx4]);
			//h = HashDihed_Pearson(type[idx1], type[idx2], type[idx3], type[idx4]);
			//h = HashDihed(type[idx1], type[idx2], type[idx3], type[idx4]);

			//nd = ndihedprm[h];
			nd = dihedral_type_count[h];

			for(k = 0; k < nd; k++){

				//a = v1 x v2
				a.x = v1.y * v2.z - v1.z * v2.y;
				a.y = v1.z * v2.x - v1.x * v2.z;
				a.z = v1.x * v2.y - v1.y * v2.x;

				//b = v3 x v2
				b.x = v3.y * v2.z - v3.z * v2.y;
				b.y = v3.z * v2.x - v3.x * v2.z;
				b.z = v3.x * v2.y - v3.y * v2.x;

				invrasqr = 1.0f / (a.x * a.x + a.y * a.y + a.z * a.z);
				invrbsqr = 1.0f / (b.x * b.x + b.y * b.y + b.z * b.z);
				invrab = sqrt(invrasqr * invrbsqr);

				cosphi = invrab * (a.x * b.x + a.y * b.y + a.z * b.z);

				//cosphi = __mycopysignf(__saturatef(abs(cosphi)), cosphi);

				//phi = acos(cosphi);

				//ab = a x b
				//ab.x = a.y*b.z - a.z*b.y;
				//ab.y = a.z*b.x - a.x*b.z;
				//ab.z = a.x*b.y - a.y*b.x;

				//gab = v2 . ab
				//gab = v2.x*ab.x + v2.y*ab.y + v2.z*ab.z;

				//if (gab > 0) phi = -phi;

				prm = dihedral_prm[k * MAX_DIHED_TYPE + h];

				//prmx = dihedprm[h + k*MAXTYP].x;
				//prmy = dihedprm[h + k*MAXTYP].y;
				//prmz = dihedprm[h + k*MAXTYP].z;

				cosphi2 = cosphi * cosphi;
				cosphi3 = cosphi2 * cosphi;
				cosphi4 = cosphi2 * cosphi2;
				//cosphi5 = cosphi2*cosphi3;
				cosphi6 = cosphi3 * cosphi3;

				cosNphi = 1;
				if(prm.n == 0){
					cosNphi = 1;
				}
				if(prm.n == 1){
					cosNphi = cosphi;
				}
				if(prm.n == 2){
					cosNphi = 2 * cosphi2 - 1;
				}
				if(prm.n == 3){
					cosNphi = 4 * cosphi3 - 3 * cosphi;
				}
				if(prm.n == 4){
					cosNphi = 8 * cosphi4 - 8 * cosphi2 + 1;
				}
				if(prm.n == 6){
					cosNphi = 32 * cosphi6 - 48 * cosphi4 + 18 * cosphi2 - 1;
				}

				if(abs(prm.d) == 180){
					cosNphi = -cosNphi;
				}

				//ener += prmx*(1.0f + cos(prmy*phi  -prmz));
				ener += prm.x * (1.0f + cosNphi);

				//printf("Energy Contribution of %d %d %d %d with torsion %f is %f\n",
				//idx1, idx2, idx3, idx4, phi, prmx*(1.0f + cos(prmy*phi  -prmz)));
			}
		}
	}
	printf("CPU Dihedral Energy=%f\n", 0.25*ener);
	//exit(-1);
	return;
}

//----------------------------------------------------------------------------
void validate_angle(){
//----------------------------------------------------------------------------

	float4 x1;
	float4 x2;
	float4 x3;

	float3 r21;
	float3 r23;

	float r21sqr;
	float r23sqr;
	float r21r23;

	float costheta;
	float theta;//, sintheta

	float dtheta;
	float ener = 0;//kfactor,

	int i;
	int j;
	int ai;
	int h;
	int idx1;
	int idx2;
	int idx3;

	for(j = 0; j < nAtom; j++){
		for(i = 0; i < angles_index[j]; i++){

			ai = angles_index[WorkgroupSize * (i + 1) + j];

			h = angles_set[Angle_Count * 0 + (ai + 1)];
			idx1 = angles_set[Angle_Count * 1 + (ai + 1)];
			idx2 = angles_set[Angle_Count * 2 + (ai + 1)];
			idx3 = angles_set[Angle_Count * 3 + (ai + 1)];

			x1 = r[idx1];
			x2 = r[idx2];
			x3 = r[idx3];

			r21.x = x1.x - x2.x;
			r21.y = x1.y - x2.y;
			r21.z = x1.z - x2.z;

			r23.x = x3.x - x2.x;
			r23.y = x3.y - x2.y;
			r23.z = x3.z - x2.z;

			// choose nearest images
			r21.x = r21.x, - copysign(RegionH.x, r21.x - RegionH.x) -
			        copysign(RegionH.x, r21.x + RegionH.x);

			r21.y = r21.y, - copysign(RegionH.y, r21.y - RegionH.y) -
			        copysign(RegionH.y, r21.y + RegionH.y);

			r21.z = r21.z, - copysign(RegionH.z, r21.z - RegionH.z) -
			        copysign(RegionH.z, r21.z + RegionH.z);

			r23.x = r23.x, - copysign(RegionH.x, r23.x - RegionH.x) -
			        copysign(RegionH.x, r23.x + RegionH.x);

			r23.y = r23.y, - copysign(RegionH.y, r23.y - RegionH.y) -
			        copysign(RegionH.y, r23.y + RegionH.y);

			r23.z = r23.z, - copysign(RegionH.z, r23.z - RegionH.z) -
			        copysign(RegionH.z, r23.z + RegionH.z);

			// calculate r21 and r23
			r21sqr = r21.x * r21.x + r21.y * r21.y + r21.z * r21.z;
			r23sqr = r23.x * r23.x + r23.y * r23.y + r23.z * r23.z;
			r21r23 = r21.x * r23.x + r21.y * r23.y + r21.z * r23.z;

			//r21m = sqrt(r21sqr);
			//r23m = sqrt(r23sqr);

			// calculate cos theta, sin theta, and theta
			costheta = r21r23 / sqrt(r21sqr * r23sqr);
			//sintheta = sqrt(1.0f - costheta*costheta);
			theta = acos(costheta);

			// calculate force contributions
			//float dtheta = theta - AHOH;
			dtheta = theta - angleprm[h].y;
			//float kfactor = -KA*dtheta/sintheta;
			//kfactor = -angleprm[h].x*dtheta/sintheta;

			ener += angleprm[h].x * dtheta * dtheta / 3;

		}//i loop

	}//j loop

	printf("Angle Energy: %f\n", ener);
	return;
}
