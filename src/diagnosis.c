/* -*- mode: C++; tab-width: 2; indent-tabs-mode: t; c-basic-offset: 2 -*- */
// vim:sts=2:sw=2:ts=2:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
#define DEBUG_PREFIX C:\\UserData\\Narayan\\debug_script\\


/*----------------------------------------------------------------------------*/
void  TakeSnapshot(int snapshot){
/*------------------------------------------------------------------------------
  Implements the djb2 hashing function
------------------------------------------------------------------------------*/

	//int total_cells; //i,n, 

#ifdef DIAGNOSE
		float *diagnose;
		diagnose = (float *)malloc(fsize*NUM_DIAG);
		FILE *diagnose_file_xyz, *diagnose_file_pairs;
		char file3[1000] = "/usa/dchapp/fenzi/FENZI/trunk/test/dmpc_small/Diagnose_File_Pairs";
		char file4[1000] = "/usa/dchapp/fenzi/FENZI/trunk/test/dmpc_small/Diagnose_File_XYZ";
		diagnose_file_pairs = fopen(file3, "w");
		diagnose_file_xyz = fopen(file4, "w");
		fprintf(diagnose_file_xyz,"%d\n Membrane\n", nAtom);

		cudaMemcpy(diagnose, diagnose_d, fsize*NUM_DIAG, cudaMemcpyDeviceToHost);
		for (n=0; n<nAtom; n++){
			//write the energy and the pairs...
			if (diagnose[NUM_DIAG*n]>1)//if the VDW energy is recorded 
			fprintf(diagnose_file_pairs, "%f\t\t\t\t%d(%s)\t\t%d(%s)\n", diagnose[NUM_DIAG*n], (int) diagnose[NUM_DIAG*n + 1], atom_type[(int) diagnose[NUM_DIAG*n + 1]]
																		,(int) diagnose[NUM_DIAG*n + 2], atom_type[(int) diagnose[NUM_DIAG*n + 2]]);
			//write xyz coordinates...
			fprintf(diagnose_file_xyz, "%s\t%f\t%f\t%f\n", atom_type[n], diagnose[NUM_DIAG*n + 3], diagnose[NUM_DIAG*n + 4], diagnose[NUM_DIAG*n + 5]); 
		}

		fclose(diagnose_file_pairs);
		fclose(diagnose_file_xyz);
		free(diagnose);
#endif

#ifdef ANGLE_DEBUG
			FILE *angle_file;
			float *angles_debug;
			angles_debug = (float*) malloc(fsize*NUM_ANGLE_DEBUG);
			angle_file = fopen("angle_debug.dat", "w");
			cudaMemcpy(angles_debug, angles_debugd, fsize*NUM_ANGLE_DEBUG, cudaMemcpyDeviceToHost);
			
			for (int i=0;i<nAtom; i++){
				//fprintf(angle_file, "%d\t\t%s\t\t%f\t\t%f\t\t%f\n", i+1, atom_type[i], angles_debug[i*NUM_ANGLE_DEBUG + 0], angles_debug[i*NUM_ANGLE_DEBUG + 1], angles_debug[i*NUM_ANGLE_DEBUG + 2]);
				fprintf(angle_file, "%f\t\t%f\t\t%f\n", angles_debug[i*NUM_ANGLE_DEBUG + 0], angles_debug[i*NUM_ANGLE_DEBUG + 1], angles_debug[i*NUM_ANGLE_DEBUG + 2]);
			}			
			fclose(angle_file);
			free(angles_debug);
#endif


#ifdef BOND_DEBUG
			FILE *bond_file;
			float *bonds_debug;
			bonds_debug = (float*) malloc(fsize*NUM_BOND_DEBUG);
			bond_file = fopen("bond_debug.dat", "w");
			cudaMemcpy(bonds_debug, bonds_debugd, fsize*NUM_BOND_DEBUG, cudaMemcpyDeviceToHost);
			
			for (int i=0;i<nAtom; i++){
				//fprintf(bond_file, "%d\t\t%s\t\t%f\t\t%f\t\t%f\n", i+1, atom_type[i], bonds_debug[i*NUM_BOND_DEBUG + 0], bonds_debug[i*NUM_BOND_DEBUG + 1], bonds_debug[i*NUM_BOND_DEBUG + 2]);
				fprintf(bond_file, "%f\t\t%f\t\t%f\n", bonds_debug[i*NUM_BOND_DEBUG + 0], bonds_debug[i*NUM_BOND_DEBUG + 1], bonds_debug[i*NUM_BOND_DEBUG + 2]);
			}
			fclose(bond_file);
			free(bonds_debug);
#endif

#ifdef DIHED_DEBUG
			FILE *dihed_file;
			float *dihedrals_debug;
			dihedrals_debug = (float*) malloc(fsize*NUM_DIHED_DEBUG);
			dihed_file = fopen("dihed_debug.dat", "w");
			cudaMemcpy(dihedrals_debug, dihedrals_debugd, fsize*NUM_DIHED_DEBUG, cudaMemcpyDeviceToHost);
			
			for (int i=0;i<nAtom; i++){
				//if (dihedrals_debug[i*NUM_DIHED_DEBUG + 1] != 0)
				//fprintf(dihed_file, "%d\t\t%s\t\t%f\t\t%f\t\t%f\n", i+1, atom_type[i], dihedrals_debug[i*NUM_DIHED_DEBUG + 0], dihedrals_debug[i*NUM_DIHED_DEBUG + 1], dihedrals_debug[i*NUM_DIHED_DEBUG + 2]);
				fprintf(dihed_file, "%f\t\t%f\t\t%f\n", dihedrals_debug[i*NUM_DIHED_DEBUG + 0], dihedrals_debug[i*NUM_DIHED_DEBUG + 1], dihedrals_debug[i*NUM_DIHED_DEBUG + 2]);
			}
			fclose(dihed_file);
			free(dihedrals_debug);

/*
			int j;
			dihed_file = fopen("C:\\UserData\\Narayan\\MD_SIM\\dihed_host_debug.dat", "w");
			for (i=0;i<nAtom; i++){
				fprintf(dihed_file, "%d %d\t\t", i, dihedrals_index[i]);
				for (j=0; j<dihedrals_index[i]; j++){
				fprintf(dihed_file, "%d ", dihedrals_index[(j+1)*nAtom + i]);
				}
				fprintf(dihed_file, "\n");
			}
			fclose(dihed_file);
*/

#endif

#ifdef DEBUG_NONB
			float *debug_nonb;
			debug_nonb = (float *) malloc(fsize*NUM_DEBUG_NONB);
			
			cudaMemcpy(debug_nonb, debug_nonbd, fsize*NUM_DEBUG_NONB, cudaMemcpyDeviceToHost);

			FILE *nonb_debug_file;
			nonb_debug_file = fopen("nonb_debug_file.dat", "w");
			for (int i=0; i<nAtom; i++){
				fprintf(nonb_debug_file, "%6.3f\t%6.3f\t%6.3f\t%6.3f\n", 
						debug_nonb[NUM_DEBUG_NONB*i + 0], debug_nonb[NUM_DEBUG_NONB*i + 1], debug_nonb[NUM_DEBUG_NONB*i + 2], debug_nonb[NUM_DEBUG_NONB*i + 3]);
			}
			fclose(nonb_debug_file);	
			free(debug_nonb);
#endif

#ifdef NUM_DEBUG_CELL
			float *debug_cell;
			debug_cell = (float *) malloc(fsize*NUM_DEBUG_CELL);
			
			cudaMemcpy(debug_cell, debug_celld, fsize*NUM_DEBUG_CELL, cudaMemcpyDeviceToHost);

			FILE *cell_debug_file;
			cell_debug_file = fopen("cell_debug_file.dat", "w");
			for (int i=0; i<nAtom; i++){
				fprintf(cell_debug_file, "%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\n", 
						debug_cell[NUM_DEBUG_CELL*i + 0], debug_cell[NUM_DEBUG_CELL*i + 1], debug_cell[NUM_DEBUG_CELL*i + 2], debug_cell[NUM_DEBUG_CELL*i + 3]
				,debug_cell[NUM_DEBUG_CELL*i + 4], debug_cell[NUM_DEBUG_CELL*i + 5], debug_cell[NUM_DEBUG_CELL*i + 6]);
			}
			fclose(cell_debug_file);	
			free(debug_cell);
#endif

#ifdef DEBUG_PME
	float *pme_debug;
	int total_cells = NBlocks*BlockSize;//TOTAL_PME_LATTICES;//FFTX * FFTY * FFTZ;
	pme_debug = (float *) malloc(total_cells * NUM_DEBUG_PME*sizeof(float));
	cudaMemcpy(pme_debug, pme_debug_d, total_cells*NUM_DEBUG_PME*sizeof(float), cudaMemcpyDeviceToHost);
	
	/*
	FILE *pme_debug_file;
		pme_debug_file = fopen("C:\\UserData\\Narayan\\debug_script\\pme_debug_file.dat", "w");
		for (int i=0; i<total_cells; i++){
			for (int j=0; j<NUM_DEBUG_PME; j++)
				fprintf(pme_debug_file, "%f\t", pme_debug[NUM_DEBUG_PME*i + j]);
			fprintf("\n");
		}
	fclose(pme_debug_file);
	*/
	for (int i=0; i<total_cells; i++)
		if (pme_debug[NUM_DEBUG_PME*i + 0]!=0)
			printf("%d\t%f\n", i, pme_debug[NUM_DEBUG_PME*i + 0]);

	free(pme_debug);
#endif

#ifdef DEBUG_NBLIST
	float *debug_nblist;
	int size = NumCells_Maximum*MAX_NBCELL;
	debug_nblist = (float *) malloc(size*NUM_DEBUG_NBLIST*sizeof(float));
	cudaMemcpy(debug_nblist , debug_nblistd, size*NUM_DEBUG_NBLIST*sizeof(float), cudaMemcpyDeviceToHost);
	
	FILE *debug_nblist_file;
		debug_nblist_file = fopen("nblist_debug_file.dat", "w");
		for (int i=0; i<size; i++){
		fprintf(debug_nblist_file, "%f\t%f\t%f\t%f\n", debug_nblist[NUM_DEBUG_NBLIST*i + 0], debug_nblist[NUM_DEBUG_NBLIST*i + 1],
			debug_nblist[NUM_DEBUG_NBLIST*i + 2], debug_nblist[NUM_DEBUG_NBLIST*i + 3]);
		}
	fclose(debug_nblist_file);
	free(debug_nblist);
#endif

	return;
}

//===================================================
void capture_Excllist(){
//===================================================

	int i, j;
	FILE *excl_file = fopen("excllist_debug.dat", "w");
	for (i=0; i<nAtom; i++){
		fprintf(excl_file, "%d\t%d", i, excllist[WorkgroupSize*0 + i]);
		
		fprintf(excl_file, "\t - ");
		for (j=1; j<=excllist[WorkgroupSize*0 + i]; j++)
			fprintf(excl_file, "%d ", excllist[WorkgroupSize*j + i]);

		fprintf(excl_file, "\n");
	}
	fclose(excl_file);
}

//===================================================
void check_capture_nbcell(){
//===================================================
	int cell_neighbors = MAX_NBCELL;
	int *cell_nonbond_h, *num_nonbond_h;

	int cell, x, y, z, n, atomid;
	int x1, y1, z1;

	int *atom_freqs;
	atom_freqs = (int *) malloc(isize);
	for (int i=0; i<WorkgroupSize; i++) atom_freqs[i]=0;

	cell_nonbond_h = (int*) malloc(NumCells_Maximum*cell_neighbors*sizeof(int));
	num_nonbond_h = (int*) malloc(NumCells_Maximum*sizeof(int));

	cudaMemcpy(cell_nonbond_h, cell_nonbond, sizeof(int)*NumCells_Maximum*cell_neighbors, cudaMemcpyDeviceToHost);
	cudaMemcpy(num_nonbond_h, num_nonbond, sizeof(int)*NumCells_Maximum, cudaMemcpyDeviceToHost);
	cudaMemcpy(r, r4d, f4size, cudaMemcpyDeviceToHost);

	FILE* Cell_List = fopen("Cell_List.dat", "w");
	FILE* Freq_List = fopen("Freq_List.dat", "w");
	FILE* Num_List = fopen("Num_List.dat", "w");

	for (cell=0;cell<NumCells_Maximum; cell++){
		z = cell/(NumCells.x*NumCells.y);
		y = (cell%(NumCells.x*NumCells.y))/(NumCells.x);
		x = (cell%(NumCells.x*NumCells.y))%(NumCells.x);

		fprintf(Num_List, "%d\t%d\t%d\t\t%d\n", z, y, x, num_nonbond_h[cell]);
		fprintf(Cell_List, "%d\t%d\t%d\t\t", z, y, x);

		for (n=0; n<num_nonbond_h[cell]; n++){
			atomid = cell_nonbond_h[n*NumCells_Maximum + cell];
			atom_freqs[atomid]++;
			fprintf(Cell_List, "%d\t", atomid);

			x1 = update_coords(r[atomid].x, Region.x)/CellSize.x;
			y1 = update_coords(r[atomid].y, Region.y)/CellSize.y;
			z1 = update_coords(r[atomid].z, Region.z)/CellSize.z;
			if ((x != x1)||(y != y1)||(z != z1))
				printf("Atom Cell Mismatch... in %d at %f %f %f in %d %d %d as opposed to %d %d %d\n", 
				atomid, r[atomid].x, r[atomid].y, r[atomid].z, x, y, z, x1, y1, z1);
		}
		fprintf(Cell_List, "\n");
	}

	for (atomid=0; atomid<nAtom; atomid++){
		fprintf(Freq_List, "%d\t%d\n", atomid, atom_freqs[atomid]);
		if (atom_freqs[atomid]!=1){
			x1 = update_coords(r[atomid].x, Region.x)/CellSize.x;
			y1 = update_coords(r[atomid].y, Region.y)/CellSize.y;
			z1 = update_coords(r[atomid].z, Region.z)/CellSize.z;
			printf("Atom missing... in %d at %f %f %f to be in %d %d %d\n", 
				atomid, r[atomid].x, r[atomid].y, r[atomid].z, x1, y1, z1);
		}
	}
		
	fclose(Freq_List);
	fclose(Num_List);
	fclose(Cell_List);

	return;
}

#ifdef PME_CALC
/*----------------------------------------------------------------------------*/
void captureLatticeList(int task, int report, int suffix){
/*----------------------------------------------------------------------------*/
	int *cellLh;
	int *numLh;
	int x, y, z, cell, n, atomid;

	int *atom_freqs;
	atom_freqs = (int *)malloc(isize);

	int *prev_atom_freqs;
	prev_atom_freqs = (int *)malloc(isize);
	
	FILE *Atom_Lattice;
	int atom_lattice = -1;//1690;//430;//6920;//
	
	float4 *r4_scaled;
	r4_scaled = (float4*) malloc(f4size);
	cudaMemcpy(r4_scaled, r4d_scaled, f4size, cudaMemcpyDeviceToHost);

	float4 *prev_r4_scaled;
	prev_r4_scaled = (float4*) malloc(f4size);
	cudaMemcpy(prev_r4_scaled, prev_r4d_scaled, f4size, cudaMemcpyDeviceToHost);

	int4 *displace;
	displace = (int4*) malloc(i4size);//for debugging purposes...
	cudaMemcpy(displace, disp, i4size, cudaMemcpyDeviceToHost);
	
	//change the parameters below for either PME or nonbond Cells
	int total_lattices = fftx*ffty*fftz;
	int lattice_neighbors = MAXLATTICE_NEIGHBORS;
	int3 num_lattices;
	//initialize
	for (n=0;n<nAtom; n++) atom_freqs[n]=0;

	cellLh = (int*) malloc(total_lattices*lattice_neighbors*sizeof(int));
	numLh = (int*) malloc(total_lattices*sizeof(int));

	if (task==0){
		//Use this for CELL_BASED debugging
		//cudaMemcpy(cellLh, cell_nonbond, sizeof(int)*total_cells*cell_neighbors, cudaMemcpyDeviceToHost);
		//cudaMemcpy(numLh, num_nonbond, sizeof(int)*total_cells, cudaMemcpyDeviceToHost);

		//Use this for PME debugging..
		cudaMemcpy(cellLh, cellL, sizeof(int)*total_lattices*lattice_neighbors, cudaMemcpyDeviceToHost);
		cudaMemcpy(numLh, numL, sizeof(int)*total_lattices, cudaMemcpyDeviceToHost);

		char Cell_ListFile[1000];
		char Freq_ListFile[1000];
		char Num_ListFile[1000];
		sprintf(Cell_ListFile, "Cell_List%d.dat", suffix);
		sprintf(Freq_ListFile, "Freq_List%d.dat", suffix);
		sprintf(Num_ListFile, "Num_List%d.dat", suffix);
		
		//Cell_List = fopen(Cell_ListFile, "w");
		//Freq_List = fopen(Freq_ListFile, "w");
		//Num_List = fopen(Num_ListFile, "w");
		
		//This is to capture the position and the list of cells for a particular atom...
		if (atom_lattice>=0) {
			char Atom_LatticeFile[1000];
			sprintf(Atom_LatticeFile, "Atom_Lattice%d_%d.dat", atom_lattice, suffix);
			Atom_Lattice = fopen(Atom_LatticeFile, "w");
			fprintf(Atom_Lattice, "%d\n", atom_lattice);
			fprintf(Atom_Lattice, "%f %f %f %f\n", r4_scaled[atom_lattice].x, r4_scaled[atom_lattice].y, r4_scaled[atom_lattice].z, r4_scaled[atom_lattice].w );
		}
	}
	//Write Cell List...
	for (cell=0;cell<total_lattices; cell++){
		z = cell/(num_lattices.x*num_lattices.y);
		y = (cell%(num_lattices.x*num_lattices.y))/(num_lattices.x);
		x = (cell%(num_lattices.x*num_lattices.y))%(num_lattices.x);
		//fprintf(Cell_List, "%d %d %d\t%d\t", x, y, z, numLh[cell]);

		for (n=0; n<nAtom; n++)
			prev_atom_freqs[n] = atom_freqs[n];

		for (n=0; n<numLh[cell]; n++){
			atomid = cellLh[cell*lattice_neighbors + n];
			atom_freqs[atomid]++;
			if ((atom_freqs[atomid] - prev_atom_freqs[atomid])>1) {
				printf("Cell %d,%d,%d Step %d, AtomID - %d, currpos %f,%f,%f, prevpos %f,%f,%f displacement %d,%d,%d currfreq %d, prevfreq %d\n", 
					x, y, z, stepCount, atomid, 
					r4_scaled[atomid].x, r4_scaled[atomid].y, r4_scaled[atomid].z, 
					prev_r4_scaled[atomid].x, prev_r4_scaled[atomid].y, prev_r4_scaled[atomid].z, 
					displace[atomid].x, displace[atomid].y, displace[atomid].z,
					atom_freqs[atomid], prev_atom_freqs[atomid]);
			}
			//fprintf(Cell_List, "%d ", atomid);
			if (atomid==atom_lattice) fprintf(Atom_Lattice, "%d\t%d\t%d\n", x, y, z);
		}
		//fprintf(Cell_List, "\n");
	}

	//Write Freq List... 3 lines
	//fprintf(Freq_List, "AtomId\tFrequency\n");
	for (n=0;n<nAtom;n++){
		//if ((displace[n].x!=0)||(displace[n].y!=0)||(displace[n].z!=0))
		//	printf("Atom ID-%d, currpos %f,%f,%f, prevpos %f,%f,%f displacement %d,%d,%d\n",
		//	n, r4_scaled[n].x, r4_scaled[n].y, r4_scaled[n].z, 
		//			prev_r4_scaled[n].x, prev_r4_scaled[n].y, prev_r4_scaled[n].z, 
		//			displace[n].x, displace[n].y, displace[n].z);
		if (atom_freqs[n]!=64) printf("Frequency Error with atom %d and frequency %d\n", n, atom_freqs[n]);
		//fprintf(Freq_List, "%d\t%d\n", n, atom_freqs[n]);
	}

	//Write Num List... 3 lines
	//fprintf(Num_List, "CellID\tNeighbors\n");
	//for (n=0;n<total_cells;n++)
		//fprintf(Num_List, "%d\t%d\n", n, numLh[n]);

	free(numLh);
	free(cellLh);
	//fclose(Num_List);
	//fclose(Freq_List);
	//fclose(Cell_List);
	if (atom_lattice >= 0) fclose(Atom_Lattice); 
}
#endif

/*----------------------------------------------------------------------------*/
void capture_nblist(){
/*----------------------------------------------------------------------------*/
int *nblist;
int i,j,n;
nblist = (int*) malloc(isize*MAXNB);
FILE *nblist_file, *nbnum_file;
nblist_file = fopen("NonBondList.dat", "w");
nbnum_file = fopen("NonBondNum.dat", "w");

cudaMemcpy(nblist, nblistd, isize*MAXNB, cudaMemcpyDeviceToHost);

for (i=0;i<nAtom;i++){
	n = nblist[i];
	fprintf(nbnum_file, "%d\t%d\n", i,n);
	fprintf(nblist_file, "%d\t%d\t", i,n);
	for (j=1;j<=n;j++){
		fprintf(nblist_file, "%d ", nblist[j*WorkgroupSize + i]);
	}
	fprintf(nblist_file, "\n");
}
fclose(nblist_file);
fclose(nbnum_file);
free(nblist);
}

//////////////////////////////////////////
#ifdef PME_CALC
//////////////////////////////////////////
/*----------------------------------------------------------------------------*/
void capture_Q(){
/*----------------------------------------------------------------------------*/
	int i,j,k;
	float real, imag;
	//int fftx=FFTX, ffty=FFTY, fftz=FFTZ;
	cufftComplex *Q;

	Q = (cufftComplex *) malloc(Total_PME_Lattices*sizeof(cufftComplex));

	cudaMemcpy(Q, Qd, sizeof(cufftComplex)*Total_PME_Lattices, cudaMemcpyDeviceToHost);
	FILE *Q_matrix;
	Q_matrix = fopen("Q_matrix.dat", "w");
	for (k=0;k<fftz; k++)
	for (j=0;j<ffty; j++){
	for (i=0;i<fftx; i++){
			real = Q[k*fftx*ffty + j*fftx + i].x;
			imag = Q[k*fftx*ffty + j*fftx + i].y;
			//fprintf(Q_matrix, "%f%c%fi\t", real, ((imag>=0)?'+':'-'), abs(imag));
			fprintf(Q_matrix, "%12.8f %12.8f\t", real, imag);
			//fprintf(Q_matrix, "%f\t", real);
	}
		fprintf(Q_matrix, "\n");
	}
	fclose(Q_matrix);
}

//////////////////////////////////////////
#endif
//////////////////////////////////////////

/*----------------------------------------------------------------------------*/
void capture_position(int task, int stepcount){
/*----------------------------------------------------------------------------*/

	//task 0: Capture the position of atoms with atom type and suffix the file name with the stepcount..
	//task 1: Capture the distance between two atoms by appending to the file..
	//task 2: Capture the xyz and charges of all the atoms..

	cudaMemcpy(r, r4d, f4size, cudaMemcpyDeviceToHost);

	if (task==0){
		float4 *r4;
		r4 = (float4*) malloc(f4size);//for debugging purposes...
		cudaMemcpy(r4, r4d, f4size, cudaMemcpyDeviceToHost);
		FILE *xyzfile_debug;
		char filename[100];
		sprintf(filename,"%s_%d.xyz", filename_prefix, stepcount);
		xyzfile_debug = fopen(filename, "w");
		fprintf(xyzfile_debug, "%d\nMEMBRANE\n",nAtom);

		for(int n=0;n<nAtom; n++){
			r4[n].z = (r4[n].z>Region.z/2.0f)?(r4[n].z - Region.z):r4[n].z;
			r4[n].y = (r4[n].y>Region.y/2.0f)?(r4[n].y - Region.y):r4[n].y;
			r4[n].x = (r4[n].x>Region.x/2.0f)?(r4[n].x - Region.x):r4[n].x;
			fprintf(xyzfile_debug, "%s\t%f\t%f\t%f\n", atom_type[n], r4[n].x, r4[n].y, r4[n].z);
		}
		fclose(xyzfile_debug);
	}

	if (task==1){
		FILE *atomfile;
		int atomid1=7928, atomid2=7943;
		char file_suffix[100];
		float nearest_dist[3], dist;
		nearest_dist[0] = nearest_image(r[atomid1].x - r[atomid2].x, RegionH.x);
		nearest_dist[1] = nearest_image(r[atomid1].y - r[atomid2].y, RegionH.y);
		nearest_dist[2] = nearest_image(r[atomid1].z - r[atomid2].z, RegionH.z);

		dist = sqrt(nearest_dist[0]*nearest_dist[0] + nearest_dist[1]*nearest_dist[1] + nearest_dist[2]*nearest_dist[2]);

		sprintf(file_suffix, "dmpc_%d_%d", atomid1, atomid2);
		atomfile = fopen(file_suffix, "a");
		fprintf(atomfile, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", stepCount, r[atomid1].x, r[atomid1].y, r[atomid1].z, r[atomid2].x, r[atomid2].y, r[atomid2].z, dist);
		fclose(atomfile);
	}

	if (task==2){
		char xyz_charges_filename[600];
		sprintf(xyz_charges_filename, "%s_xyz_charges_%d.dat", filename_prefix, stepcount);
		FILE *xyz_charges;
		xyz_charges = fopen(xyz_charges_filename, "w");
		
		for(int n=0;n<nAtom; n++)
			fprintf(xyz_charges, "%12.8f\t%12.8f\t%12.8f\t%12.8f\n", r[n].x, r[n].y, r[n].z, r[n].w);
		fclose(xyz_charges);
	}
}
