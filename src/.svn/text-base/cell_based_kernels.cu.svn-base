/******************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/******************************************************************************/

#include "globals.h"
#include "cucomplexops.h"

//------------------------------------------------------------------------------
__global__ void CheckCellOccupancy(float4* rd, int* num_nonbond){
//------------------------------------------------------------------------------
	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	//the x, y, and z indices of the cell..
	unsigned int xc = 0;
	unsigned int yc = 0;
	unsigned int zc = 0;
	unsigned int cell = 0;
	float4 pos;
#ifdef PCONSTANT
	int4 num_cells = max_numcells; //tex1Dfetch(texnumcells, 0);
	float3 box = float4_to_float3(tex1Dfetch(texbox, 0));
#endif

	//Check the number of Cell occupancy...
	if(gtid < natomd){
		pos = rd[gtid];
		//determine the x, y, z indices of the cell that the atom belongs to..
		xc = update_coords(pos.x, box.x) / CELL_X;
		yc = update_coords(pos.y, box.y) / CELL_Y;
		zc = update_coords(pos.z, box.z) / CELL_Z;

		cell = zc * num_cells.x * num_cells.y + yc * num_cells.x + xc;

		atomicAdd(&num_nonbond[cell], 1);
	}

	return;
}

//------------------------------------------------------------------------------
__global__ void CheckNonbondNum(float4* rd,
                                unsigned int* nblistd,
                                unsigned int* cell_nonbond,
                                int* num_nonbond
#ifdef SEARCH_EXCLUSION_LIST
                              , int* excllistd
#endif
                                              ){
//------------------------------------------------------------------------------
	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	float3 pos;
	float3 dr;

	int neighbor_atomid;

	short int nnb = 0;
	short int cell;
	short int i;
	short int n;

	char cx;
	char cy;
	char cz;

	char cx1;
	char cy1;
	char cz1;

	char cix;
	char ciy;
	char ciz;

	char xs;
	char xe;
	char ys;
	char ye;
	char zs;
	char ze;
#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
	int4 num_cells = tex1Dfetch(texnumcells, 0);
	int total_cells = max_numcells.w; //num_cells.w;
	float3 box = float4_to_float3(tex1Dfetch(texbox, 0));
#endif

	//copy this atom's coordinates to old coordinate
	//array (for update check later) and to pos...
	pos = float4_to_float3(rd[gtid]);

	//determine the cell the current atom belongs to..
	cx = update_coords(pos.x, box.x) / CELL_X;
	cy = update_coords(pos.y, box.y) / CELL_Y;
	cz = update_coords(pos.z, box.z) / CELL_Z;

#ifdef CELL5
	xs = ((cx == 0) || (cx == 1)) ? (cx - 3) : (cx - 2);
	ys = ((cy == 0) || (cy == 1)) ? (cy - 3) : (cy - 2);
	zs = ((cz == 0) || (cz == 1)) ? (cz - 3) : (cz - 2);

	xe = ((cx == num_cells.x - 3) || (cx == num_cells.x - 2)) ?
	     (cx + 3) : (cx + 2);

	ye = ((cy == num_cells.y - 3) || (cy == num_cells.y - 2)) ?
	     (cy + 3) : (cy + 2);

	ze = ((cz == num_cells.z - 3) || (cz == num_cells.z - 2)) ?
	     (cz + 3) : (cz + 2);
#else
	xs = (cx == 0) ? (cx - 2) : (cx - 1);
	ys = (cy == 0) ? (cy - 2) : (cy - 1);
	zs = (cz == 0) ? (cz - 2) : (cz - 1);

	xe = (cx == num_cells.x - 2) ? (cx + 2) : (cx + 1);
	ye = (cy == num_cells.y - 2) ? (cy + 2) : (cy + 1);
	ze = (cz == num_cells.z - 2) ? (cz + 2) : (cz + 1);
#endif

	for(cix = xs; cix <= xe; cix++){

		cx1 = (cix < 0) ? (cix + num_cells.x) :
		                  ((cix >= num_cells.x) ? (cix - num_cells.x) : cix);

		for(ciy = ys; ciy <= ye; ciy++){

			cy1 = (ciy < 0) ? (ciy + num_cells.y) :
			                  ((ciy >= num_cells.y) ? (ciy - num_cells.y) : ciy);

			for(ciz = zs; ciz <= ze; ciz++){

				cz1 = (ciz < 0) ? (ciz + num_cells.z) :
				                  ((ciz >= num_cells.z) ? (ciz - num_cells.z) : ciz);

#ifdef PCONSTANT //use the maximum allowed cells to calculate the cell index...
				cell = cz1 * max_numcells.x * max_numcells.y +
				       cy1 * max_numcells.x + cx1;
#else
				cell = cz1 * num_cells.x * num_cells.y + cy1 * num_cells.x + cx1;
#endif

				//iterate through the atoms in the current cell..
				n = num_nonbond[cell];

				for(i = 0;i < n; i++){
					neighbor_atomid = unpack_atomid(cell_nonbond[i * total_cells + cell]);
					dr = pos - float4_to_float3(tex1Dfetch(texcrd, neighbor_atomid));

					//nearest image
					dr.x = __my_fadd(dr.x,
					                 -__my_fadd(__mycopysignf(boxH.x,
					                                          __my_fadd(dr.x,-boxH.x)),
					                            __mycopysignf(boxH.x,
					                                          __my_fadd(dr.x, boxH.x))));

					dr.y = __my_fadd(dr.y,
					                 -__my_fadd(__mycopysignf(boxH.y,
					                                          __my_fadd(dr.y,-boxH.y)),
					                            __mycopysignf(boxH.y,
					                                          __my_fadd(dr.y,boxH.y))));

					dr.z = __my_fadd(dr.z,
					                 -__my_fadd(__mycopysignf(boxH.z,
					                                          __my_fadd(dr.z,-boxH.z)),
					                            __mycopysignf(boxH.z,
					                                          __my_fadd(dr.z,boxH.z))));

					if((dr % dr < __my_fmul(cutmaxd, cutmaxd)) &&
					   (gtid != neighbor_atomid) &&
					   (gtid < natomd) &&
					   (neighbor_atomid < natomd)
#ifdef SEARCH_EXCLUSION_LIST
//&&(exclGPU(excllistd, gtid, neighbor_atomid, 1, excllistd[gtid]) == 0)
#endif
                                       ){

						nnb++;
					}
				}//for (i=0;i<n;i++){
			}//for (ciz = zs; ciz<=ze; ciz++){
		}//for (ciy = ys; ciy<=ye; ciy++){
	}//for (cix = xs; cix<=xe; cix++){

	nblistd[gtid] = nnb;

	return;
}


//------------------------------------------------------------------------------
__global__ void CellBuild(float4* rd,
                          float4* roldd,
                          unsigned int* cell_nonbond,
                          int* num_nonbond,
                          unsigned int* nblistd
#ifdef NUM_DEBUG_CELL
                        , float* debug_celld
#endif
                                            ){
//------------------------------------------------------------------------------
	//This kernel builds the initial cell list
	//The domain is divided into cells of lengths CELL_X, CELL_Y and CELL_Z
	//whose values are defined in defs.h

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	//the x, y, and z indices of the cell..
	unsigned int xc = 0;
	unsigned int yc = 0;
	unsigned int zc = 0;

	unsigned int n = 0;
	unsigned int cell = 0;
	float4 pos;

#ifdef PCONSTANT
	int4 num_cells = max_numcells; //tex1Dfetch(texnumcells, 0);
	int total_cells = num_cells.w; //variable used for clarity...
	float3 box = float4_to_float3(tex1Dfetch(texbox, 0));
#endif

	float3 cellsize = {CELL_X, CELL_Y, CELL_Z};

	//Build the cells with their neighbors...
	if(gtid < natomd){
		pos = rd[gtid];
		//determine the x, y, z indices of the cell that the atom belongs to..
		xc = update_coords(pos.x, box.x) / cellsize.x;
		yc = update_coords(pos.y, box.y) / cellsize.y;
		zc = update_coords(pos.z, box.z) / cellsize.z;

		cell = zc * num_cells.x * num_cells.y + yc * num_cells.x + xc;

		n = atomicAdd(&num_nonbond[cell], 1);

		//once the cells are identified, update the list..
		cell_nonbond[n * total_cells + cell] = pack_type_and_atomid(gtid);

		roldd[gtid] = pos;
	}

	//initliaze the number of nonbond neighbors for all atoms...
	nblistd[gtid] = 0;

#ifdef NUM_DEBUG_CELL
	__syncthreads();
	debug_celld[NUM_DEBUG_CELL * gtid + 0] = gtid;
	debug_celld[NUM_DEBUG_CELL * gtid + 1] = xc;
	debug_celld[NUM_DEBUG_CELL * gtid + 2] = yc;
	debug_celld[NUM_DEBUG_CELL * gtid + 3] = zc;
	debug_celld[NUM_DEBUG_CELL * gtid + 4] = cell;
	debug_celld[NUM_DEBUG_CELL * gtid + 5] = n;
	debug_celld[NUM_DEBUG_CELL * gtid + 6] = cell_nonbond[n * total_cells + cell];
#endif

	return;
}

//------------------------------------------------------------------------------
//after updating the migrant atoms from the new cells,
//remove them from the current cell list...
__global__ void CellClean(float4* rd,
                          unsigned int* cell_nonbond,
                          int* num_nonbond){
//------------------------------------------------------------------------------
	short int cx;
	short int cy;
	short int cz;
	short int xc;
	short int yc;
	short int zc;

	int n;
	int atomid;
	int i;

	float4 pos;

	unsigned int cell;

#ifdef PCONSTANT
	int4 num_cells = max_numcells; //tex1Dfetch(texnumcells, 0);
	int total_cells = num_cells.w; //variable used for clarity...
	float3 box = float4_to_float3(tex1Dfetch(texbox, 0));
#endif
	float3 cellsize = {CELL_X, CELL_Y, CELL_Z};

	//unsigned int gtid = COMPUTE_GLOBAL_THREADID

	xc = threadIdx.x;
	yc = blockIdx.x;
	zc = blockIdx.y;

	cell = zc * (num_cells.x * num_cells.y) + yc * num_cells.x + xc;

	n = num_nonbond[cell];
	//update_num=n;

	//for (i=0; i<n_prev; i++){
	for(i = 0; i < n; i++){

		atomid = unpack_atomid(cell_nonbond[i * total_cells + cell]);

		pos = rd[atomid];

		cx = update_coords(pos.x, box.x) / cellsize.x;
		cy = update_coords(pos.y, box.y) / cellsize.y;
		cz = update_coords(pos.z, box.z) / cellsize.z;

		if((xc != cx) || (yc != cy) || (zc != cz)){
			//if the atom doesnot belong in the cell, remove it by moving
			//another atom from the end of list to the current position...
			n--;

			int pos_l = i * total_cells + cell;
			int pos_r = n * total_cells + cell;
			cell_nonbond[pos_l] = cell_nonbond[pos_r];

			//process the current atom again...
			i--;
			//if (n<n_prev) {i--; n_prev--;}
		}
	}

	num_nonbond[cell] = n;

	return;
}

//------------------------------------------------------------------------------
__global__ void CellUpdate(float4* rd,
                           float4* roldd,
                           unsigned int* cell_nonbond,
                           int* num_nonbond
#ifdef PCONSTANT
                         , float4* prev_boxLengthd
#endif
                                                  ){
//------------------------------------------------------------------------------
	unsigned int gtid = COMPUTE_GLOBAL_THREADID
	int update_num;
	int new_cell;//n, i, atomid,

	short int ocx = 0;
	short int ocy = 0;
	short int ocz = 0;
	short int ncx = 0;
	short int ncy = 0;
	short int ncz = 0;

	float4 new_pos;
	float4 old_pos;

#ifdef PCONSTANT
	int4 num_cells = max_numcells; //tex1Dfetch(texnumcells, 0);
	int total_cells = num_cells.w; //variable used for clarity...
	float3 box = float4_to_float3(tex1Dfetch(texbox, 0));
#endif
	float3 cellsize = {CELL_X, CELL_Y, CELL_Z};

	if(gtid < natomd){
		new_pos = rd[gtid];
		old_pos = roldd[gtid];

		//compute old cell
#ifdef PCONSTANT
		ocx = update_coords(old_pos.x, prev_boxLengthd[0].x) / cellsize.x;
		ocy = update_coords(old_pos.y, prev_boxLengthd[0].y) / cellsize.y;
		ocz = update_coords(old_pos.z, prev_boxLengthd[0].z) / cellsize.z;
#else
		ocx = update_coords(old_pos.x, box.x) / cellsize.x;
		ocy = update_coords(old_pos.y, box.y) / cellsize.y;
		ocz = update_coords(old_pos.z, box.z) / cellsize.z;
#endif

		//compute new cell
		ncx = update_coords(new_pos.x, box.x) / cellsize.x;
		ncy = update_coords(new_pos.y, box.y) / cellsize.y;
		ncz = update_coords(new_pos.z, box.z) / cellsize.z;
	}

	if((ocx != ncx) || (ocy != ncy) || (ocz != ncz)){
		new_cell = ncz * num_cells.x * num_cells.y + ncy * num_cells.x + ncx;

		update_num = atomicAdd(&num_nonbond[new_cell], 1);

		int index = update_num * total_cells + new_cell;
		cell_nonbond[index] = pack_type_and_atomid(gtid);
	}

	return;
}

#ifdef PCONSTANT
//------------------------------------------------------------------------------
__global__ void reduce_virial(float4* v4d,
                              float4* boxVelocd,
                              float4* boxAcceld,
                              float4* kineticd,
                              float4* viriald
#ifdef USE_NPT
                            , float4* boxLengthd,
                              float4* propertiesd
#endif
                                                 ){ //float4 *f4d, float4 *r4d,
//------------------------------------------------------------------------------
//reduce the virial to partial sums...
//each block reduces a part of the virial array to a partial sum
//via parallel scan, this is then passed
//to final sum to obtain a single value...
//------------------------------------------------------------------------------

	__shared__ float3 partial_sum_virial[MAX_BLOCK_SIZE];
	__shared__ float3 partial_sum_KE[MAX_BLOCK_SIZE];
	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	float3 Val_Zero = {0.0f, 0.0f, 0.0f};

	//compute virial and kinetic... and load data into shared memory...
	float mymass = 0.0f;

	float3 box = float4_to_float3(tex1Dfetch(texbox, 0));
	float3 scaled_vel;
	// = float4_to_float3(v4d[gtid]) -
	//float4_to_float3(tex1Dfetch(texcrd, gtid)*boxVelocd[0])/box;

	if(gtid < natomd){
		scaled_vel.x = v4d[gtid].x - tex1Dfetch(texcrd, gtid).x *
		               boxVelocd[0].x / box.x;

		scaled_vel.y = v4d[gtid].y - tex1Dfetch(texcrd, gtid).y *
		               boxVelocd[0].y / box.y;

		scaled_vel.z = v4d[gtid].z - tex1Dfetch(texcrd, gtid).z *
		               boxVelocd[0].z / box.z;
	}

	float Volume;
	float boxAccel;

	//partial_sum_virial[threadIdx.x] =
	//(float4_to_float3(f4d[gtid])*float4_to_float3(r4d[gtid]));
	//compute virial

	if(gtid < natomd){
		partial_sum_virial[threadIdx.x] = float4_to_float3(viriald[gtid]);
		mymass = tex1Dfetch(texprm, tex1Dfetch(textype, gtid)).x;
		partial_sum_KE[threadIdx.x] = (scaled_vel * scaled_vel) * mymass;
		//compute 2*kinetic
	}
	else{//pad the rest of the values with zero...
		partial_sum_virial[threadIdx.x] = Val_Zero;
		partial_sum_KE[threadIdx.x] = Val_Zero;
	}

	int offset = (MAX_BLOCK_SIZE >> 1);

	__syncthreads();

	while(offset >0){

		if(threadIdx.x < offset){
			int index = threadId.x + offset;
			partial_sum_virial[threadIdx.x] += partial_sum_virial[index];
			partial_sum_KE[threadIdx.x] += partial_sum_KE[index];
		}

		offset >>= 1;

		__syncthreads();
	}

	//write the results back to global memory...
	if(threadIdx.x == 0){
		kineticd[blockIdx.x] = float3_to_float4(partial_sum_KE[0]);
		//if (threadIdx.x == GPU_WARP_SIZE)
		viriald[blockIdx.x] = float3_to_float4(partial_sum_virial[0]);
	}

	//////perform a global synchronization on all threads//////////////
	///////// implementation adapted from SDK example threadfence//////////////
	__threadfence();

	__shared__ bool amLast;
	unsigned int ticket;

	//Thread 0 of each block takes a ticket
	if(threadIdx.x == 0 ){
		ticket = atomicInc(&retirementCount, gridDim.x);
		//If the ticket ID is equal to the number of blocks, we are the last block!
		amLast = (ticket == gridDim.x - 1);
	}
	__syncthreads();

	//////end global synchronization///////

	int i;
	int partialsize = natomd / MAX_BLOCK_SIZE + 1;

	if(amLast){

		//copy only the first block into the shared memory...
		if(threadIdx.x < partialsize){
			//virial
			partial_sum_virial[threadIdx.x] = float4_to_float3(viriald[threadIdx.x]);

			//KE
			partial_sum_KE[threadIdx.x] = float4_to_float3(kineticd[threadIdx.x]);
		}

		//if partialsize < blocksize, pad the rest of the values with zero...
		if(threadIdx.x >= partialsize){
			partial_sum_virial[threadIdx.x] = Val_Zero;
			partial_sum_KE[threadIdx.x] = Val_Zero;
		}

		__syncthreads();


		//reduce the partial sum across multiple blocks to a single block
		//via simple striding sum..
		for(i = threadIdx.x + MAX_BLOCK_SIZE; i < partialsize; i += MAX_BLOCK_SIZE){
			partial_sum_virial[threadIdx.x] += float4_to_float3(viriald[i]);
			partial_sum_KE[threadIdx.x] += float4_to_float3(kineticd[i]);
		}

		__syncthreads();
		//now reduce the block to a single final sum...

		offset = (MAX_BLOCK_SIZE >> 1);

		while(offset > 0){

			if(threadIdx.x < offset){
				int index = threadIdx.x + offset;
				partial_sum_virial[threadIdx.x] += partial_sum_virial[index];
				partial_sum_KE[threadIdx.x] += partial_sum_KE[index];
			}

			offset >>= 1;

			__syncthreads();
		}

		if(threadIdx.x == 0){
			kineticd[0] = float3_to_float4(partial_sum_KE[0]);
			viriald[0] = float3_to_float4(partial_sum_virial[0]) +
			             tex1Dfetch(texRBSRat, 0) * (1.0f * eEwselfd);

			Volume = box.x * box.y * box.z;
			//Enable for noncubic Nose'-Klein type barostat...
			//boxAcceld[0].x = (kineticd[0].x + viriald[0].x -
			//PREF*PU*Volume)*(1.0f/(PMASS*box.x));
			//boxAcceld[0].y = (kineticd[0].y + viriald[0].y -
			//PREF*PU*Volume)*(1.0f/(PMASS*box.y));
			//boxAcceld[0].z = (kineticd[0].z + viriald[0].z -
			//PREF*PU*Volume)*(1.0f/(PMASS*box.z));
			//boxAcceld[0].w = 0.0f;
			//End Nose-Klein Barostat...

			//Enable for Andersen type barostat...
			boxAccel = (kineticd[0].x + kineticd[0].y + kineticd[0].z +
			            viriald[0].x + viriald[0].y + viriald[0].z) /
			            (3 * Volume) - pRefd * PU;

			boxAccel = boxAccel / (3.0f * pmass_cubicd * box.x * box.x) -
			           2 * boxVelocd[0].x * boxVelocd[0].x / box.x;

			boxAcceld[0].x = boxAccel;
			boxAcceld[0].y = boxAccel * boxLengthRatiod.y;
			boxAcceld[0].z = boxAccel * boxLengthRatiod.z;
			boxAcceld[0].w = 0.0f;
			//////////End Andersen Barostat////////

			// scalar virial, pressure, kinetic, temperature
			propertiesd[0].x = viriald[0].x + viriald[0].y + viriald[0].z;
			propertiesd[0].y = (68558.64100846418f /
			                    (3.0f * boxLengthd[0].x *
			                     boxLengthd[0].y * boxLengthd[0].z)) *
			                   (kineticd[0].x + kineticd[0].y +
			                    kineticd[0].z + propertiesd[0].x);

			propertiesd[0].z = (kineticd[0].x + kineticd[0].y + kineticd[0].z) * 0.5f;
			propertiesd[0].w = 2 * propertiesd[0].z / RC /
			                   (3.0f * natomd - 3.0f - numconstraintsd);

			retirementCount = 0;
		}

		//now partial_sum[0] has the final value..
		//now compute the relavant physical parameters..
	}//if (amLast)
}
#endif

//------------------------------------------------------------------------------
__global__ void reduce_PE(float4* f4d, float4* f4d_nonbond, float4* f4d_bonded){
//------------------------------------------------------------------------------
//reduce the PE to partial sums...
//each block reduces a part of the PE to a partial sum
//via parallel scan, this is then passed to final sum to obtain a single value
//------------------------------------------------------------------------------

	__shared__ float partial_sum_PE[MAX_BLOCK_SIZE];
	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	if(gtid < natomd){
		partial_sum_PE[threadIdx.x] = f4d[gtid].w +
		                              f4d_nonbond[gtid].w +
		                              f4d_nonbond[WorkgroupSized + gtid].w +
		                              f4d_bonded[gtid].w +
		                              f4d_bonded[WorkgroupSized + gtid].w;
	}
	else{//pad the rest of the values with zero...
		partial_sum_PE[threadIdx.x] = 0.0f;
	}

	int offset = (MAX_BLOCK_SIZE >> 1);

	__syncthreads();

	while(offset > 0){

		if(threadIdx.x < offset){
			partial_sum_PE[threadIdx.x] += partial_sum_PE[threadIdx.x + offset];
		}

		offset >>= 1;

		__syncthreads();
	}

	//write the results back to global memory...
	if(threadIdx.x == 0){
		f4d[blockIdx.x].w = partial_sum_PE[0];
	}

	/////////////perform a global synchronization on all threads//////////////
	/////////implementation adapted from SDK example threadfence//////////////
	__threadfence();

	__shared__ bool amLast;
	unsigned int ticket;

	//Thread 0 of each block takes a ticket
	if(threadIdx.x == 0){
		ticket = atomicInc(&retirementCount, gridDim.x);
		// If the ticket ID is equal to the number of blocks, we are the last block!
		amLast = (ticket == gridDim.x-1);
	}

	__syncthreads();

	////////////////////end global synchronization/////////////////////

	int i;
	int partialsize = natomd/MAX_BLOCK_SIZE + 1;

	if(amLast){
		//copy only the first block into the shared memory...
		if(threadIdx.x < partialsize){
			partial_sum_PE[threadIdx.x] = f4d[threadIdx.x].w;
		}

		//if partialsize < blocksize, pad the rest of the values with zero...
		if(threadIdx.x >= partialsize){
			partial_sum_PE[threadIdx.x] = 0.0f;
		}

		__syncthreads();

		//reduce the partial sum across multiple blocks to a single block
		//via simple striding sum..
		for(i = threadIdx.x + MAX_BLOCK_SIZE; i < partialsize; i += MAX_BLOCK_SIZE){
			partial_sum_PE[threadIdx.x] += f4d[i].w;
		}

		__syncthreads();
		//now reduce the block to a single final sum...

		offset = (MAX_BLOCK_SIZE >> 1);

		while(offset >0){

			if(threadIdx.x < offset){
				partial_sum_PE[threadIdx.x] += partial_sum_PE[threadIdx.x + offset];
			}

			offset >>= 1;

			__syncthreads();
		}

		if(threadIdx.x == 0){
			f4d[0].w = partial_sum_PE[0];
			retirementCount=0;
		}
	}//if (amLast)

	//partial_sum[0] has the final value..
	//now compute the relavant physical parameters..

}


/////////////////////Reduce center of mass for restraints/////////////////
//------------------------------------------------------------------------------
__global__ void reduce_COM(char* segidd,
                           float3* com0d,
                           float3* com1d,
                           float* mass_segid0d,
                           float* mass_segid1d){
//------------------------------------------------------------------------------
//This kernel computes the center of mass of a group of atoms by their segid..

	__shared__ float3 partial_sum_COM0[MAX_BLOCK_SIZE]; //segid 0
	__shared__ float3 partial_sum_COM1[MAX_BLOCK_SIZE]; //segid 1

#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	float mymass = (gtid < natomd) ? tex1Dfetch(texprm,
	                                            tex1Dfetch(textype, gtid)).x :
	                                 0.0f;

	char segid = (gtid < natomd) ? segidd[gtid] : -1;

	float3 r;

	//read positions of the first segment
	if((gtid < natomd) && (segid == 0)){
		r = float4_to_float3(tex1Dfetch(texcrd, gtid));

		r.x = __my_add3(r.x,
		                -boxH.x,
		                -__mycopysignf(boxH.x,__my_fadd(r.x,-boxH.x)));

		r.y = __my_add3(r.y,
		                -boxH.y,
		                -__mycopysignf(boxH.y,__my_fadd(r.y,-boxH.y)));

		r.z = __my_add3(r.z,
		                -boxH.z,
		                -__mycopysignf(boxH.z,__my_fadd(r.z,-boxH.z)));

		partial_sum_COM0[threadIdx.x] = r * mymass;
	}
	else{
		partial_sum_COM0[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
	}

	//read positions of the second segment
	if((gtid < natomd) && (segid == 1)){
		r = float4_to_float3(tex1Dfetch(texcrd, gtid));

		r.x = __my_add3(r.x,
		                -boxH.x,
		                -__mycopysignf(boxH.x,__my_fadd(r.x,-boxH.x)));

		r.y = __my_add3(r.y,
		                -boxH.y,
		                -__mycopysignf(boxH.y,__my_fadd(r.y,-boxH.y)));

		r.z = __my_add3(r.z,
		                -boxH.z,
		                -__mycopysignf(boxH.z,__my_fadd(r.z,-boxH.z)));

		partial_sum_COM1[threadIdx.x] = r * mymass;
	}
	else{
		partial_sum_COM1[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
	}

	//now reduce the sums in block..
	unsigned int offset = (MAX_BLOCK_SIZE >> 1);

	__syncthreads();

	while(offset > 0){

		if(threadIdx.x < offset){
			partial_sum_COM0[threadIdx.x] += partial_sum_COM0[threadIdx.x + offset];
			partial_sum_COM1[threadIdx.x] += partial_sum_COM1[threadIdx.x + offset];
		}

		offset >>= 1;

		__syncthreads();
	}

	//write the results back to global memory...
	if(threadIdx.x == 0){
		com0d[blockIdx.x] = partial_sum_COM0[0];
		com1d[blockIdx.x] = partial_sum_COM1[0];
	}

	/////////perform a global synchronization on all threads//////////////
	///////// implementation adapted from SDK example threadfence//////////////
	__threadfence();

	__shared__ bool amLast;
	unsigned int ticket;

	//Thread 0 of each block takes a ticket
	if(threadIdx.x == 0){
		ticket = atomicInc(&retirementCount, gridDim.x);
		// If the ticket ID is equal to the number of blocks, we are the last block!
		amLast = (ticket == gridDim.x-1);
	}

	__syncthreads();

	////////////////////end global synchronization/////////////////////

	int i;
	int partialsize = natomd / MAX_BLOCK_SIZE + 1;

	if(amLast){
		//copy only the first block into the shared memory...
		if(threadIdx.x < partialsize){
			partial_sum_COM0[threadIdx.x] = com0d[threadIdx.x];
			partial_sum_COM1[threadIdx.x] = com1d[threadIdx.x];
		}

		//if partialsize < blocksize, pad the rest of the values with zero...
		if(threadIdx.x >= partialsize){
			partial_sum_COM0[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
			partial_sum_COM1[threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
		}

		__syncthreads();

		//reduce the partial sum across multiple blocks to a single block
		//via simple striding sum..
		for(i = threadIdx.x + MAX_BLOCK_SIZE; i < partialsize; i += MAX_BLOCK_SIZE){
			partial_sum_COM0[threadIdx.x] += com0d[i];
			partial_sum_COM1[threadIdx.x] += com1d[i];
		}

		__syncthreads();

		//now reduce the block to a single final sum...
		offset = (MAX_BLOCK_SIZE >> 1);

		while(offset > 0){

			if(threadIdx.x < offset){
				partial_sum_COM0[threadIdx.x] += partial_sum_COM0[threadIdx.x + offset];
				partial_sum_COM1[threadIdx.x] += partial_sum_COM1[threadIdx.x + offset];
			}

			offset >>=1;

			__syncthreads();
		}

		if(threadIdx.x == 0){
			com0d[0] = partial_sum_COM0[0] / mass_segid0d[0];
			com1d[0] = partial_sum_COM1[0] / mass_segid1d[0];
			retirementCount = 0;
		}
	}//if (amLast)

	//partial_sum[0] has the final value..
	//now compute the relavant physical parameters..
}
/////////////////////End Reduce center of mass for restraints/////////////////

//------------------------------------------------------------------------------
__global__ void reduce_PE_partial_sum(float4* f4d){
//------------------------------------------------------------------------------
//reduce the PE to partial sums...
//each block reduces a part of the PE to a partial sum
//via parallel scan, this is then passed to final sum to obtain a single value
//------------------------------------------------------------------------------

	__shared__ float partial_sum_PE[MAX_BLOCK_SIZE];
	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	if(gtid < natomd){
		partial_sum_PE[threadIdx.x] = f4d[gtid].w;
	}
	else{//pad the rest of the values with zero...
		partial_sum_PE[threadIdx.x] = 0.0f;
	}

	int offset = (MAX_BLOCK_SIZE >> 1);

	__syncthreads();

	while(offset >0){

		if(threadIdx.x < offset){
			partial_sum_PE[threadIdx.x] += partial_sum_PE[threadIdx.x + offset];
		}

		offset >>=1;

		__syncthreads();
	}

	//write the results back to global memory...
	if(threadIdx.x == 0){
		f4d[blockIdx.x].w = partial_sum_PE[0];
	}
}

//------------------------------------------------------------------------------
__global__ void reduce_PE_final_sum(float4* f4d){
//------------------------------------------------------------------------------
//The partial sum is reduced to a single sum by
//first collapsing partial sums across multiple
//blocks to a single block and performing a parallel scan on the block.
//only one block is initialized..
//------------------------------------------------------------------------------

	__shared__ float partial_sum_PE[MAX_BLOCK_SIZE];

	int i;
	int partialsize = natomd / MAX_BLOCK_SIZE + 1;

	//copy only the first block into the shared memory...
	if(threadIdx.x < partialsize){
		partial_sum_PE[threadIdx.x] = f4d[threadIdx.x].w;
	}

	//if partialsize < blocksize, pad the rest of the values with zero...
	if(threadIdx.x >= partialsize){
		partial_sum_PE[threadIdx.x] = 0.0f;
	}

	__syncthreads();

	//reduce the partial sum across multiple blocks to a single block
	//via simple striding sum..
	for(i = threadIdx.x + MAX_BLOCK_SIZE; i < partialsize; i += MAX_BLOCK_SIZE){
		partial_sum_PE[threadIdx.x] += f4d[i].w;
	}

	__syncthreads();
	//now reduce the block to a single final sum...

	int offset = (MAX_BLOCK_SIZE >> 1);

	while(offset > 0){

		if(threadIdx.x < offset){
			partial_sum_PE[threadIdx.x] += partial_sum_PE[threadIdx.x + offset];
		}

		offset >>=1;

		__syncthreads();
	}

	if(threadIdx.x == 0){
		f4d[0].w = partial_sum_PE[0];
	}

	//partial_sum[0] has the final value..
	//now compute the relavant physical parameters..

}

//------------------------------------------------------------------------------
__device__ char Search_excl_binary(int* excl,
                                   int x,
                                   int y,
                                   char start,
                                   char end){
//------------------------------------------------------------------------------
//performs a binary search on the exclusion list ...
//------------------------------------------------------------------------------
	int mid;
	int neighbor_id;

	do{
		mid = start + ((end - start) / 2);
#ifdef EXCL_SHMEM
		neighbor_id = excl[mid * CELL_ATOMS + threadIdx.x];
#else
		neighbor_id = excl[mid * WorkgroupSized + x];
#endif

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

/*
//----------------------------------------------------------------------------
__device__ char Search_exclGPU_shmem(int *excl, int y,
 unsigned char first, unsigned char last) {
//----------------------------------------------------------------------------
//Searches exclusion list on GPU for a given atom x and return the position of
//the first occurrence of a given value y
//----------------------------------------------------------------------------
	char j;

	//already checked for first and last atoms in the exclusion list in nbbuild_*
	for(j=first; j<=last; j++)
	  if(excl[MAX_NBCELL*j + threadIdx.x] == y) return 1;

	return 0;
}

//----------------------------------------------------------------------------
__device__ char exclGPU(int *excl, int x, int y, unsigned char first,
 unsigned char last) {
//----------------------------------------------------------------------------
//Searches exclusion list on GPU for a given atom x and return the position of
//the first occurrence of a given value y
//----------------------------------------------------------------------------
	char j;

	//already checked for first and last atoms in the exclusion list in nbbuild_*
	for(j=first; j<=last; j++)
	  if(excl[WorkgroupSized*j + x] == y) return 1;

	return 0;
}

*/

//------------------------------------------------------------------------------
__global__ void nbbuild_exclbitvec(float4* rd,
                                   float4* roldd,
                                   unsigned int* nblistd,
                                   unsigned int* cell_nonbond,
                                   int* num_nonbond,
                                   unsigned long long* excl_bitvecd,
                                   char* excl_bitvec_offsetd,
                                   int* excllistd
#ifdef PCONSTANT
                                 , float4* prev_boxLengthd
#endif
#ifdef DEBUG_NBLIST
                                 , float* debug_nblistd
#endif
                                                       ){
//------------------------------------------------------------------------------

	__shared__ int cell_atoms_sh[CELL_ATOMS];
	__shared__ float3 cell_atoms_pos_sh[CELL_ATOMS];

//int* cell_atoms_sh = (int *) temp_array;
//float3* cell_atoms_pos_sh = (float3 *) &cell_atoms_sh[MAX_NBCELL];

	int self_id = -1;
	int neighbor_id;

	float4 pos={0.0f, 0.0f, 0.0f, 0.0f};
	float4 dr;
#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
	int4 num_cells = tex1Dfetch(texnumcells, 0);

	int total_cells = max_numcells.w; //num_cells.w;
#endif
	//float4 neighbor_pos;
	char cx;
	char cy;
	char cz;

	char cx1;
	char cy1;
	char cz1;

	char cix;
	char ciy;
	char ciz;

	//unsigned char xc, yc, zc;
	short int n;
	short int nnb = 0;
	short int i;//neighbor_atomid,

	char xs;
	char xe;
	char ys;
	char ye;
	char zs;
	char ze;

	unsigned short int cell;

	cx = blockIdx.x % num_cells.x;
	cy = blockIdx.x / num_cells.x;
	cz = blockIdx.y;

#ifdef PCONSTANT //use the maximum num cells to calculate the cell index...
	cell = cz * (max_numcells.x * max_numcells.y) + cy * max_numcells.x + cx;
#else
	cell = cz * (num_cells.x * num_cells.y) + cy * num_cells.x + cx;
#endif

//Determine start and end of the search cells
	xs = (cx == 0) ? (cx - 2) : (cx - 1);
	ys = (cy == 0) ? (cy - 2) : (cy - 1);
	zs = (cz == 0) ? (cz - 2) : (cz - 1);

	xe = (cx == num_cells.x - 2) ? (cx + 2) : (cx + 1);
	ye = (cy == num_cells.y - 2) ? (cy + 2) : (cy + 1);
	ze = (cz == num_cells.z - 2) ? (cz + 2) : (cz + 1);

	unsigned char active_thread = (threadIdx.x < num_nonbond[cell]);

	unsigned long long exclusion_bitvector = 0;
	char exclusion_offset = 0;

	//read the position of the current atom here....
	if(active_thread){
		self_id = unpack_atomid(cell_nonbond[threadIdx.x * total_cells + cell]);
		pos = rd[self_id];

		exclusion_bitvector = excl_bitvecd[self_id];
		exclusion_offset = excl_bitvec_offsetd[self_id];
	}

	//finished reading self position...
	for(cix = xs; cix <= xe; cix++){

		cx1 = (cix < 0) ? (cix + num_cells.x) :
		                  ((cix >= num_cells.x) ? (cix - num_cells.x) : cix);

		for(ciy = ys; ciy <= ye; ciy++){

			cy1 = (ciy < 0) ? (ciy + num_cells.y) :
			                  ((ciy >= num_cells.y) ? (ciy - num_cells.y) : ciy);

			for(ciz = zs; ciz <= ze; ciz++){

				cz1 = (ciz < 0) ? (ciz + num_cells.z) :
				                  ((ciz >= num_cells.z) ? (ciz - num_cells.z) : ciz);

#ifdef PCONSTANT //use the maximum allowed cells to calculate the cell index...
				cell = cz1 * max_numcells.x * max_numcells.y + cy1 *
				       max_numcells.x + cx1;
#else
				cell = cz1 * num_cells.x * num_cells.y + cy1 * num_cells.x + cx1;
#endif

				//number of atoms in the current cell...
				n = num_nonbond[cell];

				//synchronize threads to avoid loop around racing conditions...
				__syncthreads();

				//copy positions into shared memory...
				if(threadIdx.x < n){
					int index = threadIdx.x * total_cells + cell;
					cell_atoms_sh[threadIdx.x] = cell_nonbond[index];

					cell_atoms_pos_sh[threadIdx.x] = float4_to_float3(
					tex1Dfetch(texcrd, unpack_atomid(cell_atoms_sh[threadIdx.x])));
				}

				__syncthreads();

				if(active_thread){
					//iterate through the atoms in the current cell..
					for(i = 0; i < n; i++){

						//use shared memory to compute inter-atomic distances...
						dr.x = __my_fadd(pos.x, -cell_atoms_pos_sh[i].x);
						dr.y = __my_fadd(pos.y, -cell_atoms_pos_sh[i].y);
						dr.z = __my_fadd(pos.z, -cell_atoms_pos_sh[i].z);

						dr.x = __my_add3(dr.x,
						                 -__mycopysignf(boxH.x,__my_fadd(dr.x,-boxH.x)),
						                 -__mycopysignf(boxH.x,__my_fadd(dr.x,boxH.x)));

						dr.y = __my_add3(dr.y,
						                 -__mycopysignf(boxH.y,__my_fadd(dr.y,-boxH.y)),
						                 -__mycopysignf(boxH.y,__my_fadd(dr.y,boxH.y)));

						dr.z = __my_add3(dr.z,
						                 -__mycopysignf(boxH.z,__my_fadd(dr.z,-boxH.z)),
						                 -__mycopysignf(boxH.z,__my_fadd(dr.z,boxH.z)));

						dr.w = __my_add3(__my_fmul(dr.x, dr.x),
						                 __my_fmul(dr.y, dr.y),
						                 __my_fmul(dr.z, dr.z));

						neighbor_id = unpack_atomid(cell_atoms_sh[i]);

						if((dr.w < __my_fmul(cutmaxd, cutmaxd)) &&
						   (self_id != neighbor_id)){

							char isinexcl = 0;

							//Search exclusion bit vector EXCL_BITVEC
							int offset = (neighbor_id - self_id) + exclusion_offset;
							if((offset < 64) && (offset >= 0)){
								isinexcl = (((exclusion_bitvector >> offset) & 1) != 0);
							}

							if(isinexcl == 0){
								nnb++;
								//use shared memory...
								nblistd[nnb * WorkgroupSized + self_id] = cell_atoms_sh[i];
							}
						}
					}// for i<n
				}//if active_thread

			}
		}
	}

	//__syncthreads();

	if(active_thread){
		nblistd[self_id] = nnb;
		//copy this atom's coordinates to old c
		//oordinate array (for update check later)
		roldd[self_id] = pos;
	}
#ifdef PCONSTANT
	if((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)){
		prev_boxLengthd[0] = tex1Dfetch(texbox, 0);
	}
#endif

#ifdef DEBUG_NBLIST
	int gtid = COMPUTE_GLOBAL_THREADID
	debug_nblistd[gtid * NUM_DEBUG_NBLIST + 0] = self_id;
	debug_nblistd[gtid * NUM_DEBUG_NBLIST + 1] = var;
	debug_nblistd[gtid * NUM_DEBUG_NBLIST + 2] = var2;
	debug_nblistd[gtid * NUM_DEBUG_NBLIST + 3] = var3;
#endif

	return;
}

//------------------------------------------------------------------------------
__global__ void nbbuild_excllist(float4* rd,
                                 float4* roldd,
                                 unsigned int* nblistd,
                                 unsigned int* cell_nonbond,
                                 int* num_nonbond,
                                 unsigned long long* excl_bitvecd,
                                 char* excl_bitvec_offsetd,
                                 int* excllistd
#ifdef PCONSTANT
                               , float4* prev_boxLengthd
#endif
#ifdef DEBUG_NBLIST
                               , float* debug_nblistd
#endif
                                                     ){
//------------------------------------------------------------------------------

	__shared__ int cell_atoms_sh[CELL_ATOMS];
	__shared__ float3 cell_atoms_pos_sh[CELL_ATOMS];

#ifdef EXCL_SHMEM
	__shared__ int excl_shmem[CELL_ATOMS * EXCL_COUNT_PERATOM];
#endif

	//int* cell_atoms_sh = (int *) temp_array;
	//float3* cell_atoms_pos_sh = (float3 *) &cell_atoms_sh[MAX_NBCELL];

	int self_id = -1;
	int neighbor_id;

	float4 pos={0.0f, 0.0f, 0.0f, 0.0f};
	float4 dr;
#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
	int4 num_cells = tex1Dfetch(texnumcells, 0);

	int total_cells = max_numcells.w; //num_cells.w;
#endif
	//float4 neighbor_pos;
	char cx;
	char cy;
	char cz;

	char cx1;
	char cy1;
	char cz1;

	char cix;
	char ciy;
	char ciz;

	//unsigned char xc, yc, zc;
	short int n;
	short int nnb = 0;
	short int i;//neighbor_atomid,

	char xs;
	char xe;
	char ys;
	char ye;
	char zs;
	char ze;

	unsigned short int cell;

	cx = blockIdx.x % num_cells.x;
	cy = blockIdx.x / num_cells.x;
	cz = blockIdx.y;

#ifdef PCONSTANT //use the maximum num cells to calculate the cell index...
	cell = cz *(max_numcells.x * max_numcells.y) + cy * max_numcells.x + cx;
#else
	cell = cz * (num_cells.x * num_cells.y) + cy * num_cells.x + cx;
#endif

	//Determine start and end of the search cells
	xs = (cx == 0) ? (cx - 2) : (cx - 1);
	ys = (cy == 0) ? (cy - 2) : (cy - 1);
	zs = (cz == 0) ? (cz - 2) : (cz - 1);

	xe =(cx == num_cells.x - 2) ? (cx + 2) : (cx + 1);
	ye =(cy == num_cells.y - 2) ? (cy + 2) : (cy + 1);
	ze =(cz == num_cells.z - 2) ? (cz + 2) : (cz + 1);

	unsigned char active_thread = (threadIdx.x < num_nonbond[cell]);

	unsigned char num_excl = 0;
	int excl_first = -1;
	int excl_last = -1;

	//read the position of the current atom here....
	if(active_thread){
		self_id = unpack_atomid(cell_nonbond[threadIdx.x * total_cells + cell]);
		pos = rd[self_id];

		if(num_excl = excllistd[self_id]){
			excl_first = excllistd[1 * WorkgroupSized + self_id];
			excl_last = excllistd[num_excl * WorkgroupSized + self_id];
		}
#ifdef EXCL_SHMEM
		//copy exclusion list to shared memory...
		for(i = 0; i < num_excl; i++){
			int pos_l = threadIdx.x + i * CELL_ATOMS;
			int pos_r = self_id + i * WorkgroupSized;
			excl_shmem[pos_l] = excllistd[pos_r];
		}
#endif
	}

	//finished reading self position...
	for(cix = xs; cix <= xe; cix++){

		cx1 = (cix < 0) ? (cix + num_cells.x) :
		                  ((cix >= num_cells.x) ? (cix - num_cells.x) : cix);

		for(ciy = ys; ciy <= ye; ciy++){

			cy1 = (ciy < 0) ? (ciy + num_cells.y) :
			                  ((ciy >= num_cells.y) ? (ciy - num_cells.y) : ciy);

			for(ciz = zs; ciz <= ze; ciz++){

				cz1 = (ciz < 0) ? (ciz + num_cells.z) :
				                  ((ciz >= num_cells.z) ? (ciz - num_cells.z) : ciz);

#ifdef PCONSTANT //use the maximum allowed cells to calculate the cell index...
				cell = cz1 * max_numcells.x * max_numcells.y +
				       cy1 * max_numcells.x + cx1;
#else
				cell = cz1 * num_cells.x * num_cells.y + cy1 * num_cells.x + cx1;
#endif

				//number of atoms in the current cell...
				n = num_nonbond[cell];

				//synchronize threads to avoid loop around racing conditions...
				__syncthreads();

				//copy positions into shared memory...
				if(threadIdx.x < n){
					int index = threadIdx.x * total_cells + cell;
					cell_atoms_sh[threadIdx.x] = cell_nonbond[index];

					cell_atoms_pos_sh[threadIdx.x] = float4_to_float3(
					  tex1Dfetch(texcrd, unpack_atomid(cell_atoms_sh[threadIdx.x])));
				}

				__syncthreads();

				if(active_thread){
					//iterate through the atoms in the current cell..
					for(i = 0; i < n; i++){

						//use shared memory to compute inter-atomic distances...
						dr.x = __my_fadd(pos.x, -cell_atoms_pos_sh[i].x);
						dr.y = __my_fadd(pos.y, -cell_atoms_pos_sh[i].y);
						dr.z = __my_fadd(pos.z, -cell_atoms_pos_sh[i].z);

						dr.x = __my_add3(dr.x,
						                 -__mycopysignf(boxH.x,__my_fadd(dr.x,-boxH.x)),
						                 -__mycopysignf(boxH.x,__my_fadd(dr.x,boxH.x)));

						dr.y = __my_add3(dr.y,
						                 -__mycopysignf(boxH.y,__my_fadd(dr.y,-boxH.y)),
						                 -__mycopysignf(boxH.y,__my_fadd(dr.y,boxH.y)));

						dr.z = __my_add3(dr.z,
						                 -__mycopysignf(boxH.z,__my_fadd(dr.z,-boxH.z)),
						                 -__mycopysignf(boxH.z,__my_fadd(dr.z,boxH.z)));

						dr.w = __my_add3(__my_fmul(dr.x, dr.x),
						                 __my_fmul(dr.y, dr.y),
						                 __my_fmul(dr.z, dr.z));

						neighbor_id = unpack_atomid(cell_atoms_sh[i]);

						if((dr.w < __my_fmul(cutmaxd, cutmaxd)) &&
						   (self_id != neighbor_id)){

							char isinexcl = 0;

							//check to see if it is in the range
							//of the ordered exclusion atoms..
							isinexcl = ((num_excl) &&
							            (neighbor_id >= excl_first) &&
							            (neighbor_id <= excl_last));
							if((isinexcl) &&
							   (num_excl > 2) &&
							   (neighbor_id != excl_first) &&
							   (neighbor_id !=excl_last)){
								//first and last atoms from the list are already checked..
								//just check the remaining
								isinexcl = Search_excl_binary(excllistd,
								                              self_id,
								                              neighbor_id,
								                              2,
								                              num_excl-1);
							}

							if(isinexcl == 0){
								nnb++;
								//use shared memory...
								nblistd[nnb * WorkgroupSized + self_id] = cell_atoms_sh[i];
							}
						}
					}// for i<n
				}//if active_thread
			}
		}
	}

	//__syncthreads();

	if(active_thread){
		nblistd[self_id] = nnb;
		//copy this atom's coordinates to old
		//coordinate array (for update check later)
		roldd[self_id] = pos;
	}
#ifdef PCONSTANT
	if((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)){
		prev_boxLengthd[0] = tex1Dfetch(texbox, 0);
	}
#endif

#ifdef DEBUG_NBLIST
	int gtid = COMPUTE_GLOBAL_THREADID
	debug_nblistd[gtid * NUM_DEBUG_NBLIST + 0] = self_id;
	debug_nblistd[gtid * NUM_DEBUG_NBLIST + 1] = var;
	debug_nblistd[gtid * NUM_DEBUG_NBLIST + 2] = var2;
	debug_nblistd[gtid * NUM_DEBUG_NBLIST + 3] = var3;
#endif

	return;
}

//------------------------------------------------------------------------------

#ifdef PME_CALC
//------------------------------------------------------------------------------
__global__ void CheckLatticeNum(float4* r4d, int* numL){
//------------------------------------------------------------------------------

	int i;
	int j;
	int k;

	int ri;
	int rj;
	int rk;

	int cell;
	int cell2;
	int cell3;

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	float3 scaled_pos;
#ifdef PCONSTANT
	float4 frac = tex1Dfetch(texfrac, 0);
#endif

	if(gtid < natomd){

		scaled_pos.x = update_coords(__my_fmul(frac.x, r4d[gtid].x), FFTX);
		scaled_pos.y = update_coords(__my_fmul(frac.y, r4d[gtid].y), FFTY);
		scaled_pos.z = update_coords(__my_fmul(frac.z, r4d[gtid].z), FFTZ);
		//scaled_pos.w = r4d[gtid].w;

		//convert to integers..
		ri = floor(scaled_pos.x);
		rj = floor(scaled_pos.y);
		rk = floor(scaled_pos.z);

		for(k = rk - 3; k <= rk; k++){
			cell3 = update_coords(k, FFTZ) * FFTX * FFTY;

			for(j = rj - 3; j <= rj; j++){
				cell2 = update_coords(j, FFTY) * FFTX + cell3;

				for(i = ri - 3; i <= ri; i++){

					cell = update_coords(i, FFTX) + cell2;

					atomicAdd(&numL[cell], 1);

				}//for (i = ri-3; i <= ri; i++){
			}//for (j = rj-3; j <= rj; j++){
		}//for (k = rk-3; k <= rk; k++){
	}//if (gtid < natomd){

	return;
}

//------------------------------------------------------------------------------
//////////////////////////updating the cell list/////////////////////////////
//Iterates through the global list of atoms for each cell..
//Called only at the start of the simulation to build the list.
__global__ void LatticeBuild(float4* r4d,
                             float4* r4d_scaled,
                             int* cellL,
                             int* numL
#ifdef DEBUG_PME
                           , float* pme_debug_d
#endif
                                               ){
//------------------------------------------------------------------------------
	//float4 r;
	//float dx, dy, dz;
	int i;
	int j;
	int k;
	int n;

	int ri;
	int rj;
	int rk;

	int cell;
	int cell2;
	int cell3;

	//float dx, dy, dz;
	unsigned int gtid = COMPUTE_GLOBAL_THREADID
#ifdef PCONSTANT
	float4 frac = tex1Dfetch(texfrac, 0);
#endif

	if (gtid < natomd){

		r4d_scaled[gtid].x = update_coords(__my_fmul(frac.x, r4d[gtid].x), FFTX);
		r4d_scaled[gtid].y = update_coords(__my_fmul(frac.y, r4d[gtid].y), FFTY);
		r4d_scaled[gtid].z = update_coords(__my_fmul(frac.z, r4d[gtid].z), FFTZ);
		r4d_scaled[gtid].w = r4d[gtid].w;

		//convert to integers..
		ri = floor(r4d_scaled[gtid].x);
		rj = floor(r4d_scaled[gtid].y);
		rk = floor(r4d_scaled[gtid].z);

		for(k = rk - 3; k <= rk; k++){
			cell3 = update_coords(k, FFTZ) * FFTX * FFTY;

			for(j = rj - 3; j <= rj; j++){
				cell2 = update_coords(j, FFTY) * FFTX + cell3;

				for(i = ri - 3; i <= ri; i++){

					cell = update_coords(i, FFTX) + cell2;

					n = atomicAdd(&numL[cell], 1);

					cellL[n * TOTAL_PME_LATTICES + cell] = gtid;

				}//for (i = ri-3; i <= ri; i++){
			}//for (j = rj-3; j <= rj; j++){
		}//for (k = rk-3; k <= rk; k++){
	}

	return;
}

//------------------------------------------------------------------------------
__global__ void LatticeUpdate(float4* r4d_scaled,
                              int4* disp,
                              int* cellL,
                              int* numL
#ifdef DEBUG_PME
                            , float* pme_debug_d
#endif
                                                ){
//------------------------------------------------------------------------------

	int x;
	int y;
	int z;
	int n;
	int i;
	int lattice;
	int new_lattice;
	int atomid;

	short int update_num;

	float dx;
	float dy;
	float dz;

	short int nlattice_x;
	short int nlattice_y;
	short int nlattice_z;

	char has_updated;
	//int3 displacement;
	float3 scaled_pos;

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

	lattice = gtid;

	float fftxh = FFTXH;
	float fftyh = FFTYH;
	float fftzh = FFTZH;

	n = numL[lattice];

	z = lattice/(FFTX * FFTY);
	y = (lattice%(FFTX * FFTY))/(FFTX);
	x = (lattice%(FFTX * FFTY))%(FFTX);

	////if the compiler doesnot optimize the above calculations use the following
	//z = FFTX * FFTY;
	//y = cell % z;
	//x = y % FFTX;
	//y = y / FFTX;
	//z = cell / z;

	for(i = 0; i < n; i++){

		atomid = cellL[i * TOTAL_PME_LATTICES + lattice];

		//if ((disp[atomid].x!=0)||(disp[atomid].y!=0)||(disp[atomid].z!=0)){

		scaled_pos = float4_to_float3(r4d_scaled[atomid]);
		//displacement = int4_to_int3(disp[atomid]);

		dx = nearest_image(scaled_pos.x - x, fftxh); //fftxh //FFTXH
		dy = nearest_image(scaled_pos.y - y, fftyh); //fftyh //FFTYH
		dz = nearest_image(scaled_pos.z - z, fftzh); //fftzh //FFTZH

		has_updated = 0;
		//do in parallel...

		//if greater than 8 units or less than
		//-4 translate the entire block of cells
		if(((dx >= 2 * 4) || (dy >= 2 * 4) ||
		    (dz >= 2 * 4) || (dx <= -4) ||
		    (dy <= -4) || (dz <= -4)) &&
		   (!has_updated)){

			nlattice_x = x + disp[atomid].x; //displacement.x;
			nlattice_y = y + disp[atomid].y; //displacement.y;
			nlattice_z = z + disp[atomid].z; //displacement.z;
			has_updated = 1;
		}

		//if the above condition is not satisfied one or a few of the following is true...

		//if greater than 4 but less than 8 translate only those affected..
		if((dx > 4) && (!has_updated)){
			nlattice_x = x + 4;
			nlattice_y = y + disp[atomid].y; //displacement.y;
			nlattice_z = z + disp[atomid].z; //displacement.z;
			has_updated = 1;
		}

		//if less than 0 but greater than -4 translate only those affected..
		if((dx < 0) && (!has_updated)){
			nlattice_x = x - 4;
			nlattice_y = y + disp[atomid].y; //displacement.y;
			nlattice_z = z + disp[atomid].z; //displacement.z;
			has_updated = 1;
		}

		//if greater than 4 but less than 8 translate only those affected..
		if((dy > 4) && (!has_updated)){
			nlattice_x = x;
			nlattice_y = y + 4;
			nlattice_z = z + disp[atomid].z; //displacement.z;
			has_updated = 1;
		}

		//if less than 0 but greater than -4 translate only those affected..
		if((dy < 0) && (!has_updated)){
			nlattice_x = x;
			nlattice_y = y - 4;
			nlattice_z = z + disp[atomid].z; //displacement.z;
			has_updated = 1;
		}

		//if greater than 4 but less than 8 translate only those affected..
		if((dz > 4) && (!has_updated)){
			nlattice_x = x;
			nlattice_y = y;
			nlattice_z = z + 4;
			has_updated = 1;
		}

		//if less than 0 but greater than -4 translate only those affected..
		if((dz < 0) && (!has_updated)){
			nlattice_x = x;
			nlattice_y = y;
			nlattice_z = z - 4;
			has_updated = 1;
		}

		if(has_updated){
			new_lattice = update_coords(nlattice_z, FFTZ) * FFTX * FFTY +
							      update_coords(nlattice_y, FFTY) * FFTX +
							      update_coords(nlattice_x, FFTX);

			update_num = (short int)atomicAdd(&numL[new_lattice], 1);

			cellL[update_num * TOTAL_PME_LATTICES + new_lattice] = atomid;
		}//if (has_updated){
	}//for (i=0; i<n; i++){

	return;
}

//------------------------------------------------------------------------------
__global__ void BCMultiply(cufftComplex* Qd
#ifdef DEBUG_PME
                         , float* pme_debug_d
#endif
                                             ){
//------------------------------------------------------------------------------

	unsigned int gtid = COMPUTE_GLOBAL_THREADID
	int mz;
	int my;
	int mx;
	int cell;

	//int num_threads = gridDim.x*gridDim.y*blockDim.x;
	//cufftComplex val = {1, 0};
	float val = 0.0f;
#ifdef PCONSTANT
	float3 box = float4_to_float3(tex1Dfetch(texbox, 0));
#endif

	float PIxV = PI * (TOTAL_PME_LATTICES);//INVERSE_CUFFT does not divide by V
	//determine the indices..
	cell = gtid;
	//for (cell=gtid; cell<TOTAL_PME_LATTICES; cell+=num_threads){

	mz = cell / (FFTX * FFTY);
	my = (cell % (FFTX * FFTY)) / FFTX;
	mx = (cell % (FFTX * FFTY)) % FFTX;

	// //if the compiler doesnot optimize the above calculations use the following
	//mz = FFTX * FFTY;
	//my = cell % mz;
	//mx = my % FFTX;
	//my = my / FFTX;
	//mz = cell / mz;

	//multiply by B and C
	//val =(b1d[mx]*(~b1d[mx]))*(b2d[my]*(~b2d[my]))*(b3d[mz]*(~b3d[mz]));
	//val = (b1d[mx].x)*(b1d[mx].x)*
	//(b2d[my].x)*(b2d[my].x)*(b3d[mz].x)*(b3d[mz].x);

	val = sqr_b1d[mx] * sqr_b2d[my] * sqr_b3d[mz];
	val = val * c1d[mx] * c2d[my] * c3d[mz];

	mx = (mx > FFTXH) ? (mx - FFTX) : mx;
	my = (my > FFTYH) ? (my - FFTY) : my;
	mz = (mz > FFTZH) ? (mz - FFTZ) : mz;

	val = (cell == 0) ? 0.0f :
	                    (val / (((float)(mx * mx)) / (box.x * box.x) +
	                     ((float)(my * my)) / (box.y * box.y) +
	                     ((float)(mz * mz))  /(box.z * box.z)));

	Qd[cell] = Qd[cell] * (val / PIxV);

	return;
}
#endif //#ifdef PME_CALC

//------------------------------------------------------------------------------
//initializes B and C for PME and VanderWaals and Electostatic force tables...
void InitDeviceConstants(){
//------------------------------------------------------------------------------
#ifdef PME_CALC
	int m;
	int k;
	int l;
	//cufftComplex *b1, *b2, *b3;
	float* sqr_b1;
	float* sqr_b2;
	float* sqr_b3;
	float* c1;
	float* c2;
	float* c3;

	//b1 = (cufftComplex *) malloc(fftx*sizeof(cufftComplex));
	//b2 = (cufftComplex *) malloc(ffty*sizeof(cufftComplex));
	//b3 = (cufftComplex *) malloc(fftz*sizeof(cufftComplex));

	cufftComplex temp_val;

	sqr_b1 = (float*)malloc(fftx * sizeof(float));
	sqr_b2 = (float*)malloc(ffty * sizeof(float));
	sqr_b3 = (float*)malloc(fftz * sizeof(float));

	c1 = (float*)malloc(fftx * sizeof(float));
	c2 = (float*)malloc(ffty * sizeof(float));
	c3 = (float*)malloc(fftz * sizeof(float));

	cufftComplex denom;
	cufftComplex im = {0, 1};

	int K[3] = {fftx, ffty, fftz};
	float L[3] = {Region.x, Region.y, Region.z};

	float mk;
	float mkw;

	for(l = 0; l < 3; l++){
		for(m = 0; m < K[l]; m++){
			denom = make_cufftComplex(0.0f, 0.0f);
			mk = ((float)m) / K[l];
			//mkw = (m > K[l]/2)?(mk-1):mk;
			mkw = (float)((m > K[l] / 2) ? (m - K[l]) : m);

			mkw = mkw / L[l];

			for(k = 0; k < 3; k++){
				denom = denom + exp(make_cufftComplex(0.0f, 2.0f * PI * mk * k)) *
				        (float)M4(k + 1);
			}

			if(l == 0){
				temp_val = exp(im * 2.0 * (float)PI * mk) / denom; //b1[m]
				sqr_b1[m] = (float)(temp_val * (~temp_val)).x;
				//dont forget PIxV * m^2 in denominator while FFT, or MultiplyBC
				c1[m] = (float)exp(-PI * PI * mkw * mkw / (KAPPa * KAPPa));
			}

			if(l == 1){
				temp_val = exp(im * 2.0 * (float)PI * mk) / denom; //b2[m]
				sqr_b2[m] = (float)(temp_val * (~temp_val)).x;
				//dont forget PIxV * m^2 in denominator while FFT, or MultiplyBC
				c2[m] = (float)exp(-PI * PI * mkw * mkw / (KAPPa * KAPPa));
			}

			if(l == 2){
				temp_val  = exp(im * 2.0 * (float)PI * mk) / denom; //b3[m]
				sqr_b3[m] = (float)(temp_val * (~temp_val)).x;
				//dont forget PIxV * m^2 in denominator while FFT, or MultiplyBC
				c3[m] = (float)exp(-PI * PI * mkw * mkw / (KAPPa * KAPPa));
			}
		}
	}

	//copy the PME constants B and C
	if((fftx > MAXFFT_X) || (ffty > MAXFFT_Y) || (fftz > MAXFFT_Z)){
		printf("Error! Maximum allocated FFT dimension(s) less than"
		       " PME gridsize\n Recompile the code with a larger FFT dimension...");
		exit(0);
	}

	cudaMemcpyToSymbol(sqr_b1d, sqr_b1, fftx * sizeof(float));
	cudaMemcpyToSymbol(sqr_b2d, sqr_b2, ffty * sizeof(float));
	cudaMemcpyToSymbol(sqr_b3d, sqr_b3, fftz * sizeof(float));

	cudaMemcpyToSymbol(c1d, c1, fftx * sizeof(float));
	cudaMemcpyToSymbol(c2d, c2, ffty * sizeof(float));
	cudaMemcpyToSymbol(c3d, c3, fftz * sizeof(float));

	//finished updating the constants B and C now compute spline coefficients..

	float M4H[NUM_SPLINE_SAMPLES + 1];
	float dM4H[NUM_SPLINE_SAMPLES + 1];
	int i;
	float arg;
	for(i = 0; i < NUM_SPLINE_SAMPLES + 1; i++){
		arg = (float)i * 2.0f / NUM_SPLINE_SAMPLES;
		//only the values from 0 to 2 are computed
		//and stored as the spline function is symmetric...
		M4H[i] = M4(arg);
		dM4H[i] = M3(arg) - M3(arg - 1);
	}

	cudaMemcpy(M4d, M4H, sizeof(float) * (NUM_SPLINE_SAMPLES + 1),
	           cudaMemcpyHostToDevice);

	cudaMemcpy(dM4d, dM4H, sizeof(float) * (NUM_SPLINE_SAMPLES + 1),
	           cudaMemcpyHostToDevice);

	cudaBindTexture(0, texM4, M4d, sizeof(float) * (NUM_SPLINE_SAMPLES + 1));
	cudaBindTexture(0, texdM4, dM4d, sizeof(float) * (NUM_SPLINE_SAMPLES + 1));

	#endif //PME_CALC

	checkCUDAError("InitDeviceConstants");
	return;
}

