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

#ifdef PME_CALC

//==============================================================================
///Function to spread charge and update the neighbor list of the cell..
__global__ void ChargeSpread_small(cufftComplex* Qd,
                                   float4* r4d_scaled,
                                   int* cellL,
                                   int* numL
#ifdef DEBUG_PME
                                 , float* pme_debug_d
#endif
                                                     ){
//==============================================================================
	//total number of threads is equal to fftx*ffty*fftz for Chargespread...
	int x;
	int y;
	int z;
	int n;
	int atomid;
	int cell;
	int i;

	float charge_at_cell = 0.0f;
	float dx;
	float dy;
	float dz;
	float dw;

	unsigned int gtid = COMPUTE_GLOBAL_THREADID
	float4 pos;

	/*
	__shared__ float M4d_sh_x[NUM_SPLINE_SAMPLES + 1];
	__shared__ float M4d_sh_y[NUM_SPLINE_SAMPLES + 1];
	__shared__ float M4d_sh_z[NUM_SPLINE_SAMPLES + 1];

	for (i=threadIdx.x; i<NUM_SPLINE_SAMPLES + 1; i+=blockDim.x){

	//copy global spline values to shared memory...
	M4d_sh_x[i] = M4d_sh_y[i] = M4d_sh_z[i] = tex1Dfetch(texM4, i); //M4d_c[i];
	}

	__syncthreads();
	*/

	cell = gtid;
	n = numL[cell];

	if(n > 0){
		atomid  = cellL[ cell];
	}

	z = cell / (FFTX * FFTY);
	y = (cell % (FFTX * FFTY)) / (FFTX);
	x = (cell % (FFTX * FFTY)) % (FFTX);

	// If n is a power of 2, (i/n) is equivalent to
	//(i>>log2(n)) and (i%n) is equivalent to (i&(n-1));
	//int FFTval0 =  FFTX * FFTY;
	//float FFTval1 = 1.0f * FFTval0;
	//int FFTvalue = log2(FFTval1);
	//z = cell>>FFTvalue;
	//y = (cell&(FFTval0-1))>>(int)log2(1.0f*FFTX);
	//x = (cell&(FFTval0-1))&(FFTX-1);

	for(i=1; i<=n; i++){

		//r4d_scaled[atomid]; //tex1Dfetch(texsclcrd, atomid);
		pos = tex1Dfetch(texsclcrd, atomid);
		dx = pos.x - x;
		dy = pos.y - y;
		dz = pos.z - z;
		dw = pos.w;

		dx = nearest_image(dx, FFTXH);
		dy = nearest_image(dy, FFTYH);
		dz = nearest_image(dz, FFTZH);

		if((dx >= 0) && (dx <= 4) && (dy >= 0) &&
		   (dy <= 4) && (dz >= 0) && (dz <= 4)){

			/*
			charge_at_cell += dw	*M4d_sh_x[spline_index(dx)]
			*M4d_sh_y[spline_index(dy)]
			*M4d_sh_z[spline_index(dz)];

			*/

			charge_at_cell += dw * tex1Dfetch(texM4, spline_index(dx)) *
			                  tex1Dfetch(texM4, spline_index(dy)) *
			                  tex1Dfetch(texM4, spline_index(dz));

			atomid = cellL[i * TOTAL_PME_LATTICES + cell];
		}
		else{
			//copy an element from the end of the list to the current position
			//and setup to be processed in the next iteration...
			n--;
			i--;
			if((n > 0) && (i < n)){
				atomid = cellL[i * TOTAL_PME_LATTICES + cell] =
				  cellL[n * TOTAL_PME_LATTICES + cell];
			}
		}
	}

	Qd[cell] = make_cufftComplex(charge_at_cell, 0.0f);
	numL[cell] = n;
	//}
	//The charge matrix Qd is now updated..
	return;
}

//==============================================================================
///Function to spread charge and update the neighbor list of the cell..
__global__ void ChargeSpread_medium(cufftComplex* Qd,
                                    float4* r4d_scaled,
                                    int* cellL,
                                    int* numL
#ifdef DEBUG_PME
                                  , float* pme_debug_d
#endif
                                                      ){
//==============================================================================
	//total number of threads is equal to fftx*ffty*fftz for Chargespread...
	int x;
	int y;
	int z;
	int n;
	int atomid;
	int cell;
	int i;

	float charge_at_cell = 0.0f;
	float dx;
	float dy;
	float dz;
	float dw;

	unsigned int gtid = COMPUTE_GLOBAL_THREADID
	float4 pos;

	__shared__ float M4d_sh_x[NUM_SPLINE_SAMPLES + 1];
	__shared__ float M4d_sh_y[NUM_SPLINE_SAMPLES + 1];
	__shared__ float M4d_sh_z[NUM_SPLINE_SAMPLES + 1];

	for(i = threadIdx.x; i < NUM_SPLINE_SAMPLES + 1; i += blockDim.x){

		//copy global spline values to shared memory...
		M4d_sh_z[i] = tex1Dfetch(texM4, i); //M4d_c[i];
		M4d_sh_y[i] = M4d_sh_z[i];
		M4d_sh_x[i] = M4d_sh_y[i];
	}

	__syncthreads();

	cell = gtid;

	n = numL[cell];

	if(n > 0){
		atomid = cellL[0 * TOTAL_PME_LATTICES + cell];
	}

	z = cell / (FFTX * FFTY);
	y = (cell % (FFTX * FFTY)) / (FFTX);
	x = (cell % (FFTX * FFTY)) % (FFTX);
	//If n is a power of 2, (i/n) is equivalent to
	//(i>>log2(n)) and (i%n) is equivalent to (i&(n-1));
	//int FFTval0 =  FFTX * FFTY;
	//float FFTval1 = 1.0f * FFTval0;
	//int FFTvalue = log2(FFTval1);
	//z = cell>>FFTvalue;
	//y = (cell&(FFTval0-1))>>(int)log2(1.0f*FFTX);
	//x = (cell&(FFTval0-1))&(FFTX-1);

	for(i = 1; i <= n; i++){

		//r4d_scaled[atomid]; //tex1Dfetch(texsclcrd, atomid);
		pos = tex1Dfetch(texsclcrd, atomid);
		dx = pos.x - x;
		dy = pos.y - y;
		dz = pos.z - z;
		dw = pos.w;

		dx = nearest_image(dx, FFTXH);
		dy = nearest_image(dy, FFTYH);
		dz = nearest_image(dz, FFTZH);

		if((dx >= 0) && (dx <= 4) && (dy >= 0) &&
		   (dy <= 4) && (dz >= 0) && (dz <= 4)){


			charge_at_cell += dw * M4d_sh_x[spline_index(dx)] *
			                  M4d_sh_y[spline_index(dy)] *
			                  M4d_sh_z[spline_index(dz)];

			/*

			charge_at_cell += dw    *tex1Dfetch(texM4, spline_index(dx))
			*tex1Dfetch(texM4, spline_index(dy))
			*tex1Dfetch(texM4, spline_index(dz));
			*/

			atomid = cellL[i * TOTAL_PME_LATTICES + cell];
		}
 		else{
			//copy an element from the end of the list to the current position
			//and setup to be processed in the next iteration...
			n--;
			i--;
			if((n > 0) && (i < n)){
				atomid = cellL[i * TOTAL_PME_LATTICES + cell] =
				  cellL[n * TOTAL_PME_LATTICES + cell];
			}
		}
	}

	Qd[cell] = make_cufftComplex(charge_at_cell, 0.0f);
	numL[cell] = n;
	//}
	//The charge matrix Qd is now updated..
	return;
}

/*
//==============================================================================
///Function to spread charge and update the neighbor list of the cell..
__global__ void ChargeSpread(cufftComplex *Qd, float4 *r4d_scaled,
int *cellL, int *numL
#ifdef DEBUG_PME
,float *pme_debug_d
#endif
){
//==============================================================================
//total number of threads is different from WorkgroupSized for PME kernels...
int x, y, z, n, atomid, cell, i;
short int new_num;
float charge_at_cell, dx, dy, dz;
// Here gtid contains the x,y information...
unsigned int gtid = COMPUTE_GLOBAL_THREADID
float4 pos;

__shared__ float M4d_sh[NUM_SPLINE_SAMPLES + 1];

for (i=threadIdx.x; i<NUM_SPLINE_SAMPLES + 1; i+=blockDim.x){

	//copy global spline values to shared memory...
	M4d_sh[i] = tex1Dfetch(texM4, i); //M4d_c[i];
}

__syncthreads();

cell = gtid;

	n = numL[cell];

	z = cell/(FFTX * FFTY);
	y = (cell%(FFTX * FFTY))/(FFTX);
	x = (cell%(FFTX * FFTY))%(FFTX);

	charge_at_cell=0.0f;

	new_num = 0;
	for (i=0; i<n; i++){

		atomid = cellL[i*TOTAL_PME_LATTICES + cell];

		pos = r4d_scaled[atomid]; //tex1Dfetch(texsclcrd, atomid);

			dx = nearest_image(pos.x - x, FFTXH);
	    	dy = nearest_image(pos.y - y, FFTYH);
	    	dz = nearest_image(pos.z - z, FFTZH);

		if ((dx>=0)&&(dx<=4)&&(dy>=0)&&(dy<=4)&&(dz>=0)&&(dz<=4)){

			charge_at_cell += pos.w	*M4d_sh[spline_index(dx)]
						*M4d_sh[spline_index(dy)]
						*M4d_sh[spline_index(dz)];

		if (new_num!=i) cellL[new_num*TOTAL_PME_LATTICES + cell] = atomid;

		new_num++;
		}
	}

	Qd[cell] = make_cufftComplex(charge_at_cell, 0.0f);
	numL[cell] = new_num;

//The charge matrix Qd is now updated..
return;
}


#ifdef CHARGESPREAD_ATOMIC
//==============================================================================
///Function to spread charge and update the neighbor list of the cell..
__global__ void ChargeSpread_Atomic(cufftComplex *Qd, float4 *r4d){
//==============================================================================

int cell, cell3, cell2, i;
float addcharge, dx, dy, dz;
float x_scaled, y_scaled, z_scaled;
int rx, ry, rz, xi, yi, zi;
// Here gtid contains the x,y information...
unsigned int gtid = COMPUTE_GLOBAL_THREADID
float4 pos;

#ifdef PCONSTANT
	float4 frac = tex1Dfetch(texfrac, 0);
#endif

__shared__ float M4d_sh[NUM_SPLINE_SAMPLES + 1];

for (i=threadIdx.x; i<NUM_SPLINE_SAMPLES + 1; i+=blockDim.x){

	//copy global spline values to shared memory...
	M4d_sh[i] = tex1Dfetch(texM4, i); //M4d_c[i];
}

__syncthreads();

if (gtid<natomd){
	pos = r4d[gtid];

	x_scaled = __my_fmul(frac.x , pos.x);
	y_scaled = __my_fmul(frac.y , pos.y);
	z_scaled = __my_fmul(frac.z , pos.z);

	rx = floor(x_scaled);
	ry = floor(y_scaled);
	rz = floor(z_scaled);

	for(zi=rz - 3; zi<=rz; zi++){

		cell3 = update_coords(zi, FFTZ) * FFTX * FFTY;
		dz = nearest_image(z_scaled - zi, FFTZH);

		for(yi=ry - 3; yi<=ry; yi++){

			cell2 = update_coords(yi, FFTY) * FFTX + cell3;
			dy = nearest_image(y_scaled - yi, FFTYH);

			for(xi=rx - 3; xi<=rx; xi++){

				cell =	update_coords(xi, FFTX) + cell2;
				dx = nearest_image(x_scaled - xi, FFTXH);

				addcharge = pos.w*M4d_sh[spline_index(dx)]
						*M4d_sh[spline_index(dy)]
						*M4d_sh[spline_index(dz)];

				atomicAdd(&Qd[cell].x, addcharge);
			}
		}
	}
}//if (gtid<natomd)
//The charge matrix Qd is now updated..
return;
}
#endif		//CHARGESPREAD_ATOMIC
*/


//==============================================================================
__global__ void PMEForce_medium(cufftComplex* Qd,
                                float4* f4d,
                                float* ePMEd
#ifdef PCONSTANT
                              , float4* viriald
#endif
#ifdef DEBUG_PME
                              , float* pme_debug_d
#endif
                                                  ){
//==============================================================================

	int i;
	int j;
	int k;
	int cell;//, t1;

	float forcex = 0.0f;
	float forcey = 0.0f;
	float forcez = 0.0f;
	float enPME = 0.0f;

	float x1;
	float y1;
	float z1;
	float x;
	float y;
	float z;
	float w;

	//float myMass=1.0f;
	char s1;
	char s2;
	char s3;//signs of the odd function... derivtive of M4...
	unsigned int gtid = COMPUTE_GLOBAL_THREADID

#ifdef PCONSTANT
	float4 frac = tex1Dfetch(texfrac, 0);
	float4 RBSRat = tex1Dfetch(texRBSRat, 0);
#endif

	int cell2;
	int cell3;

	float forcek1;
	float forcek2;
	float forcek3;
	float forcej1;
	float forcej2;
	float forcej3;
	float enerk;
	float enerj;
	float var; //forcei1, forcei2, forcei3, eneri,

	// __shared__ float M4d_sh[NUM_SPLINE_SAMPLES + 1];
	__shared__ float dM4d_sh[NUM_SPLINE_SAMPLES + 1];

	for(i = threadIdx.x; i < NUM_SPLINE_SAMPLES + 1; i += blockDim.x){
		//copy global spline values to shared memory...
		dM4d_sh[i] = tex1Dfetch(texdM4, i); //dM4d_c[i];
		// M4d_sh[i] = tex1Dfetch(texM4, i); //M4d_c[i];
	}

	__syncthreads();

	float4 pos = tex1Dfetch(texsclcrd, gtid);
	x = pos.x;
	y = pos.y;
	z = pos.z;
	w = pos.w;

	//read precomputed value of "fft(B.C.ifft(Q)) = convolution(B.C, Q)"
	//copy only x component, because forcex,
	//forcey and forcez as well as the emPME are all real
	float FBCQ = 0.0f;

	if(gtid < natomd){
		for (k = ceil(z - 4); k < z; k++){

			z1 = z - k;
			s3 = (z1 > 2) ? -1 : 1;
			cell3 = update_coords(k, FFTZ) * FFTX * FFTY;

			var =  tex1Dfetch(texM4, spline_index(z1)); //M4d_sh[spline_index(z1)];
			forcek1 = var;
			forcek2 = var;
			forcek3 = s3 * dM4d_sh[spline_index(z1)];
			enerk = var;

			for(j = ceil(y - 4); j < y; j++){

				y1 = y - j;
				cell2 = update_coords(j, FFTY) * FFTX + cell3;
				s2 = (y1 > 2) ? -1 : 1;
				var = tex1Dfetch(texM4, spline_index(y1)); //M4d_sh[spline_index(y1)];
				forcej1 = var * forcek1;
				forcej2 = s2 * dM4d_sh[spline_index(y1)] * forcek2;
				forcej3 = var * forcek3;
				enerj = var * enerk;

				//forcei1 = forcei2 = forcei3 = 0;
				/*
				for (i = ceil(x-4); i < x; i++){

				x1 = x - i;
				s1 = (x1>2)?-1:1;
				cell = update_coords(i, FFTX) + cell2;

				//copy only x component, because forcex, forcey
				//and forcez as well as the emPME are all real
				FBCQ = Qd[cell].x;

				var = M4d_sh[spline_index(x1)];
				forcex += s1 *FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				}//for (k = ceil(z-4); k < z; k++){
				*/

				//LOOP UNROLL
				i = ceil(x - 4);
				x1 = x - i;
				s1 = (x1 > 2) ? -1 : 1;
				cell = update_coords(i, FFTX) + cell2;
				FBCQ = Qd[cell].x;

				var = tex1Dfetch(texM4, spline_index(x1)); //M4d_sh[spline_index(x1)];
				forcex += s1 * FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				i = i + 1;
				x1 = x - i;
				s1 = (x1 > 2) ? -1 : 1;
				cell = update_coords(i, FFTX) + cell2;
				FBCQ = Qd[cell].x;

				var = tex1Dfetch(texM4, spline_index(x1)); //M4d_sh[spline_index(x1)];
				forcex += s1 * FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				i = i + 1;
				x1 = x - i;
				s1 = (x1 > 2) ? -1 : 1;
				cell = update_coords(i, FFTX) + cell2;
				FBCQ = Qd[cell].x;

				var = tex1Dfetch(texM4, spline_index(x1)); //M4d_sh[spline_index(x1)];
				forcex += s1 * FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				i = i + 1;
				x1 = x - i;
				s1 = (x1 > 2) ? -1 : 1;
				cell = update_coords(i, FFTX) + cell2;
				FBCQ = Qd[cell].x;

				var = tex1Dfetch(texM4, spline_index(x1)); //M4d_sh[spline_index(x1)];
				forcex += s1 * FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				//END LOOP UNROLL

			}//for (j = ceil(y-4); j < y; j++)

		}//for (i = ceil(x-4); i < x; i++){

		//force is negative of gradient
		forcex = -frac.x * frac.w * w * forcex * CC;
		forcey = -frac.y * frac.w * w * forcey * CC;
		forcez = -frac.z * frac.w * w * forcez * CC;
		enPME = 0.5 * CC * frac.w * w * enPME;

	}//if(gtid<natomd)

	f4d[gtid] = make_float4(forcex, forcey, forcez, enPME);

	ePMEd[gtid] = enPME;

#ifdef PCONSTANT
	viriald[gtid] += (make_float4(1.0f, 1.0f, 1.0f, 0.0f) -
	                  RBSRat * 2.0f) * (1.0f * enPME);
#endif
}

//==============================================================================
__global__ void PMEForce_large(cufftComplex* Qd,
                               float4* f4d,
                               float* ePMEd
#ifdef PCONSTANT
                             , float4* viriald
#endif
#ifdef DEBUG_PME
                             , float* pme_debug_d
#endif
                                                 ){
//==============================================================================

	int i;
	int j;
	int k;
	int cell;//, t1;

	float forcex = 0.0f;
	float forcey = 0.0f;
	float forcez = 0.0f;
	float enPME = 0.0f;

	float x1;
	float y1;
	float z1;
	float x;
	float y;
	float z;
	float w;

	//float myMass=1.0f;
	char s1;
	char s2;
	char s3;//signs of the odd function... derivtive of M4...

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

#ifdef PCONSTANT
	float4 frac = tex1Dfetch(texfrac, 0);
	float4 RBSRat = tex1Dfetch(texRBSRat, 0);
#endif

	int cell2;
	int cell3;

	float forcek1;
	float forcek2;
	float forcek3;
	float forcej1;
	float forcej2;
	float forcej3;
	float enerk;
	float enerj;
	float var; //forcei1, forcei2, forcei3, eneri,

	__shared__ float M4d_sh[NUM_SPLINE_SAMPLES + 1];
	__shared__ float dM4d_sh[NUM_SPLINE_SAMPLES + 1];

	for(i = threadIdx.x; i < NUM_SPLINE_SAMPLES + 1; i += blockDim.x){

		//copy global spline values to shared memory...
		dM4d_sh[i] = tex1Dfetch(texdM4, i); //dM4d_c[i];
		M4d_sh[i] = tex1Dfetch(texM4, i); //M4d_c[i];
	}

	__syncthreads();

	float4 pos = tex1Dfetch(texsclcrd, gtid);
	x = pos.x;
	y = pos.y;
	z = pos.z;
	w = pos.w;

	//read precomputed value of "fft(B.C.ifft(Q)) = convolution(B.C, Q)"
	//copy only x component, because forcex,
	//forcey and forcez as well as the emPME are all real
	float FBCQ = 0.0f;

	if(gtid < natomd){
		for(k = ceil(z - 4); k < z; k++){

			z1 = z - k;
			s3 = (z1 > 2) ? -1 : 1;
			cell3 = update_coords(k, FFTZ) * FFTX * FFTY;

			var = M4d_sh[spline_index(z1)];
			forcek1 = var;
			forcek2 = var;
			forcek3 = s3 * dM4d_sh[spline_index(z1)];
			enerk = var;

			for(j = ceil(y - 4); j < y; j++){

				y1 = y - j;
				cell2 = update_coords(j, FFTY) * FFTX + cell3;
				s2 = (y1 > 2) ? -1 : 1;
				var = M4d_sh[spline_index(y1)];
				forcej1 = var * forcek1;
				forcej2 = s2 * dM4d_sh[spline_index(y1)] * forcek2;
				forcej3 = var * forcek3;
				enerj = var * enerk;

				//forcei1 = forcei2 = forcei3 = 0;
				/*
				for (i = ceil(x-4); i < x; i++){

				x1 = x - i;
				s1 = (x1>2)?-1:1;
				cell = update_coords(i, FFTX) + cell2;

				//copy only x component, because forcex,
				//forcey and forcez as well as the emPME are all real
				FBCQ = Qd[cell].x;

				var = M4d_sh[spline_index(x1)];
				forcex += s1 *FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				}//for (k = ceil(z-4); k < z; k++){
				*/

				//LOOP UNROLL
				i = ceil(x - 4);
				x1 = x - i;
				s1 = (x1 > 2) ? -1 : 1;
				cell = update_coords(i, FFTX) + cell2;
				FBCQ = Qd[cell].x;

				var = M4d_sh[spline_index(x1)];
				forcex += s1 * FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				i = i + 1;
				x1 = x - i;
				s1 = (x1 > 2) ? -1 : 1;
				cell = update_coords(i, FFTX) + cell2;
				FBCQ = Qd[cell].x;

				var = M4d_sh[spline_index(x1)];
				forcex += s1 * FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				i = i + 1;
				x1 = x - i;
				s1 = (x1 > 2) ? -1 : 1;
				cell = update_coords(i, FFTX) + cell2;
				FBCQ = Qd[cell].x;

				var = M4d_sh[spline_index(x1)];
				forcex += s1 * FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				i = i + 1;
				x1 = x - i;
				s1 = (x1 > 2) ? -1 : 1;
				cell = update_coords(i, FFTX) + cell2;
				FBCQ = Qd[cell].x;

				var = M4d_sh[spline_index(x1)];
				forcex += s1 * FBCQ * dM4d_sh[spline_index(x1)] * forcej1;
				forcey += FBCQ * var * forcej2;
				forcez += FBCQ * var * forcej3;
				enPME += FBCQ * var * enerj;

				//END LOOP UNROLL

			}//for (j = ceil(y-4); j < y; j++)

		}//for (i = ceil(x-4); i < x; i++){

		//force is negative of gradient
		forcex = -frac.x * frac.w * w * forcex * CC;
		forcey = -frac.y * frac.w * w * forcey * CC;
		forcez = -frac.z * frac.w * w * forcez * CC;
		enPME = 0.5 * CC * frac.w * w * enPME;

	}//if(gtid<natomd)

	f4d[gtid] = make_float4(forcex, forcey, forcez, enPME);

	ePMEd[gtid] = enPME;

#ifdef PCONSTANT
	viriald[gtid] += (make_float4(1.0f, 1.0f, 1.0f, 0.0f) -
	                  RBSRat * 2.0f) * (1.0f * enPME);
#endif
}

#endif//#ifdef PME_CALC

//==============================================================================
