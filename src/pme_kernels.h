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

#ifndef _PME_KENRELS_H_
#define _PME_KENRELS_H_

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
                                                     );

//==============================================================================
///Function to spread charge and update the neighbor list of the cell..
__global__ void ChargeSpread_medium(cufftComplex* Qd,
                                    float4* r4d_scaled,
                                    int* cellL,
                                    int* numL
#ifdef DEBUG_PME
                                  , float* pme_debug_d
#endif
                                                      );

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
                                                  );

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
                                                 );

#endif//#ifdef PME_CALC

//==============================================================================

#include "pme_kernels.cu"

#endif //_PME_KENRELS_H_
