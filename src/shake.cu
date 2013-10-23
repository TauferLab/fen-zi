/*****************************************************************************/
//
// MD code for GPU based on a modified version of the CHARMM force field, with
// reaction force field (RF) and Particle Mesh Ewald (PME)
//
// Global Computing Lab, University of Delaware
// Author(s): Narayan Ganesan (narayan.ganesan.8@gmail.com)
// Contact(s): Michela Taufer (taufer@udel.edu)
// Reference(s):
//
/*****************************************************************************/
#include "globals.h"
#include "cucomplexops.h"

//Shake implementation in FENZI using constraint algorithm...
// solve_bond_constraints
// solve_velocity_constraints

//------------------------------------------------------------------------------
__global__ void solve_bond_constraints(int2* constraintsd,
                                       unsigned char* constraints_by_atomd,
                                       float2* constraintsprmd,
                                       int* atoms_in_clusterd,
                                       float4* r4shaked,
                                       float4* r4d,
                                       float4* v4d
#ifdef PCONSTANT
                                     , float4* viriald
#endif
                                                      ){
//------------------------------------------------------------------------------
//constraints_clusterd - constraints by group....
//ordered by cluster, that containts a dependent set of constraints..

//constraintsd - list of atoms in a constraint,
//bond length and the multiplier ordered by constraint id..

//atoms_constraintsd - list of constraints an atom belongs to..
//ordered by atomid

	unsigned char constraints_by_atom[ATOMS_IN_CLUSTER][CONSTRAINTS_PER_ATOM + 1];
	float mass[ATOMS_IN_CLUSTER];

	int2 constraints[CLUSTER_SIZE];
	float3 r_diff[CLUSTER_SIZE];

	float lambda[CLUSTER_SIZE];
	float3 dC_dl[CLUSTER_SIZE][CLUSTER_SIZE];

	float J[CLUSTER_SIZE][CLUSTER_SIZE];
	float d[CLUSTER_SIZE];

	float3 r[ATOMS_IN_CLUSTER];
	//array to hold the updated positions of atoms...
	float3 r_new[ATOMS_IN_CLUSTER];

	unsigned char i1;
	unsigned char j1;
	unsigned char c;
	unsigned char l;

	//unsigned char i2, j2;
	unsigned char n;

	int k1;
	int k2;
	int k3;

	char signval;
	char i = (-1);

	int gi;
	int gj;

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	for(k1 = 0; k1 < CLUSTER_SIZE; k1++){

		//Ck[k1] = make_float3(0.0f, 0.0f, 0.0f);
		lambda[k1] = 0.0f;
		d[k1] = 0.0f;
		for(k2 = 0; k2 < CLUSTER_SIZE; k2++){
			dC_dl[k1][k2] = make_float3(0.0f, 0.0f, 0.0f);
		}
	}

	n = constraintsd[0 * nClustersd + gtid].x;
//copy the data to shared memory and registers...//
	for(k1 = 1; k1 <= n; k1++){

		gi = constraintsd[k1 * nClustersd + gtid].x;
		gj = constraintsd[k1 * nClustersd + gtid].y;

		i1 = (gi >> 24); //local atomid unsigned char
		j1 = (gj >> 24); //local atomid unsigned char

		gi = gi & ((1 << 24) - 1);
		gj = gj & ((1 << 24) - 1);

		constraints[k1 - 1].x = i1;
		constraints[k1 - 1].y = j1;

		r_diff[k1 - 1] = float4_to_float3(r4shaked[gi] - r4shaked[gj]);

		// choose nearest image
		r_diff[k1 - 1].x = __my_add3(r_diff[k1 - 1].x,
		                             -__mycopysignf(boxH.x,
		                                            __my_fadd(r_diff[k1 - 1].x,
		                                                      -boxH.x)),
		                             -__mycopysignf(boxH.x,
		                                            __my_fadd(r_diff[k1 - 1].x,
		                                                      boxH.x)));

		r_diff[k1 - 1].y = __my_add3(r_diff[k1 - 1].y,
		                             -__mycopysignf(boxH.y,
		                                            __my_fadd(r_diff[k1 - 1].y,
		                                                      -boxH.y)),
		                             -__mycopysignf(boxH.y,
		                                            __my_fadd(r_diff[k1 - 1].y,
		                                                      boxH.y)));

		r_diff[k1 - 1].z = __my_add3(r_diff[k1 - 1].z,
		                             -__mycopysignf(boxH.z,
		                                            __my_fadd(r_diff[k1 - 1].z,
		                                                      -boxH.z)),
		                             -__mycopysignf(boxH.z,
		                                            __my_fadd(r_diff[k1 - 1].z,
		                                                      boxH.z)));

		lambda[k1 - 1] = constraintsprmd[(k1 - 1) * nClustersd + gtid].x;
		d[k1 - 1] = constraintsprmd[(k1 - 1) * nClustersd + gtid].y;

		mass[i1] = tex1Dfetch(texprm, tex1Dfetch(textype, gi)).x;
		mass[j1] = tex1Dfetch(texprm, tex1Dfetch(textype, gj)).x;
		r[i1] = float4_to_float3(r4d[gi]);
		r[j1] = float4_to_float3(r4d[gj]);

		constraints_by_atom[i1][0] = constraints_by_atomd[0 * natomd + gi];
		c = constraints_by_atom[i1][0];

		for(k2 = 1; k2 <= c; k2++){
			constraints_by_atom[i1][k2] = constraints_by_atomd[k2 * natomd + gi];
		}

		constraints_by_atom[j1][0] = constraints_by_atomd[0 * natomd + gj];
		c = constraints_by_atom[j1][0];

		for(k2 = 1; k2 <= c; k2++){
			constraints_by_atom[j1][k2] = constraints_by_atomd[k2 * natomd + gj];
		}

	}
///////////////////////////////////////end Copy data////////////////////////////

////////////////////////Construct dC_dlambda once/////////////////////
	for(k1 = 0; k1 < n; k1++){
		i1 = constraints[k1].x;
		j1 = constraints[k1].y;

		c = constraints_by_atom[i1][0];
		for(k2 = 1; k2 <= c; k2++){
			l = constraints_by_atom[i1][k2];
			signval = 1 - 2 * (char(l >> 7)); //signval is char.. extract sign bit...
			l = l & 0x7F;
			dC_dl[k1][l] += (r_diff[l] * (1.0f * signval / mass[i1]));
			//(r_diff[l]*(2.0f*signval/mass[i1]));
		}

		c = constraints_by_atom[j1][0];

		for(k2 = 1; k2 <= c; k2++){
			l = constraints_by_atom[j1][k2];
			signval = 1 - 2 * (char(l >> 7)); //signval is char.. extract sign bit...
			l = l & 0x7F;
			dC_dl[k1][l] -= (r_diff[l] * (1.0f * signval / mass[j1]));
			//(r_diff[l]*(2.0f*signval/mass[j1]));
		}
	}

//////////////////end Construct dC_dlambda... ///////////////////

////////now determine the lagrange multipliers via iterative Newton's method...
	float tempvec[CLUSTER_SIZE];
	float multfac;
	float tolerance;
	float sigma[CLUSTER_SIZE];
	//unsigned char count=0;
	float3 Ck;

/////////////////do While loop///////////////////
	do{
////////////////////////////////////////////////

		for(i1 = 0; i1 < ATOMS_IN_CLUSTER; i1++){
			r_new[i1] = make_float3(0.0f, 0.0f, 0.0f);
		}

		for(k1 = 0; k1 < n; k1++){
			i1 = constraints[k1].x;
			j1 = constraints[k1].y;
			r_new[i1] += r_diff[k1] * (lambda[k1] / mass[i1]);
			r_new[j1] -= r_diff[k1] * (lambda[k1] / mass[j1]);
		}

//now construct the Jacobian... and the constraint differences, sigma...(RHS)
		for(k1 = 0; k1 < n; k1++){
			i1 = constraints[k1].x;
			j1 = constraints[k1].y;
			Ck = r[i1] - r[j1];
			//choose nearest image..
			Ck.x = __my_add3(Ck.x,
			                 -__mycopysignf(boxH.x,__my_fadd(Ck.x, -boxH.x)),
			                 -__mycopysignf(boxH.x,__my_fadd(Ck.x, boxH.x)));

			Ck.y = __my_add3(Ck.y,
			                 -__mycopysignf(boxH.y,__my_fadd(Ck.y, -boxH.y)),
			                 -__mycopysignf(boxH.y,__my_fadd(Ck.y, boxH.y)));

			Ck.z = __my_add3(Ck.z,
			                 -__mycopysignf(boxH.z,__my_fadd(Ck.z, -boxH.z)),
			                 -__mycopysignf(boxH.z,__my_fadd(Ck.z, boxH.z)));

			Ck += (r_new[i1] - r_new[j1]);

			sigma[k1] = sum(Ck * Ck) - d[k1] * d[k1]; //sqrt(sum(Ck*Ck)) - d[k1];
			for(k2 = 0; k2 < n; k2++){
				J[k1][k2] = sum(dC_dl[k1][k2] * Ck) * 2.0f;
				//sum(dC_dl[k1][k2]*Ck)*2.0f;
			}
		}
	//////////end Jacobian/////////
	//now solve for constraints, by Newtons Method...

////////////Start Gaussian Elimination...
		for(k1 = 0; k1 < n; k1++){
			for(k2 = k1 + 1; k2 < n; k2++){
				multfac = J[k2][k1] / J[k1][k1];

				for(k3 = k1 + 1; k3 < n; k3++){
					J[k2][k3] -= (multfac * J[k1][k3]);
				}

				sigma[k2] -= (multfac * sigma[k1]);
			}
		}

		//the back substitution:
		tolerance = 0.0f;
		for(k1 = n - 1; k1 >= 0; k1--){
			tempvec[k1] = sigma[k1];

			for(k2 = k1 + 1; k2 < n; k2++){
				tempvec[k1] -= (J[k1][k2] * tempvec[k2]);
			}

			tempvec[k1] = tempvec[k1] / J[k1][k1];

			//subtract the solution for Newton's iteration...
			lambda[k1] -= tempvec[k1];

			tolerance += (tempvec[k1] * tempvec[k1]);
		}

//////////////////////End Gaussian Elimination...	/////////////////////////

///////////end do While loop////////////////////////////////////////////////////
	}while(tolerance >= shaketold); //(++count<4);
//////////end solve for the system...///////////////////////////////////////////

	float3 constraint_force;
#ifdef PCONSTANT
	float3 cons_virial;
	float3 virial[ATOMS_IN_CLUSTER];
#endif

//now update the coordinates, velocities and virial...
	for(i1 = 0; i1 < ATOMS_IN_CLUSTER; i1++){
		r_new[i1] = make_float3(0.0f, 0.0f, 0.0f);
#ifdef PCONSTANT
		virial[i1] = make_float3(0.0f, 0.0f, 0.0f);
#endif
	}

	for(k1 = 0; k1 < n; k1++){
		gi = constraintsd[(k1 + 1) * nClustersd + gtid].x;
		gj = constraintsd[(k1 + 1) * nClustersd + gtid].y;

		i1 = (gi >> 24); //local atomid unsigned char
		j1 = (gj >> 24); //local atomid unsigned char

		gi = gi & ((1 << 24) - 1);
		gj = gj & ((1 << 24) - 1);

		constraint_force = r_diff[k1] * lambda[k1];

#ifdef PCONSTANT
		cons_virial = (constraint_force * r_diff[k1] * 0.5f) /
		              (TU * deltaTd * TU * deltaTd);

		virial[i1] += cons_virial;
		virial[j1] += cons_virial;
#endif

		r_new[i1] += (constraint_force / mass[i1]);
		r_new[j1] -= (constraint_force / mass[j1]);
	}

	for(i = 1; i <= atoms_in_clusterd[gtid]; i++){
		gi = atoms_in_clusterd[nClustersd * i + gtid];

		i1 = (gi >> 24); //local atomid unsigned char
		gi = gi & ((1 << 24) - 1);

		r4d[gi].x += r_new[i1].x;
		r4d[gi].y += r_new[i1].y;
		r4d[gi].z += r_new[i1].z;

		v4d[gi] += r_new[i1] / (TU * deltaTd);
#ifdef PCONSTANT
		viriald[gi] += float3_to_float4(virial[i1]);
#endif
	}
}

//------------------------------------------------------------------------------
__global__ void solve_velocity_constraints(int2* constraintsd,
                                           unsigned char* constraints_by_atomd,
                                           float2* constraintsprmd,
                                           int* atoms_in_clusterd,
                                           float4* v4d //, float4 *r4d
#ifdef PCONSTANT
                                         , float4* viriald
#endif
                                                          ){
//------------------------------------------------------------------------------

//constraints_clusterd - constraints by group....
//ordered by cluster, that containts a dependent set of constraints..

//constraintsd - list of atoms in a constraint,
//bond length and the multiplier ordered by constraint id..

//atoms_constraintsd - list of constraints an atom belongs to..
//ordered by atomid

	unsigned char constraints_by_atom[ATOMS_IN_CLUSTER][CONSTRAINTS_PER_ATOM + 1];
	float mass[ATOMS_IN_CLUSTER];

	int2 constraints[CLUSTER_SIZE];
	float3 r_diff[CLUSTER_SIZE];

	float lambda[CLUSTER_SIZE];
	float3 dC_dl[CLUSTER_SIZE][CLUSTER_SIZE];

	float J[CLUSTER_SIZE][CLUSTER_SIZE];

	float3 v[ATOMS_IN_CLUSTER];

	//array to hold the updated positions of atoms...
	float3 v_new[ATOMS_IN_CLUSTER];

	unsigned char i1;
	unsigned char j1;
	unsigned char c;
	unsigned char l; //i2, j2,
	unsigned char n;

	int k1;
	int k2;
	int k3;

	char signval;
	char i = (-1);

	int gi;
	int gj;

	unsigned int gtid = COMPUTE_GLOBAL_THREADID

#ifdef PCONSTANT
	float3 boxH = float4_to_float3(tex1Dfetch(texboxH, 0));
#endif

	for(k1 = 0; k1 < CLUSTER_SIZE; k1++){

		//Ck[k1] = make_float3(0.0f, 0.0f, 0.0f);
		lambda[k1] = 0.0f;
		for (k2 = 0; k2 < CLUSTER_SIZE; k2++){
			dC_dl[k1][k2] = make_float3(0.0f, 0.0f, 0.0f);
		}
	}

	n = constraintsd[0 * nClustersd + gtid].x;
//copy the data to shared memory and registers.../////////////
	for(k1 = 1; k1 <= n; k1++){

		gi = constraintsd[k1 * nClustersd + gtid].x;
		gj = constraintsd[k1 * nClustersd + gtid].y;

		i1 = (gi >> 24); //local atomid unsigned char
		j1 = (gj >> 24); //local atomid unsigned char

		gi = gi & ((1 << 24) - 1);
		gj = gj & ((1 << 24) - 1);

		constraints[k1 - 1].x = i1;
		constraints[k1 - 1].y = j1;

		r_diff[k1 - 1] = float4_to_float3(tex1Dfetch(texcrd, gi) -
		                                  tex1Dfetch(texcrd, gj));

		//choose nearest image
		r_diff[k1 - 1].x = __my_add3(r_diff[k1-1].x,
		                             -__mycopysignf(boxH.x,
		                                            __my_fadd(r_diff[k1 - 1].x,
		                                                      -boxH.x)),
		                             -__mycopysignf(boxH.x,
		                                            __my_fadd(r_diff[k1 - 1].x,
		                                            boxH.x)));

		r_diff[k1- 1].y = __my_add3(r_diff[k1 - 1].y,
		                             -__mycopysignf(boxH.y,
		                                            __my_fadd(r_diff[k1 - 1].y,
		                                                      -boxH.y)),
		                             -__mycopysignf(boxH.y,
		                                            __my_fadd(r_diff[k1 - 1].y,
		                                                      boxH.y)));

		r_diff[k1 - 1].z = __my_add3(r_diff[k1 - 1].z,
		                             -__mycopysignf(boxH.z,
		                                            __my_fadd(r_diff[k1 - 1].z,
		                                                      -boxH.z)),
		                             -__mycopysignf(boxH.z,
		                                            __my_fadd(r_diff[k1 - 1].z,
		                                                      boxH.z)));

		lambda[k1 - 1] = constraintsprmd[(k1 - 1) * nClustersd + gtid].x;

		mass[i1] = tex1Dfetch(texprm, tex1Dfetch(textype, gi)).x;
		mass[j1] = tex1Dfetch(texprm, tex1Dfetch(textype, gj)).x;
		v[i1] = float4_to_float3(v4d[gi]);
		v[j1] = float4_to_float3(v4d[gj]);

		constraints_by_atom[i1][0] = constraints_by_atomd[0 * natomd + gi];
		c = constraints_by_atom[i1][0];

		for(k2 = 1; k2 <= c; k2++){
			constraints_by_atom[i1][k2] = constraints_by_atomd[k2 * natomd + gi];
		}

		constraints_by_atom[j1][0] = constraints_by_atomd[0 * natomd + gj];
		c = constraints_by_atom[j1][0];

		for(k2 = 1; k2 <= c; k2++){
			constraints_by_atom[j1][k2] = constraints_by_atomd[k2 * natomd + gj];
		}

	}
///////////////////////////////////////end Copy data////////////////////////////

////////////////////////Construct dC_dlambda once/////////////////////
	for(k1 = 0; k1 < n; k1++){

		i1 = constraints[k1].x;
		j1 = constraints[k1].y;

		c = constraints_by_atom[i1][0];
		for(k2 = 1; k2 <= c; k2++){
			l = constraints_by_atom[i1][k2];
			signval = 1 - 2 * (char(l >> 7)); //signval is char.. extract sign bit...
			l = l & 0x7F;
			dC_dl[k1][l] += (r_diff[l] * (1.0f * signval / mass[i1]));
		}

		c = constraints_by_atom[j1][0];
		for(k2 = 1; k2 <= c; k2++){
			l = constraints_by_atom[j1][k2];
			signval = 1 - 2 * (char(l >> 7)); //signval is char.. extract sign bit...
			l = l & 0x7F;
			dC_dl[k1][l] -= (r_diff[l] * (1.0f * signval / mass[j1]));
		}
	}
//////////////////end Construct dC_dlambda... ///////////////////

////////now determine the lagrange multipliers via iterative Newton's method...
	float tempvec[CLUSTER_SIZE];
	float multfac;
	float tolerance;
	float sigma[CLUSTER_SIZE];
	float3 Ck;

/////////////////do While loop///////////////////
	do{
////////////////////////////////////////////////

		for(i1 = 0; i1 < ATOMS_IN_CLUSTER; i1++){
			v_new[i1] = make_float3(0.0f, 0.0f, 0.0f);
		}

		for(k1 = 0; k1 < n; k1++){
			i1 = constraints[k1].x;
			j1 = constraints[k1].y;
			v_new[i1] += r_diff[k1] * (lambda[k1] / mass[i1]);
			v_new[j1] -= r_diff[k1] * (lambda[k1] / mass[j1]);
		}

//now construct the Jacobian... and the constraint differences, sigma...(RHS)
		for(k1 = 0; k1 < n; k1++){
			i1 = constraints[k1].x;
			j1 = constraints[k1].y;
			Ck = v[i1] - v[j1];

			Ck += (v_new[i1] - v_new[j1]);

			sigma[k1] = sum(Ck * r_diff[k1]);
			for(k2 = 0; k2 < n; k2++){
				J[k1][k2] = sum(dC_dl[k1][k2] * r_diff[k1]);
			}
		}

	//////////end Jacobian/////////
	//now solve for constraints, by Newtons Method...

////////////Start Gaussian Elimination...

		for(k1 = 0; k1 < n; k1++){
			for(k2 = k1 + 1; k2 < n; k2++){
				multfac = J[k2][k1] / J[k1][k1];

				for(k3 = k1 + 1; k3 < n; k3++){
					J[k2][k3] -= (multfac * J[k1][k3]);
				}

				sigma[k2] -= (multfac * sigma[k1]);
			}
		}

		//the back substitution:
		tolerance = 0.0f;
		for(k1 = n - 1; k1 >= 0; k1--){
			tempvec[k1] = sigma[k1];

			for(k2 = k1 + 1; k2 < n; k2++){
				tempvec[k1] -= (J[k1][k2] * tempvec[k2]);
			}

			tempvec[k1] = tempvec[k1] / J[k1][k1];

			//subtract the solution for Newton's iteration...
			lambda[k1] -= tempvec[k1];

			tolerance += (tempvec[k1] * tempvec[k1]);
		}

//////////////////////End Gaussian Elimination... /////////////////////////

///////////end do While loop////////////////////////////////////////////////////
	}while(tolerance >= shaketold); //(++count<4);
//////////end solve for the system...///////////////////////////////////////////

	float3 constraint_force;
#ifdef PCONSTANT
	float3 cons_virial;
	float3 virial[ATOMS_IN_CLUSTER];
#endif

	//now update the velocities and virial...
	for(i1 = 0; i1 < ATOMS_IN_CLUSTER; i1++){
		v_new[i1] = make_float3(0.0f, 0.0f, 0.0f);
#ifdef PCONSTANT
		virial[i1] = make_float3(0.0f, 0.0f, 0.0f);
#endif
	}

	for(k1 = 0; k1 < n; k1++){

		gi = constraintsd[(k1 + 1) * nClustersd + gtid].x;
		gj = constraintsd[(k1 + 1) * nClustersd + gtid].y;

		i1 = (gi >> 24); //local atomid unsigned char
		j1 = (gj >> 24); //local atomid unsigned char

		gi = gi & ((1 << 24) - 1);
		gj = gj & ((1 << 24) - 1);

		constraint_force = r_diff[k1] * lambda[k1];

#ifdef PCONSTANT
		cons_virial = (constraint_force * r_diff[k1] * 0.5f) / (TU * deltaTd);
		virial[i1] += cons_virial;
		virial[j1] += cons_virial;
#endif

		v_new[i1] += (constraint_force / mass[i1]);
		v_new[j1] -= (constraint_force / mass[j1]);
	}

	for(i = 1; i <= atoms_in_clusterd[gtid]; i++){
		gi = atoms_in_clusterd[nClustersd * i + gtid];

		i1 = (gi >> 24); //local atomid unsigned char
		gi = gi & ((1 << 24) - 1);

		v4d[gi].x += v_new[i1].x;
		v4d[gi].y += v_new[i1].y;
		v4d[gi].z += v_new[i1].z;
#ifdef PCONSTANT
		viriald[gi] += float3_to_float4(virial[i1]);
#endif
	}
}
