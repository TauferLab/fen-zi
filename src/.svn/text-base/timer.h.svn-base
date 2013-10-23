/******************************************************************************/
//
// CHARMM-G, a MD Molecular Simulation code based on the CHARMM Force Field,
// with Reaction Forcefield and Particle Mesh Ewald
// 
// University of Delaware
// Authors: Omar Padron
// 
/******************************************************************************/

#ifndef _TIMER_CU_
#define _TIMER_CU_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h> //I've ommited this line.

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
	#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
	#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
struct timezone{
	int tz_minuteswest; /* minutes W of Greenwich */
	int tz_dsttime;     /* type of dst correction */
};
 
int gettimeofday(struct timeval *tv, struct timezone *tz){
	FILETIME ft;
	unsigned __int64 tmpres = 0;
	static int tzflag;
 
	if(NULL != tv){
		GetSystemTimeAsFileTime(&ft);
 
		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;
 
		/*converting file time to unix epoch*/
		tmpres -= DELTA_EPOCH_IN_MICROSECS; 
		tmpres /= 10;  /*convert into microseconds*/
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	}
 
	if(NULL != tz){
		if(!tzflag){
			_tzset();
			tzflag++;
		}
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}
 
	return 0;
}
#else
#include <sys/time.h>
#endif //_WIN32


typedef struct timeval CUDATimer;

void   initTimer(void);
void   tic      (void);
double toc      (void);
void   freeTimer(void);

CUDATimer *stack;

int stackSize;
int stackCap;

void initTimer(){
	stackSize = 0;
	stackCap = 10;
	stack = (CUDATimer *)malloc(stackCap * sizeof(CUDATimer));
}

void tic(){
	stackSize++;
	if(stackSize > stackCap){
		stackCap = stackSize + 9;
		stack = (CUDATimer *)realloc((void *)stack, stackCap * sizeof(CUDATimer));
	}
	gettimeofday(stack + stackSize - 1, NULL);
}

double toc(){
	CUDATimer now;
	double retval;
	--stackSize;
	gettimeofday(&now, NULL);
	retval = (double)(now.tv_sec - stack[stackSize].tv_sec) +
	                  (double)(now.tv_usec - stack[stackSize].tv_usec) * 1e-6;
	return retval;
}

void freeTimer(){
	free(stack);
}

#endif //_TIMER_CU_
