#include "omp.h"
#include <string.h>

#define lowerb(id, p, n)  ( id * (n/p) + (id < (n%p) ? id : n%p) )
#define numElem(id, p, n) ( (n/p) + (id < (n%p)) )
#define upperb(id, p, n)  ( lowerb(id, p, n) + numElem(id, p, n) - 1 )

#define min(a, b) ( (a < b) ? a : b )
#define max(a, b) ( (a > b) ? a : b )

extern int userparam;

// Function to copy one matrix into another
void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey) {
	int nblocksi = omp_get_max_threads();

	# pragma omp parallel
	{	
	int blocki = omp_get_thread_num();
	int i_start = lowerb(blocki, nblocksi, sizex);
	int i_end = upperb(blocki, nblocksi, sizex);
	for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++)
		for (int j=1; j<=sizey-2; j++)
			v[i*sizey+j] = u[i*sizey+j];
	}
}

// 2D-blocked solver: one iteration step
double solve (double *u, double *unew, unsigned sizex, unsigned sizey) {
	double tmp, diff, sum=0.0;
	
	if ( u==unew ) {
		int nblocksi=omp_get_max_threads();
		int nblocksj = userparam;

		#pragma omp parallel for ordered(2) private(diff) reduction(+:sum)
		for (int ii =  0; ii < nblocksi; ii++) {
			for (int jj =  0; jj < nblocksj; jj++) {
				int i_start = lowerb(ii, nblocksi, sizex);
				int i_end = upperb(ii, nblocksi, sizex);
				int j_start = lowerb(jj, nblocksj, sizex);
				int j_end = upperb(jj, nblocksj, sizex);
				#pragma omp ordered depend (sink: ii-1, jj)
				for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
					for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++) {
					tmp = 0.25 * ( 	u[ i*sizey	   + (j-1) ] +  // left
									u[ i*sizey	   + (j+1) ] +  // right
									u[ (i-1)*sizey + j     ] +  // top
									u[ (i+1)*sizey + j     ] ); // bottom
					diff = tmp - u[i*sizey+ j];
					sum += diff * diff;
					unew[i*sizey+j] = tmp;
					}
				}
				#pragma omp ordered depend(source)
			}
		}
	}

	else {
		#pragma omp parallel private(diff) reduction(+:sum)
		{
		int nblocksi = omp_get_num_threads();
		int blocki = omp_get_thread_num();
		int i_start = lowerb(blocki, nblocksi, sizex);
		int i_end = upperb(blocki, nblocksi, sizex);
		for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
			for (int j=1; j<=sizey-2; j++) {
				tmp = 0.25 * ( 	u[ i*sizey	   + (j-1) ] +  // left
								u[ i*sizey	   + (j+1) ] +  // right
								u[ (i-1)*sizey + j     ] +  // top
								u[ (i+1)*sizey + j     ] ); // bottom
				diff = tmp - u[i*sizey+ j];
				sum += diff * diff;
				unew[i*sizey+j] = tmp;
			}
		}
		}
	}
	return sum;
}


