#include "Material_sort.cl"
#include "common_defines.h"
#include "debug.cl"

typedef struct __attribute__((packed)) __attribute__((aligned(16))) {
        double energy;
	int iter_num;
        int mat;
} Meta;

channel Meta MAT_QUEUE __attribute__((depth(32)));
channel int2 ADDR_QUEUE __attribute__((depth(321)));

__attribute__((max_global_work_dim(0)))
__kernel void grid_search(int lookups, __global GridPoint_Array *restrict A) {
	int n_isotopes = 355;
	int n_gridpoints = 11303;
	__local GridPoint_Array d;
	__local GridPoint_Array d_tmp;
	for (int i = 0; i < lookups; i++) {
	    	SearchContext ct = read_channel_altera(SC_QUEUE1);
	    	int mat = ct.mat;
	    	double p_energy = ct.energy;
		long mid;
	    	long ul = ct.ul;
	    	long ll = ct.ll;
		take_snapshot(0, mat);
	    	#pragma unroll
	    	for (int cid = 0; cid < SIZE; cid++) {
	      		mid = ((ul + ll) >> 1);
	      		d = A[mid];	
	      		ul = (d.energy > p_energy) ? mid : ul;
	      		ll = (d.energy > p_energy) ? ll : mid;
	    	}

		d_tmp = d;
		int iter_num = num_nucs[mat];
		int start_idx = cumulative_nucs[mat];
		Meta meta = {p_energy, iter_num, mat};
		write_channel_altera(MAT_QUEUE, meta);
		for( int j = 0; j < iter_num; j++ )
		{
			int tmp = start_idx + j;
			int p_nuc = mats[tmp];
			int energy_at_nuc = (int) d_tmp.xs_ptrs[p_nuc]; 
			int nu_idx = p_nuc * n_gridpoints + energy_at_nuc;
			int2 addr = (int2) (nu_idx, tmp);
			write_channel_altera(ADDR_QUEUE, addr);
		}
  	}
}
