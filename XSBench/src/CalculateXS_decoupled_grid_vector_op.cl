#include "concs.cl"

channel Meta MAT_ACCU_QUEUE __attribute__((depth(32)));
channel double16 NU_QUEUE __attribute__((depth(350)));

__attribute__((max_global_work_dim(0)))
__kernel void calculate_macro_xs(int lookups,
				__global double16 *restrict nuclide_grids)
{
	for(int i = 0; i < lookups; i++) {
		Meta iter_meta = read_channel_altera(MAT_QUEUE);
		int iter_num = iter_meta.iter_num;
		write_channel_altera(MAT_ACCU_QUEUE, iter_meta);
		for( int j = 0; j < iter_num; j++ )
		{
			int2 addr = read_channel_altera(ADDR_QUEUE);
			double16 nu_data = nuclide_grids[addr.s0];
			nu_data.sf = concs[addr.s1].d;
			write_channel_altera(NU_QUEUE, nu_data);
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void accumulate_macro_xs(int lookups,
				__global ulong *restrict vhash) {
	ulong result = 0;
	for (int i = 0; i < lookups; i++) {
		Meta iter_meta = read_channel_altera(MAT_ACCU_QUEUE);
		double energy = iter_meta.energy; 
		int iter_num = iter_meta.iter_num;
		int mat = iter_meta.mat;
		float macro_xs_vector[5] = {0.0f};
		for(int j = 0; j < iter_num; j++) 
		{
			double16 nu_data = read_channel_altera(NU_QUEUE);	
			double8 low = nu_data.lo;
			double8 high = nu_data.hi;

			double8 xs_vector;
			double conc = nu_data.sf;
			double f = (high.s0 - energy) / (high.s0 - low.s0);
			xs_vector = mad(-f, (high - low), high) * conc;
			macro_xs_vector[0] += (float)xs_vector.s1; //mad(xs_vector.s1, conc, macro_xs_vector[0]);
			macro_xs_vector[1] += (float)xs_vector.s2;
			macro_xs_vector[2] += (float)xs_vector.s3;
			macro_xs_vector[3] += (float)xs_vector.s4;
			macro_xs_vector[4] += (float)xs_vector.s5;
		}

		ulong vhash_result = 0;
		unsigned int hash = 5381;	
		hash = ((hash << 5) + hash) + (int)energy;
		hash = ((hash << 5) + hash) + (int)mat;
		#pragma unroll
		for(int k = 0; k < 5; k++)
			hash = ((hash << 5) + hash) + macro_xs_vector[k];
		vhash_result = hash % 1000;
		result += vhash_result;
	}
	*vhash = result;
}
