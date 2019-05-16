#include "../host/inc/common_defines.h"
#include "../host/inc/debug_defines.h"
#include "../../common/kernels/debug.cl"

__kernel void mirror_content( unsigned max_i, global int* restrict out)
{
	for (int i = 1; i < max_i; i++) {
		out[max_i*2-i] = out[i];
	}
}
