#pragma OPENCL EXTENSION cl_altera_channels : enable channel 

#include "common_defines.h"
#include "debug.cl"

channel short delay_c __attribute__((depth(DEBUG_CHANNEL_DEPTH)));

unsigned short lfsr (unsigned short seed) { 
    unsigned short bit;
    unsigned short out;

    bit  = ((seed >> 0) ^ (seed >> 2) ^ (seed >> 3) ^ (seed >> 5) ) & 1;
    out =  (seed >> 1) | (bit << 15);
    return out;
}

__attribute__((max_global_work_dim(0)))
__kernel void matrix_multiply
                    (__global DATA_TYPE* restrict data_a
                    ,__global DATA_TYPE* restrict data_b
                    ,__global DATA_TYPE* restrict data_c
                    ,int row_a
                    ,int col_a
                    ,int col_b
                    ) {
    add_watch(0,(size_t) &data_a[0]);
    add_watch(1,(size_t) &data_b[0]);
    add_watch(2,(size_t) &data_c[5]);
    for(int i = 0; i < row_a ; i++) { 
        for(int j = 0; j < col_b; j++) { 
         DATA_TYPE acc=0;
         DATA_TYPE a=0; 
         DATA_TYPE b=0;
            for(int k = 0; k < col_a; k++) { 
                take_snapshot(0,k);
                a     = data_a[i*col_a + k];
                b     = data_b[k*col_b + j];
                acc  += a*b; 

//               write_channel_altera(delay_c,(ushort) a);

                monitor_address(0,(size_t) &data_a[i*col_a + k], 1);
                monitor_address(1,(size_t) &data_b[k*col_b + j], 2+j);
                take_snapshot(1,acc);
            }
            data_c[i*col_b + j] = acc;
            monitor_address(2,(size_t) &data_c[i*col_b + j], i+j);
        }
    }
}


__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void r_pipe ()  { 

unsigned short delay_read;
unsigned short seed = 0x3431;

   // #pragma acc kernels loop independent
   // for(ulong infinite_counter = 0; infinite_counter < ULONG_MAX ; infinite_counter++) { 
   //     if(delay_read == 0) {
   //         seed        = lfsr(seed);
   //         delay_read  = seed & 0x7;
   //         take_snapshot(2,(stamp_t) read_channel_altera(delay_c));
   //     }    
   //     else delay_read--; 
   // }
}


