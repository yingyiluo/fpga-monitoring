#ifndef DEBUG__CL 
#define DEBUG__CL

#include "timer.h"
#include "debug_defines.h"

#if NUM_DEBUG_POINTS > 0 
channel short            time_in_c    [NUM_DEBUG_POINTS] __attribute__((depth(DEBUG_CHANNEL_DEPTH)));
channel debug_command_e  command_c    [NUM_DEBUG_POINTS] __attribute__((depth(DEBUG_CHANNEL_DEPTH)));
channel stamp_t          read_stamp_c [NUM_DEBUG_POINTS] __attribute__((depth(DEBUG_CHANNEL_DEPTH)));
#endif //NUM_DEBUG_POINTS 0

#if NUM_WATCH_POINTS > 0
channel int              addr_in_c   [NUM_WATCH_POINTS] __attribute__((depth(DEBUG_CHANNEL_DEPTH)));
channel watch_s          watch_in_c  [NUM_WATCH_POINTS] __attribute__((depth(DEBUG_CHANNEL_DEPTH)));
channel watch_s          watch_out_c [NUM_WATCH_POINTS] __attribute__((depth(DEBUG_CHANNEL_DEPTH)));
channel debug_command_e  watch_cmd_c [NUM_WATCH_POINTS] __attribute__((depth(DEBUG_CHANNEL_DEPTH)));
#endif //NUM_WATCH_POINTS > 0


void add_watch(uint id, size_t address) { 
#if NUM_WATCH_POINTS > 0
    if(id < NUM_WATCH_POINTS) 
        (void) write_channel_nb_altera(addr_in_c[id],(int) address);
#endif 
}

void monitor_address(uint id, size_t addr, ushort tag) { 
#if NUM_WATCH_POINTS > 0
    watch_s in;
    in.addr = (uint) addr; 
    in.tag  = tag;
    if(id < NUM_WATCH_POINTS) {
        (void) write_channel_nb_altera(watch_in_c[id], in);
        mem_fence(CLK_CHANNEL_MEM_FENCE);
    }
#endif 
}



void take_snapshot(uint id, stamp_t in) { 
#if NUM_DEBUG_POINTS > 0
    if(id < NUM_DEBUG_POINTS) {
        (void) write_channel_nb_altera(time_in_c[id],(short) in);
        mem_fence(CLK_CHANNEL_MEM_FENCE);
    }
#endif 
}

#if NUM_DEBUG_POINTS > 0

#ifdef EMULATOR
void sample_stamp_emulate(uchar buffer_id) {
    //printf("in sample stamp emulate, id: %d\n", buffer_id);
    #include "sample_stamp_body.cl"
}

#if NUM_DEBUG_POINTS > 0 
    void kernel sample_stamp_0() { sample_stamp_emulate(0); }
#endif  //0
#if NUM_DEBUG_POINTS > 1 
    void kernel sample_stamp_1() { sample_stamp_emulate(1); }
#endif  //1
#if NUM_DEBUG_POINTS > 2 
    void kernel sample_stamp_2() { sample_stamp_emulate(2); }
#endif  //2
#if NUM_DEBUG_POINTS > 3 
    void kernel sample_stamp_3() { sample_stamp_emulate(3); }
#endif  //3
#if NUM_DEBUG_POINTS > 4 
    void kernel sample_stamp_4() { sample_stamp_emulate(4); }
#endif  //4
#if NUM_DEBUG_POINTS > 5 
    void kernel sample_stamp_5() { sample_stamp_emulate(5); }
#endif  //5
#if NUM_DEBUG_POINTS > 6 
    void kernel sample_stamp_6() { sample_stamp_emulate(6); }
#endif  //6
#if NUM_DEBUG_POINTS > 7 
    void kernel sample_stamp_7() { sample_stamp_emulate(7); }
#endif  //7
#if NUM_DEBUG_POINTS > 8 
    void kernel sample_stamp_8() { sample_stamp_emulate(8); }
#endif  //8
#if NUM_DEBUG_POINTS > 9 
    void kernel sample_stamp_9() { sample_stamp_emulate(9); }
#endif  //9
#if NUM_DEBUG_POINTS > 10 
    void kernel sample_stamp_10() { sample_stamp_emulate(10); }
#endif  //10
#if NUM_DEBUG_POINTS > 11 
    #pragma error "Maximum 11 channels are supported for emulator" 
#endif //11

#else 

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(NUM_DEBUG_POINTS,1)))
kernel
void sample_stamp() { 
    uchar           buffer_id;
    buffer_id       = get_compute_id(0);
    #include "sample_stamp_body.cl" 
}

#endif //EMULATOR

__attribute__((max_global_work_dim(0)))
__kernel
void read_stamp( debug_command_e  cmd,
                __global stamp_t* restrict output,
                uint trace_buffer_id) {

    uint id = trace_buffer_id;

    //printf("enter read stamp \n");     
    #pragma unroll
    for(int i = 0; i < NUM_DEBUG_POINTS; i++) {
        //printf("before write\n");     
        if(i == id) 
             write_channel_altera(command_c[i], (debug_command_e) cmd);
        //printf("after write \n");     
    }

    if(cmd == D_READ) { 
    //printf("enter if D_READ\n");     
    //buffer_s out;
    stamp_t out;
        for(int read_count = 0; read_count <= DEBUG_SAMPLE_DEPTH; read_count++) { 
            //printf("read count: %d\n", read_count);     
            #pragma unroll
            for(int i = 0; i < NUM_DEBUG_POINTS; i++) {
                if(i == id) 
                    out = read_channel_altera(read_stamp_c[i]);
            }
            //output[read_count] = (stamp_t) ((((ulong) out.hi) << 32) | out.lo);
            output[read_count] = out; 
            mem_fence(CLK_CHANNEL_MEM_FENCE);
            //printf("Read Out = %0lu \n", output[read_count]);     
        }
    }
}
#endif //NUM_DEBUG_POINTS

#if NUM_WATCH_POINTS > 0

#ifdef EMULATOR
void watch_addr_emulate(uchar watch_id) {
    #include "watch_addr_body.cl"
}

#if NUM_WATCH_POINTS > 0 
    void kernel watch_addr_0() { watch_addr_emulate(0); }
#endif  //0
#if NUM_WATCH_POINTS > 1 
    void kernel watch_addr_1() { watch_addr_emulate(1); }
#endif  //1
#if NUM_WATCH_POINTS > 2 
    void kernel watch_addr_2() { watch_addr_emulate(2); }
#endif  //2
#if NUM_WATCH_POINTS > 3 
    void kernel watch_addr_3() { watch_addr_emulate(3); }
#endif  //3
#if NUM_WATCH_POINTS > 4 
    void kernel watch_addr_4() { watch_addr_emulate(4); }
#endif  //4
#if NUM_WATCH_POINTS > 5 
    void kernel watch_addr_5() { watch_addr_emulate(5); }
#endif  //5
#if NUM_WATCH_POINTS > 6 
    void kernel watch_addr_6() { watch_addr_emulate(6); }
#endif  //6
#if NUM_WATCH_POINTS > 7 
    void kernel watch_addr_7() { watch_addr_emulate(7); }
#endif  //7
#if NUM_WATCH_POINTS > 8 
    void kernel watch_addr_8() { watch_addr_emulate(8); }
#endif  //8
#if NUM_WATCH_POINTS > 9 
    void kernel watch_addr_9() { watch_addr_emulate(9); }
#endif  //9
#if NUM_WATCH_POINTS > 10 
    void kernel watch_addr_10() { watch_addr_emulate(10); }
#endif  //10
#if NUM_WATCH_POINTS > 11 
    #pragma error "Maximum 11 channels are supported for emulator" 
#endif //11

#else 

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(NUM_WATCH_POINTS,1)))
kernel
void watch_addr() { 
    uint watch_id; 
    watch_id = get_compute_id(0);
    #include "watch_addr_body.cl"
}

#endif //EMULATOR

__attribute__((max_global_work_dim(0))) 
__kernel 
void read_watch(debug_command_e cmd,
                __global watch_s* restrict output,
                uint watch_id) { 
    uint id = watch_id;
    #pragma unroll
    for(int i = 0; i < NUM_WATCH_POINTS; i++) {
        if(i == id) 
             write_channel_altera(watch_cmd_c[i], (debug_command_e) cmd);
    }

    if(cmd == D_READ) { 
        for(int read_count = 0; read_count < WATCH_SAMPLE_DEPTH+2; read_count++) { 
            #pragma unroll
            for(int i = 0; i < NUM_WATCH_POINTS; i++) {
                if(i == id) {
                    //printf("Read Out = %0d addr %0d tag = %0d \n", read_count, output[read_count].addr, output[read_count].tag);
                    output[read_count] = read_channel_altera(watch_out_c[i]);
                    mem_fence(CLK_CHANNEL_MEM_FENCE);
                }
            }
        }
    }
}
#endif //NUM_WATCH_POINTS

#endif //DEBUG__CL
