#ifndef DEBUG__H
#define DEBUG__H

#include <stdio.h>
#include <CL/opencl.h>
#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

#include "debug_defines.h"



/* Intiailize the timer infrastructure for debug and launch auto kernels if using emulator */
void init_debug(const cl_context         context
               ,const cl_program         program
               ,const cl_device_id       device
               ,      cl_kernel**        kernel
               ,      cl_command_queue** queue);


/* Read the sampled data into an array */
void read_debug(const cl_context         context
               ,const cl_program         program
               ,const cl_kernel*         kernel
               ,const cl_command_queue*  queue
               ,const cl_int             buffer_id
               ,stamp_t**                time_stamp);


/* Read the sampled data from all trace buffers into an array */
void read_debug_all_buffers(const cl_context         context
                           ,const cl_program         program
                           ,const cl_kernel*         kernel
                           ,const cl_command_queue*  queue
                           ,stamp_t**                time_stamp); 

/* Reset the sampler for next set of sample inputs */
void reset_debug(const cl_kernel*          kernel
                ,const cl_command_queue*   queue
                ,const cl_int              buffer_id); 


/* Reset all the trace buffers */
void reset_debug_all_buffers(const cl_kernel*        kernel
                            ,const cl_command_queue* queue);




/* Stop the trace buffer storing data*/
void stop_debug(const cl_kernel*          kernel
               ,const cl_command_queue*   queue
               ,const cl_int              buffer_id); 




/* Start storing time stamp data into trace buffer */
void start_debug(const cl_kernel*          kernel
                ,const cl_command_queue*   queue
                ,const cl_int              buffer_id); 





/* Start storing time stamp data into trace cyclic buffer */
void start_cyclic_debug(const cl_kernel*          kernel
                       ,const cl_command_queue*   queue
                       ,const cl_int              buffer_id);




/* Print the read timer values for all the channels */
void print_debug(const stamp_t* time_stamp); 


void read_watch(const cl_context         context
               ,const cl_kernel*         kernel
               ,const cl_command_queue*  queue
               ,const cl_int             watch_id
               ,watch_s**                watch_point);

void read_watch_all_buffers(const cl_context         context
                           ,const cl_kernel*         kernel
                           ,const cl_command_queue*  queue
                           ,watch_s**                watch_point); 


void print_watch(const watch_s* watch_point);

void reset_watch_all(const cl_kernel*        kernel
                    ,const cl_command_queue* queue); 


void control_watch(const cl_kernel*        kernel
                  ,const cl_command_queue* queue
                  ,const cl_int            watch_id
                  ,const debug_command_e   cmd); 

void reset_watch(const cl_kernel*          kernel
                ,const cl_command_queue*   queue
                ,const cl_int              watch_id);


void stop_watch(const cl_kernel*          kernel
               ,const cl_command_queue*   queue
               ,const cl_int              watch_id); 


void start_watch(const cl_kernel*          kernel
                ,const cl_command_queue*   queue
                ,const cl_int              watch_id);

void start_cyclic_watch(const cl_kernel*          kernel
                       ,const cl_command_queue*   queue
                       ,const cl_int              watch_id);



#endif //DEBUG__H
