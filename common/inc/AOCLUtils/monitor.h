#ifndef AOCL_UTILS_MONITOR_H
#define AOCL_UTILS_MONITOR_H

#include <stdio.h>
#include <CL/opencl.h>

int print_monitor(FILE *f);
cl_int monitor_and_finish(cl_command_queue queue, cl_event event, FILE *f);
cl_program getCPUKernel(cl_context context, const char *kernel_name, const cl_device_id *device_list);

#endif
