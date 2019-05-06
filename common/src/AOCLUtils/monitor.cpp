#include <sys/types.h>
#include <time.h>
#include <CL/opencl.h>
#include <aocl_mmd.h>
#include <unistd.h>
#include "AOCLUtils/opencl.h"
#include "AOCLUtils/monitor.h"

extern int aclnalla_pcie0_handle;
const char *boardname = "aclnalla_pcie0";

/**
  * The function displays the temperature and power values of the FPGA board 
  * using two API functions. The two function definitions can be found in the
  * Intel FPGA SDK for OpenCL Custom Platform Toolkit User Guide.
  */
int print_monitor(FILE *f) {
	float temp, power;
	size_t retsize;
	double ts = aocl_utils::getCurrentTimestamp();
	aocl_mmd_get_info(aclnalla_pcie0_handle, AOCL_MMD_TEMPERATURE, sizeof(float), (void *)&temp, &retsize);
	aocl_mmd_card_info(boardname, AOCL_MMD_POWER, sizeof(float), (void *)&power, &retsize);
	return fprintf(f, "[%f] temp=%f, power=%f\n", ts, temp, power);
}

/**
  * The function displays the temperature and power values of the FPGA board every 
  * one second until the execution of the event is complete or fails.
  */
cl_int monitor_and_finish(cl_command_queue queue, cl_event event, FILE *f) {
	cl_int ret;
	size_t retsize;
	
	struct timespec tick;
	int sleep_ret = 0;
	clock_gettime(CLOCK_REALTIME, &tick);
	while (1) {
		if (sleep_ret == 0)
			print_monitor(f);
		clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &ret, &retsize);
		if (retsize != sizeof(cl_int) || ret == CL_COMPLETE) {
			break;
		} else {
			if (sleep_ret == 0) {
				tick.tv_sec++;
			}
			sleep_ret = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &tick, NULL);
		}
	}
	print_monitor(f);
	return CL_SUCCESS;
}

/**
  * This function opens the OpenCL program at the path ../../device and then reads it into 
  * the memory. From the memory, the API functions create a program object to build the 
  * program executable for the devices specified in the list.
  */
cl_program getCPUKernel(cl_context context, const char *kernel_name, const cl_device_id *device_list) {
	cl_int status;
	cl_program program;
	char *kernel_text;
	size_t text_size;
	FILE *fp;
	char fn[1024];

	snprintf(fn, 1024, "../../device/%s.cl", kernel_name);

	fp = fopen(fn, "r");
	if (!fp) {
		printf("Error: Failed to open: %s.\n", fn);
		return NULL;
	}
	
	fseek(fp, 0, SEEK_END);
	text_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	kernel_text = (char *)malloc(text_size + 1);
	fread(kernel_text, text_size, 1, fp);
	fclose(fp);
	kernel_text[text_size] = 0;

	program = clCreateProgramWithSource(context, 1, (const char **) &kernel_text, 0, &status);
	status = clBuildProgram(program, 1, device_list, "", NULL, NULL);
  if (status != CL_SUCCESS) {
    printf("Error: Failed to build program executable.\n");
    return NULL;
  }
	return program;
}
