// Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

/*
 * Single Asian Option Pricing
 * Author: Deshanand Singh (dsingh@altera.com)
 *
 * This benchmark demonstrates an OpenCL implementation of an Asian Option Pricing Algorithm
 * implemented on Altera FPGAs.
 *
 * An Asian Option is a financial instrument whose price is path dependent. In this benchmark,
 * we consider the arithmetic Asian option. This option depends on the average of a number of
 * sampled point from the start time to the time of maturity. Upon maturity, the average price
 * is compared to the strike price for the computation of put or call.
 *
 * An excellent overview of these techniques can be found here:
 *    http://quantstart.com/articles/Asian-option-pricing-with-C-via-Monte-Carlo-Methods
 *
 * This OpenCL FPGA implementation consists of four parts:
 *
 * - Mersenne Twister Initialization - creates an initial state for the MT19937 random number generator
 *
 * - Mersenne Twister RNG Generation - produces a stream of 32-bit psuedorandom numbers
 *   A good overview can be found here: http://en.wikipedia.org/wiki/Mersenne_twister
 *
 * - "Black Scholes" - this kernel models the movement of the stock price using geometric brownian motion
 *   as described by the black scholes model. This kernel evaluates a number of paths (nr_sims) from an initial stock
 *   price (S_0). Each path is simulated through (N) time intervals where the average stock price is computed.
 *   This kernel is also multithreaded where each work-item runs its own batch of simulations. The total number
 *   of work-items is given by NUM_THREADS and launched using exactly ONE workgroup. The reason for this choice
 *   will become obvious as we dive into the implementation.
 *
 * - Accumulate Sums - each of the work-items above computes the total payoff across M paths. The accumulate
 *   sums kernel then sums each of these results together to produce a final reduction of these values and
 *   produce the total payoff over NUM_THREADS * nr_sims paths.
 *
 * An architectural view of the implementation is shown here:
 *
 *     Task                   Task               ND-range             Task
 * +----------------+      +------------+      +------------+      +------------+
 * | Mersenne       |      | Mersenne   |      | Black      |      | Accumulate |
 * | Twister        |----->| Twister    |----->| Scholes    |----->| Sums       |
 * | Initialization |      | Generation |      | Simulation |      |            |
 * +----------------+      +------------+      +------------+      +------------+
 *                  Channel             Channel             Channel
 *                INIT_STREAM         RANDOM_STREAM    ACCUMULATE_STREAM
 *
 * There are a few key things to notice:
 * - Channels are an Altera Vendor extension that allows for direct communication of data between
 *   kernels or between kernels and IO devices. This feature reduces the need to read and write
 *   intermediate data using global memory as in traditional OpenCL implementations. This drastically
 *   reduces the storage and bandwidth requirements for large financial simulations.
 *
 * - This application uses both OpenCL Tasks and ND-range kernels.
 *   - The ND-range is best used for large groups of completely independent parallel work-items that require
 *     little to no information sharing.
 *
 *   - In certain situations, it would be ideal for work-items to share information with each other
 *     without using barrier synchronization and local memory. This is not possible using the convential
 *     semantics of an ND-range kernel; however, a similar effect can be achieved by automatically pipelining
 *     loop iterations described in single work-item kernels. Recall that Altera's OpenCL kernel architecture is based
 *     on pipeline parallelism. Loop pipelining allows us to launch iterations of a loop as if they were parallel
 *     work-items. Dependencies between iterations are allowed as the generated hardware will ensure that
 *     a value computed in one iteration is available before it is read by a subsequent iteration. For example
 *     consider the psuedocode below for a typical random number generator:
 *
 *     __kernel __attribute((task)) void generate_rngs(int num_rnds)
 *     {
 *        vector state = initial_state;
 *        for (int i=0; i<num_rnds; i++) {
 *           state = next_state_function( state );
 *           unsigned u = extract_random_number( state );
 *           altera_write_channel(RANDOM_NUMBER_STREAM, u);
 *        }
 *     }
 *     In this kernel, it is difficult to use traditional ND-range techniques because there is a dependency
 *     between iterations since the line "state = next_state_function( state );" implies the state on the next
 *     iteration is a function of the previous iteration's state. In Altera's OpenCL implementation, the use
 *     of pipeline parallelism allows for extremely efficient execution of these types of kernels. The underlying
 *     implementation will look somewhat like the following:
 *
 *              +----------------------+
 *              |                      |
 *     +--------v-------+  +-----------^---------+
 *     | state register |  | next_state_function |
 *     +--------v-------+  +-----------^---------+
 *              |                      |
 *              +----------------------+
 *              |
 *     +--------v-------+
 *     | extract random |
 *     +--------v-------+
 *              |
 *     +--------v-------+
 *     | write channel  |
 *     +--------v-------+
 *
 *     On each clock cycle, the next state function is fed back to the state register. In many cases, this strategy
 *     is more efficient than considering any alternative style of parallelization. However, this must be used
 *     judiciously as there are some cases where the next state function is so complex that it may take multiple
 *     clock cycles for the function to be computed. In these cases the hardware will be underutilized and aoc will
 *     print a user message such as:
 *       Compiler Warning: Parallel execution at 25% of pipeline throughput
 *     In the case of random number generation, the state transition functions are usually simple bitwise operations
 *     that can be easily computed within a single cycle. It is interesting to note that in this application, we
 *     decompose the mersenne twister into an intiailization and generation kernel.
 *
 *     - The initialization kernel has a complex next state function, that slows down the rate at which the hardware
 *       can produce initial results. This is just fine for this application because the initialization is just a small
 *       fraction of the overall runtime.
 *
 *     - The generation kernel is functioning to generate random numbers on each clock cycle. This is very important
 *       since these random numbers feed the datapath which computes the movement of the stock prices.
 *
 *   - One of the interesting features of this benchmark is that random numbers are written to a channel from an
 *     OpenCL task and read from a channel in an OpenCL ND-range kernel. While there is no difficulty with doing
 *     this, there are ordering semantics that needs to be noted when ND-range multithreaded kernels are reading
 *     from a channel. Altera guarantees that OpenCL barriers enforce an an ascending order of work-items within a workgroup.
 *     At this point, there is no guarantee about the ordering of workgroups themselves and hence we use one large
 *     workgroup in this example. A barrier before the channel read forces an ordering that guarantees perfectly
 *     deterministic and repeatable results. Please note that the use of an OpenCL barrier to guarantee channel ordering is
 *     only required for kernels that have loops.
 *
 *   - The accumulate sums kernel is an OpenCL task that performs the final reduction. In this case, the loop dependency
 *     is "total_sum += partial_sum;". The total_sum and partial_sum variables are double precision variables. The addition
 *     of double precision numbers is relatively complex and is akin to having a complex next state function in the
 *     random number generation. As in the case of mersenne twister initialization, the final reduction executes at the
 *     end of the computation and consumes a small fraction of the overall runtime so processing the reductions at a lower
 *     rate will have no effect.
 *
 * Finally, we note that several online tools exist can be used to produce comparison points that verify these results
 * including:
 *    http://www.coggit.com/freetools
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#if defined WINDOWS
#include <windows.h>
#endif

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "common_defines.h"

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024

// Number of simulations per work-item. Command line can override
cl_ulong nr_sims = 100000;
// Number of time steps per simulation
#define N 256

// Risk free rate of return
#define RISK_FREE_RATE 0.08f

// Time horizon that we're interested in
#define TIME_HORIZON   1.0f

// Number of work-items running independent simulations of the asian option movement
// This must match the required work group size defined in the kernel.
#define NUM_THREADS 8192

// Maximum FPGAs supported in this application. Only limited by the number of boards
// in the system
#define MAX_DEVICES 16

// Name of the pre compiled binary resulting from running aoc to completion
#define PRECOMPILED_BINARY "asian_option"

bool use_cpu = false;

static cl_platform_id platform;
static cl_context my_context;

static cl_device_id device[MAX_DEVICES];
// A command queue for each kernel that is running in parallel
static cl_command_queue mersenne_init_queue[MAX_DEVICES];
static cl_command_queue mersenne_generate_queue[MAX_DEVICES];
static cl_command_queue black_scholes_queue[MAX_DEVICES];
static cl_command_queue accumulate_queue[MAX_DEVICES];

// Kernel objects for the mersenne intialization and generation, black scholes simulation and final accumulation
static cl_kernel mersenne_twister_init[MAX_DEVICES];
static cl_kernel mersenne_twister_generate[MAX_DEVICES];
static cl_kernel black_scholes[MAX_DEVICES];
static cl_kernel accumulate_sums[MAX_DEVICES];

static cl_program program;
static cl_int status;

cl_kernel*        debug_kernel;
cl_command_queue*  debug_queue;
stamp_t*           time_stamp;
watch_s*           watch_points;
std::string imageFilename;
std::string aocxFilename;
std::string deviceInfo;

// Device and host buffers for single result
#if USE_SVM_API == 0
static cl_mem kernel_result[MAX_DEVICES];
#else
static cl_double *kernel_result[MAX_DEVICES];
#endif /* USE_SVM_API == 0 */
static cl_double X[MAX_DEVICES];
static cl_double price[MAX_DEVICES];

// Event associated with black scholes kernel
static cl_event e[MAX_DEVICES];

double kernel_cpu(int nthreads, int m, int n, float drift, float vol, float S_0, float K);

// free the resources allocated during initialization
void cleanup() {
  if(program)
    clReleaseProgram(program);

  for (int i=0; i<MAX_DEVICES; i++) {
     if(kernel_result[i])
#if USE_SVM_API == 0
       clReleaseMemObject(kernel_result[i]);
#else
       clSVMFree(my_context, kernel_result[i]);
#endif /* USE_SVM_API == 0 */
     if(black_scholes[i])
       clReleaseKernel(black_scholes[i]);
     if(mersenne_twister_generate[i])
       clReleaseKernel(mersenne_twister_generate[i]);
     if(mersenne_twister_init[i])
       clReleaseKernel(mersenne_twister_init[i]);
     if(accumulate_sums[i])
       clReleaseKernel(accumulate_sums[i]);
     if(black_scholes_queue[i])
       clReleaseCommandQueue(black_scholes_queue[i]);
     if(mersenne_generate_queue[i])
       clReleaseCommandQueue(mersenne_generate_queue[i]);
     if(mersenne_init_queue[i])
       clReleaseCommandQueue(mersenne_init_queue[i]);
     if(accumulate_queue[i])
       clReleaseCommandQueue(accumulate_queue[i]);
   }
  if(my_context)
    clReleaseContext(my_context);
}

void asian_option_computation_cpu (
   int device_id,
   int m, int n,
   float sigma, float r,
   float T, float K, float S_0)
{
   float delta_t = T / n;
   float drift = exp(delta_t * (r - 0.5 * sigma * sigma));
   float vol = sigma * sqrt(delta_t);

   X[device_id] = kernel_cpu(NUM_THREADS, m, n, drift, vol, S_0, K);
}

// In the case of multiple FPGAs in the system, we'll use each to evaluate a different options
// in parallel for the purposes of demonstration.
//
void launch_asian_option_computation(
   int device_id,
   int m, int n,
   float sigma, float r,
   float T, float K, float S_0)
{
   print_monitor(stdout);
   printf("launch_asian_option@%f.\n", getCurrentTimestamp());
   // Precompute parameters on the host
   cl_float delta_t = T / n;
   cl_float drift   = (cl_float) (exp(delta_t*(r - 0.5*sigma*sigma)));
   cl_float vol     = (cl_float) (sigma * sqrt(delta_t));

   // Set the mersene twister generaate kernel parameters
   // This is the total number of random numbers that need to be generated
   const cl_ulong total_rnds = (nr_sims*(cl_ulong)N*(cl_ulong)NUM_THREADS);
   status = clSetKernelArg(mersenne_twister_generate[device_id], 0, sizeof(cl_ulong), (void*)&total_rnds);
   checkError(status,"mersenne_twister_generate: Failed set arg 0.");

   // Set the black scholes kernel parameters
   status = clSetKernelArg(black_scholes[device_id], 0, sizeof(cl_int), (void*)&m);
   checkError(status,"black_scholes: Failed set arg 0.");

   status = clSetKernelArg(black_scholes[device_id], 1, sizeof(cl_int), (void*)&n);
   checkError(status,"black_scholes: Failed Set arg 1.");

   status = clSetKernelArg(black_scholes[device_id], 2, sizeof(cl_float), (void*)&drift);
   checkError(status,"black_scholes: Failed Set arg 2.");

   status = clSetKernelArg(black_scholes[device_id], 3, sizeof(cl_float), (void*)&vol);
   checkError(status,"black_scholes: Failed Set arg 3.");

   status = clSetKernelArg(black_scholes[device_id], 4, sizeof(cl_float), (void*)&S_0);
   checkError(status,"black_scholes: Failed Set arg 4.");

   status = clSetKernelArg(black_scholes[device_id], 5, sizeof(cl_float), (void*)&K);
   checkError(status,"black_scholes: Failed Set arg 5.");

   // Set the accumulate sums kernel parameters
   // This is a single location in memory where the result is written back
#if USE_SVM_API == 0
   status = clSetKernelArg(accumulate_sums[device_id], 0, sizeof(cl_mem), (void*)&kernel_result[device_id]);
#else
   status = clSetKernelArgSVMPointer(accumulate_sums[device_id], 0, (void*)kernel_result[device_id]);
#endif /* USE_SVM_API == 0 */
   checkError(status,"accumulate_sums: Failed set arg 0.");

   // Launch the four kernels
   // 1. Mersenne Twister Initialization
   status = clEnqueueTask(mersenne_init_queue[device_id], mersenne_twister_init[device_id], 0, NULL, NULL);
   checkError(status,"mersenne_twister_init: Failed to launch kernel.");

   // 2. Mersenne Twister Generation
   status = clEnqueueTask(mersenne_generate_queue[device_id], mersenne_twister_generate[device_id], 0, NULL, NULL);
   checkError(status,"mersenne_twister_generate: Failed to launch kernel.");

   // 3. Black Scholes Computation
   const size_t local_size  = NUM_THREADS;
   const size_t global_size = NUM_THREADS;
   status = clEnqueueNDRangeKernel(black_scholes_queue[device_id], black_scholes[device_id], 1, NULL, &global_size, &local_size, 0, NULL, &e[device_id]);
   checkError(status,"black_scholes: Failed to launch kernel.");

   // 4. Accumulate Final Result
   status = clEnqueueTask(accumulate_queue[device_id], accumulate_sums[device_id], 0, NULL, NULL);
   checkError(status,"accumulate_sums: Failed to launch kernel.");
#if USE_SVM_API == 1
   clFinish(accumulate_queue[device_id]);
#endif /* USE_SVM_API == 1 */

#if NUM_DEBUG_POINTS > 0
        //Read timer output from device
        printf("Read the timers\n");
        printf("main, num_debug_points ");
        printf(" %d\n", NUM_DEBUG_POINTS);
        read_debug_all_buffers(my_context,program,debug_kernel,debug_queue,&time_stamp);
       // print_debug(time_stamp);
       // reset_debug_all_buffers(debug_kernel,debug_queue);
#endif 
   // For safety, but not really necessary. Ensure that all the accelerators have launched.
   clFlush(mersenne_init_queue[device_id]);
   clFlush(mersenne_generate_queue[device_id]);
   clFlush(black_scholes_queue[device_id]);
   clFlush(accumulate_queue[device_id]);
}

double get_result(int device_id)
{
   cl_event finish_event;
   printf("get_result@%f.\n", getCurrentTimestamp());
#if USE_SVM_API == 0
   // Read back the single result from the kernel
   status = clEnqueueReadBuffer(accumulate_queue[device_id], kernel_result[device_id], CL_FALSE, 0, sizeof(cl_double), &X[device_id], 0, NULL, &finish_event);
   checkError(status,"Failed to enqueue buffer kernel_result.");

   double fpga_sum = X[device_id];
#else
   status = clEnqueueSVMMap(accumulate_queue[device_id], CL_FALSE, CL_MAP_READ,
       (void *)kernel_result[device_id], sizeof(double), 0, NULL, &finish_event);
   checkError(status, "Failed to map kernel_result[%d]", device_id);
#endif /* USE_SVM_API == 0 */
   printf("after get_result@%f.\n", getCurrentTimestamp());
   monitor_and_finish(accumulate_queue[device_id], finish_event, stdout);

   // Compute the discounted price
   double num_sims = (double)nr_sims*(double)NUM_THREADS;
#if USE_SVM_API == 0
   double avg_price = fpga_sum / num_sims;
#else
   double avg_price = *kernel_result[device_id] / num_sims;
   status = clEnqueueSVMUnmap(accumulate_queue[device_id], (void *)kernel_result[device_id], 0, NULL, NULL);
   checkError(status, "Failed to unmap kernel_result[%d]", device_id);
#endif /* USE_SVM_API == 0 */
   return exp(-RISK_FREE_RATE*TIME_HORIZON)*avg_price;
}

int main(int argc, char **argv) {
  Options options(argc, argv);
  cl_uint num_devices;

  if(!setCwdToExeDir()) {
    return false;
  }

  if(options.has("sims")) {
    nr_sims = options.get<unsigned>("sims");
    printf("Number of simulations is set to %ld\n", nr_sims);
  }

  if (options.has("cpu")) {
    use_cpu = true;
    printf("Using CPU.\n");
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return -1;
  }

  // User-visible output - Platform information
  {
    char char_buffer[STRING_BUFFER_LEN];
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }

  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, MAX_DEVICES, &device[0], &num_devices);
  checkError(status,"Failed clGetDeviceIDs.");

  // create a context
  my_context = clCreateContext(0, num_devices, &device[0], &oclContextCallback, NULL, &status);
  checkError(status,"Failed clCreateContext.");

  print_monitor(stdout);

  // Create command queues and memory buffers per device
  for (unsigned i=0; i<num_devices; i++) {
    // create a command queue
    black_scholes_queue[i] = clCreateCommandQueue(my_context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status,"Failed clCreateCommandQueue : black_scholes_queue");

    // create a command queue
    mersenne_generate_queue[i] = clCreateCommandQueue(my_context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status,"Failed clCreateCommandQueue : mersenne_generate_queue");

    // create a command queue
    mersenne_init_queue[i] = clCreateCommandQueue(my_context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status,"Failed clCreateCommandQueue : mersenne_init_queue");

    // create a command queue
    accumulate_queue[i] = clCreateCommandQueue(my_context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status,"Failed clCreateCommandQueue : accumulate_queue");

    // create the output buffer
#if USE_SVM_API == 0
    kernel_result[i] = clCreateBuffer(my_context, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &status);
    checkError(status,"Failed clCreateBuffer.");
#else
    cl_device_svm_capabilities caps = 0;

    status = clGetDeviceInfo(
      device[i],
      CL_DEVICE_SVM_CAPABILITIES,
      sizeof(cl_device_svm_capabilities),
      &caps,
      0
    );
    checkError(status, "Failed to get device info");

    if (!(caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
      printf("The host was compiled with USE_SVM_API, however the device currently being targeted does not support SVM.\n");
      // Free the resources allocated
      cleanup();
      return -1;
    }

    kernel_result[i] = (cl_double *)clSVMAlloc(my_context, CL_MEM_READ_WRITE, sizeof(cl_double), 0);
    if (!kernel_result[i]) {
      printf("Can't allocate memory\n");
      // Free the resources allocated
      cleanup();
      return -1;
    }
#endif /* USE_SVM_API == 0 */
  }

  printf("Programming Device(s)\n");

  // Create the program.
  std::string binary_file = getBoardBinaryFile(PRECOMPILED_BINARY, device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(my_context, binary_file.c_str(), &device[0], num_devices);

  // Create the kernels per device
  for (unsigned i=0; i<num_devices; i++) {
    black_scholes[i] = clCreateKernel(program, "black_scholes", &status);
    checkError(status,"Failed clCreateKernel : black_scholes");

    accumulate_sums[i] = clCreateKernel(program, "accumulate_partial_results", &status);
    checkError(status,"Failed clCreateKernel : accumulate_partial_results");

    mersenne_twister_generate[i] = clCreateKernel(program, "mersenne_twister_generate", &status);
    checkError(status,"Failed clCreateKernel : mersenne_twister_generate");

    mersenne_twister_init[i] = clCreateKernel(program, "mersenne_twister_init", &status);
    checkError(status,"Failed clCreateKernel : mersenne_twister_init");
  }
  // init debug
  init_debug(my_context,program,device[0],&debug_kernel,&debug_queue);
  // These are just example parameters for the sake of demonstration.
  float sigma = 0.3f, strike_price=29.0f, initial_price = 30.0f;

  printf("Starting Computations\n");
  double start = getCurrentTimestamp() * 1.0e+9;
  if (use_cpu) {
    for (unsigned i = 0; i < num_devices; i++)
      asian_option_computation_cpu(i, nr_sims, N, sigma, RISK_FREE_RATE, TIME_HORIZON, (strike_price-i), initial_price );
    double num_sims = (double)nr_sims*(double)NUM_THREADS;
    for (unsigned i = 0; i < num_devices; i++)
      price[i] = exp(-RISK_FREE_RATE * TIME_HORIZON) * X[i] / num_sims;
  } else {
    for (unsigned i=0; i<num_devices; i++) {
    // In the case of multiple devices, we submit a different problem to each such as looking at different strike prices
      launch_asian_option_computation(i, nr_sims, N, sigma, RISK_FREE_RATE, TIME_HORIZON, (strike_price-i), initial_price );
    }
    for (unsigned i=0; i<num_devices; i++) {
      price[i] = get_result(i);
    }
  }

  double end = getCurrentTimestamp() * 1.0e+9;
  for (unsigned i=0; i<num_devices; i++) {
    printf( "DEVICE %d: r=%.2f sigma=%.2f T=%.1f S0=%.1f K=%.1f : Resulting Price is %lf\n", i, RISK_FREE_RATE, sigma, TIME_HORIZON,
       initial_price, (strike_price-i), price[i]);
  }
  // Print out througput
  double diff = end-start;
  double number_of_sims = (double)nr_sims * (double)NUM_THREADS * (double)N * (double)num_devices;
  printf("%d Devices ran a total of %lg Simulations\n", num_devices, number_of_sims);
  printf("Total Time(sec) = %.4f\n", diff*1e-9);
  printf("Throughput = %.2lf Billion Simulations / second\n", number_of_sims/diff);
  cleanup();
  return 0;
}

