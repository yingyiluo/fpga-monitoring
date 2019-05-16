#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;

// quartus version
int version = 16;
// CL binary name
const char *binary_prefix = "mirror_nodebug";
// The set of simultaneous kernels
enum KERNELS {
  K_MIRROR,
  K_NUM_KERNELS
};
static const char *kernel_names[K_NUM_KERNELS] =
{
  "mirror_content"
};

// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queues[K_NUM_KERNELS];
static cl_kernel kernels[K_NUM_KERNELS];
static cl_program program = NULL;
static cl_int status = 0;

// debug interface
cl_kernel*        debug_kernel;
cl_command_queue*  debug_queue;
stamp_t*           time_stamp;
watch_s*           watch_points;

// Function prototypes
void test(long N, long M);
bool init();
void cleanup();

// Host memory buffers
int *h_outData;
int *out_copy;
// Device memory buffers
cl_mem d_outData;

int main(int argc, char **argv) {
  long N = (1 << 28); //16M
  long M = 2000;
  Options options(argc, argv);

  if(options.has("v")) {
    version = options.get<int>("v");
  }

  if(options.has("n")) {
    N = options.get<long>("n");
  }
  if(options.has("m")) {
    M = options.get<long>("m");
  }
  printf("Number of elements in the array is set to %ld\n", N);
  printf("Total data points to search is %ld\n", M);

  if (!init())
    return false;
  // init debug
  init_debug(context,program,device,&debug_kernel,&debug_queue);
  printf("Init complete!\n");

  // Allocate host memory
  h_outData = (int *)alignedMalloc(sizeof(int) * N);
  out_copy = (int *)alignedMalloc(sizeof(int) * N);
  if (!(h_outData)) {
    printf("ERROR: Couldn't create host buffers\n");
    return false;
  }

  // Test
  test(N, M);

  cleanup();
  return 0;
}

// Test channel
void test(long N, long M) {
  // Initialize input and produce verification data
  for (long i = 0; i < N; i++) {
    h_outData[i] = (int)i;
    out_copy[i] = (int)i;
  }

  // Create device buffers - assign the buffers in different banks for more efficient
  // memory access 
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  status = clEnqueueWriteBuffer(queues[K_MIRROR], d_outData, CL_TRUE, 0, sizeof(cl_int) * N, h_outData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");

  // Set the kernel arguments
  status = clSetKernelArg(kernels[K_MIRROR], 0, sizeof(cl_uint), (void*)&M);
  checkError(status, "Failed to set kernel_writer arg 0");
  status = clSetKernelArg(kernels[K_MIRROR], 1, sizeof(cl_mem), (void*)&d_outData);
  checkError(status, "Failed to set kernel_writer arg 1");

  double time = getCurrentTimestamp();
  cl_event kernel_event;
  //TODO: compare with clEnqueueNDRangeKernel when using channel
  // Write
  status = clEnqueueTask(queues[K_MIRROR], kernels[K_MIRROR], 0, NULL, NULL);
  checkError(status, "Failed to launch kernel_writer");
  //clGetProfileInfoIntelFPGA(kernel_event);
  //clWaitForEvents(1, &kernel_event);
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    status = clFinish(queues[i]);
    checkError(status, "Failed to finish (%d: %s)", i, kernel_names[i]);
  }

  // Record execution time
  time = getCurrentTimestamp() - time;

  // Copy results from device to host
  status = clEnqueueReadBuffer(queues[K_MIRROR], d_outData, CL_TRUE, 0, sizeof(cl_int) * N, h_outData, 0, NULL, NULL);
  checkError(status, "Failed to copy data from device");

#if NUM_DEBUG_POINTS > 0
	//Read timer output from device
	read_debug_all_buffers(context,program,debug_kernel,debug_queue,&time_stamp);
        print_debug(time_stamp);
        reset_debug_all_buffers(debug_kernel,debug_queue);
#endif //NUM_DEBUG_POINTS
  printf("\nVerifying\n");
  ulong cs = 0;
  for(int i = 1; i < M; i++) {
    out_copy[2*M - i] = out_copy[i];
  }
  
  for(int i; i < N; i++) {
    if(out_copy[i] != h_outData[i]) {
      printf("Verification Failed\n");
      break;
    }
  }
    printf("Verification Succeeded\n");
  printf("\nProcessing time = %.4fms\n", (float)(time * 1E3));
}

// Set up the context, device, kernels, and buffers...
bool init() {
  cl_int status;

  // Start everything at NULL to help identify errors
  for(int i = 0; i < K_NUM_KERNELS; ++i){
    kernels[i] = NULL;
    queues[i] = NULL;
  }

  // Locate files via. relative paths
  if(!setCwdToExeDir())
    return false;

  // Get the OpenCL platform.
  if(version == 16)
    platform = findPlatform("Altera");
  else
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find OpenCL platform\n");
    return false;
  }

  // Query the available OpenCL devices and just use the first device if we find
  // more than one
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;
  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queues
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    queues[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue (%d)", i);
  }

  // Create the program.
  std::string binary_file = getBoardBinaryFile(binary_prefix, device);
  printf("Using AOCX: %s\n\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  for(int i=0; i<K_NUM_KERNELS; ++i) {
    kernels[i] = clCreateKernel(program, kernel_names[i], &status);
    checkError(status, "Failed to create kernel (%d: %s)", i, kernel_names[i]);
  }

  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  for(int i=0; i<K_NUM_KERNELS; ++i)
    if(kernels[i]) 
      clReleaseKernel(kernels[i]);  
  if(program) 
    clReleaseProgram(program);
  for(int i=0; i<K_NUM_KERNELS; ++i)
    if(queues[i]) 
      clReleaseCommandQueue(queues[i]);
  if(context) 
    clReleaseContext(context);
}
