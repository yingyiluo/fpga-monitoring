#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "XSbench_header.h"

using namespace aocl_utils;

// CL binary name
//const char *binary_prefix = "Simulation_cachebs_pipe_grid";
const char *binary_prefix = "debug";
// The set of simultaneous kernels
enum KERNELS {
  K_SIMULATION,
  K_GRIDSEARCH,
  K_CAL_MACRO_XS,
  K_ACCU_MACRO_XS,
  K_NUM_KERNELS
};
static const char *kernel_names[K_NUM_KERNELS] =
{
  "simulation",
  "grid_search",
  "calculate_macro_xs",
  "accumulate_macro_xs"
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

int NUM_STAGE = 10;
int num_points;
BSCache *h_inCache = NULL;
cl_mem d_num_nucs, d_concs, d_energy, d_energy_grid_array, d_nuclide_grids, d_mats, d_inCache, d_vhash, d_energy_grid_xs;
int *energy_grid_xs;
cl_double16 *lh_nu_grids;
int main( int argc, char* argv[] )
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	int version = 18;
	int mype = 0;
	//unsigned long vhash = 0;
	int nprocs = 1;

	// rand() is only used in the serial initialization stages.
	// A custom RNG is used in parallel portions.
	#ifdef VERIFICATION
	srand(26);
	#else
	srand(time(NULL));
	#endif

	// Process CLI Fields -- store in "Inputs" structure
	Inputs in = read_CLI( argc, argv );

	// Print-out of Input Summary
	if( mype == 0 )
		print_inputs( in, nprocs, version );

	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// =====================================================================

	// Allocate & fill energy grids
	#ifndef BINARY_READ
	if( mype == 0) printf("Generating Nuclide Energy Grids...\n");
	#endif

	NuclideGridPoint ** nuclide_grids = gpmatrix(in.n_isotopes,in.n_gridpoints);
	
	generate_grids( nuclide_grids, in.n_isotopes, in.n_gridpoints );	

	// Sort grids by energy
	#ifndef BINARY_READ
	if( mype == 0) printf("Sorting Nuclide Energy Grids...\n");
	sort_nuclide_grids( nuclide_grids, in.n_isotopes, in.n_gridpoints );
	#endif

	// If using a unionized grid search, initialize the energy grid
	// Otherwise, leave these as null
	GridPoint * energy_grid = NULL;
	//int * index_data = NULL;

	if( in.grid_type == UNIONIZED )
	{
		// Prepare Unionized Energy Grid Framework
		#ifndef BINARY_READ
		energy_grid = generate_energy_grid( in.n_isotopes,
				in.n_gridpoints, nuclide_grids ); 	
		#else
		energy_grid = (GridPoint *)malloc( in.n_isotopes *
				in.n_gridpoints * sizeof( GridPoint ) );
		index_data = (int *) malloc( in.n_isotopes * in.n_gridpoints
				* in.n_isotopes * sizeof(int));
		for( int i = 0; i < in.n_isotopes*in.n_gridpoints; i++ )
			energy_grid[i].xs_ptrs = &index_data[i*in.n_isotopes];
		#endif

		// Double Indexing. Filling in energy_grid with pointers to the
		// nuclide_energy_grids.
		#ifndef BINARY_READ
		initialization_do_not_profile_set_grid_ptrs( energy_grid, nuclide_grids, in.n_isotopes, in.n_gridpoints );
		#endif
	}
	/*
	else if( in.grid_type == HASH )
	{
		energy_grid = generate_hash_table( nuclide_grids, in.n_isotopes, in.n_gridpoints, in.hash_bins );
	}
	*/
	#ifdef BINARY_READ
	if( mype == 0 ) printf("Reading data from \"XS_data.dat\" file...\n");
	binary_read(in.n_isotopes, in.n_gridpoints, nuclide_grids, energy_grid, in.grid_type);
	#endif

	// Get material data
	if( mype == 0 )
		printf("Loading Mats...\n");
	int *num_nucs  = load_num_nucs(in.n_isotopes);
	int **mats     = load_mats(num_nucs, in.n_isotopes);

	double **concs = load_concs(num_nucs);

	#ifdef BINARY_DUMP
	if( mype == 0 ) printf("Dumping data to binary file...\n");
	binary_dump(in.n_isotopes, in.n_gridpoints, nuclide_grids, energy_grid, in.grid_type);
	if( mype == 0 ) printf("Binary file \"XS_data.dat\" written! Exiting...\n");
	return 0;
	#endif

	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation
	// =====================================================================
	if( mype == 0 )
	{
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

	long n_iso_grid = in.n_isotopes * in.n_gridpoints;
	//double *energy = (double *) alignedMalloc(n_iso_grid * sizeof(double));
	energy_grid_xs = (int *) alignedMalloc(n_iso_grid * in.n_isotopes * sizeof(int));
	GridPoint_Array *energy_grid_array = (GridPoint_Array *) alignedMalloc(n_iso_grid * sizeof(GridPoint_Array));
	num_points = 0;
	for (int i = 0; i < NUM_STAGE; i++)
		num_points += pow(2, i);
	h_inCache = (BSCache *) alignedMalloc(sizeof(BSCache) * num_points);

	int sample = 0;
	double interval = (double)n_iso_grid/(num_points+1);
	for(int i = 0; i < n_iso_grid; i++){
		//energy[i] = energy_grid[i].energy;
		energy_grid_array[i].energy = energy_grid[i].energy;
		if( i == (int)((sample+1) * interval) )
		{
			h_inCache[sample].data = energy_grid[i].energy;
			h_inCache[sample].index = i;
			sample++;
		}
		int index = i * in.n_isotopes;
		for(int j = 0; j < in.n_isotopes; j++) {
			energy_grid_array[i].xs_ptrs[j] = (short)energy_grid[i].xs_ptrs[j];
			energy_grid_xs[index + j] = energy_grid[i].xs_ptrs[j];
		}
	}

	unsigned long *vhash = (unsigned long *) alignedMalloc(sizeof(unsigned long));
	if(!init())
		return false;
	// init debug
        init_debug(context,program,device,&debug_kernel,&debug_queue);
	printf("Init complete!\n");
	// Run simulation
	run_simulation_v2(in, energy_grid_array, energy_grid, nuclide_grids, num_nucs, mats, concs, vhash);
	cleanup();
	return 0;
}

void run_simulation_v2(Inputs in, GridPoint_Array *energy_grid_array,
		GridPoint *energy_grid,
		NuclideGridPoint **nuclide_grids, 
		int *num_nucs, int **mats, double **concs, 
		unsigned long *vhash)
{
	long n_iso_grid = in.n_isotopes * in.n_gridpoints;
	int total_nucs = 0;
	for(int i = 0; i < 12; i++)
		total_nucs += num_nucs[i];
  	// Create device buffers - assign the buffers in different banks for more efficient
  	// memory access 
	lh_nu_grids = (cl_double16*) alignedMalloc(sizeof(cl_double16) * n_iso_grid);
	cl_double8 *tmp = (cl_double8*) alignedMalloc(sizeof(cl_double8) * n_iso_grid);
	memcpy(tmp, *nuclide_grids, sizeof(cl_double8) * n_iso_grid);
	int i = 0;
	FILE *f0 = fopen("nu0.txt", "w");
	FILE *f1 = fopen("nu4.txt", "w");
	FILE *f2 = fopen("nu5.txt", "w");
	FILE *f3 = fopen("nu24.txt", "w");
	FILE *f4 = fopen("nu41.txt", "w");
	for(int a = 0; a < in.n_isotopes; a++) {
	   for(int b = 0; b < in.n_gridpoints; b++) {
		/*
		if (a == 0){
			fprintf(f0, "(double8) (%.24lf, %.24lf, %.24lf, %.24lf, %.24lf, %.24lf, 0, 0),\n", tmp[i].s0, tmp[i].s1, tmp[i].s2, tmp[i].s3, tmp[i].s4, tmp[i].s5);
		}		

		if (a == 4){
			fprintf(f1, "nu4[%d] = (cl_double8) (%.24lf, %.24lf, %.24lf, %.24lf, %.24lf, %.24lf, 0, 0);\n", b, tmp[i].s0, tmp[i].s1, tmp[i].s2, tmp[i].s3, tmp[i].s4, tmp[i].s5);
		}		

		if (a == 5){
			fprintf(f2, "(cl_double8) (%.24lf, %.24lf, %.24lf, %.24lf, %.24lf, %.24lf, 0, 0),\n", tmp[i].s0, tmp[i].s1, tmp[i].s2, tmp[i].s3, tmp[i].s4, tmp[i].s5);
		}		
		
		if (a == 24){
			fprintf(f3, "(cl_double8) (%.24lf, %.24lf, %.24lf, %.24lf, %.24lf, %.24lf, 0, 0),\n", tmp[i].s0, tmp[i].s1, tmp[i].s2, tmp[i].s3, tmp[i].s4, tmp[i].s5);
		}	
			
		if (a == 41){
			fprintf(f4, "(cl_double8) (%.24lf, %.24lf, %.24lf, %.24lf, %.24lf, %.24lf, 0, 0),\n", tmp[i].s0, tmp[i].s1, tmp[i].s2, tmp[i].s3, tmp[i].s4, tmp[i].s5);
		}*/		
		if(i == n_iso_grid - 1) {
			//lh_nu_grids[i].lo = (cl_double8) ((*nuclide_grids)[i-1].energy, (*nuclide_grids)[i-1].total_xs, (*nuclide_grids[i-1]).elastic_xs, (*nuclide_grids[i-1]).absorbtion_xs, (*nuclide_grids)[i-1].fission_xs, (*nuclide_grids)[i-1].nu_fission_xs, 0, 0);
			lh_nu_grids[i].lo = tmp[i-1];
                        lh_nu_grids[i].hi = tmp[i];
		}else{
			lh_nu_grids[i].lo = tmp[i];
			lh_nu_grids[i].hi = tmp[i+1];
		}
		i++;
	    }
	} 
	fclose(f0);	
	d_energy_grid_array = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, n_iso_grid * sizeof(GridPoint_Array), NULL, &status);
	checkError(status, "Failed to create input energy_grid_xs buffer. \n");

	d_nuclide_grids = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA, n_iso_grid * sizeof(cl_double16), NULL, &status);
	checkError(status, "Failed to create nuclide_grids input buffer.\n");

	d_vhash = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_BANK_2_ALTERA, sizeof(unsigned long), NULL, &status);
	checkError(status, "Failed to create output buffer.\n");

//	d_energy_grid_xs = clCreateBuffer(context, CL_MEM_READ_ONLY, n_iso_grid * in.n_isotopes * sizeof(cl_int), NULL, &status);
//	checkError(status, "Failed to create input energy_grid_xs buffer. \n");

//	status = clEnqueueWriteBuffer(queues[K_SIMULATION], d_inCache, CL_TRUE, 0, num_points * sizeof(BSCache), h_inCache, 0, NULL, NULL);
//	checkError(status, "Failed to enqueue write buffer.\n");

	status = clEnqueueWriteBuffer(queues[K_GRIDSEARCH], d_energy_grid_array, CL_TRUE, 0, n_iso_grid * sizeof(GridPoint_Array), energy_grid_array, 0, NULL, NULL);
	checkError(status, "Failed to enqueue write buffer.\n");

//	status = clEnqueueWriteBuffer(queues[K_CAL_MACRO_XS], d_concs, CL_TRUE, 0, total_nucs * sizeof(double), *concs, 0, NULL, NULL);
//	checkError(status, "Failed to enqueue write buffer.\n");

//	status = clEnqueueWriteBuffer(queues[K_CAL_MACRO_XS], d_energy_grid_xs, CL_TRUE, 0, n_iso_grid * in.n_isotopes * sizeof(cl_int), energy_grid_xs, 0, NULL, NULL);
//	checkError(status, "Failed to enqueue write buffer.\n");
 
	status = clEnqueueWriteBuffer(queues[K_CAL_MACRO_XS], d_nuclide_grids, CL_TRUE, 0, n_iso_grid * sizeof(cl_double16), lh_nu_grids, 0, NULL, NULL);
	checkError(status, "Failed to enqueue write buffer.\n");


	int arg = 0;
	status = clSetKernelArg(kernels[K_SIMULATION], arg++, sizeof(int), &in.lookups);
	checkError(status, "Failed to set arg 0");
	//status = clSetKernelArg(kernels[K_SIMULATION], arg++, sizeof(cl_mem), &d_inCache);
        //checkError(status, "Failed to set arg 1");

	arg = 0;
	status = clSetKernelArg(kernels[K_GRIDSEARCH], arg++, sizeof(int), &in.lookups);
	checkError(status, "Failed to set arg 0");
	status = clSetKernelArg(kernels[K_GRIDSEARCH], arg++, sizeof(cl_mem), &d_energy_grid_array);
	checkError(status, "Failed to set arg 1");

	arg = 0;
	status = clSetKernelArg(kernels[K_CAL_MACRO_XS], arg++, sizeof(int), &in.lookups);
	checkError(status, "Failed to set arg 0");
//	status = clSetKernelArg(kernels[K_CAL_MACRO_XS], arg++, sizeof(cl_mem), &d_concs);
//	checkError(status, "Failed to set arg 1");
//	status = clSetKernelArg(kernels[K_CAL_MACRO_XS], arg++, sizeof(cl_mem), &d_energy_grid_xs);
//	checkError(status, "Failed to set arg 2");
	status = clSetKernelArg(kernels[K_CAL_MACRO_XS], arg++, sizeof(cl_mem), &d_nuclide_grids);
	checkError(status, "Failed to set arg 3");

	arg = 0;
	status = clSetKernelArg(kernels[K_ACCU_MACRO_XS], arg++, sizeof(int), &in.lookups);
	checkError(status, "Failed to set arg 0");

	status = clSetKernelArg(kernels[K_ACCU_MACRO_XS], arg++, sizeof(cl_mem), &d_vhash);
	checkError(status, "Failed to set arg 1");

	// Record start time
	double time = getCurrentTimestamp();
	printf("Start simulation!\n");
	status = clEnqueueTask(queues[K_SIMULATION], kernels[K_SIMULATION], 0, NULL, NULL);
	checkError(status, "Failed to launch kernel simulation");
	status = clEnqueueTask(queues[K_GRIDSEARCH], kernels[K_GRIDSEARCH], 0, NULL, NULL);
	checkError(status, "Failed to launch kernel gridsearch");
	status = clEnqueueTask(queues[K_CAL_MACRO_XS], kernels[K_CAL_MACRO_XS], 0, NULL, NULL);
	checkError(status, "Failed to launch kernel cal_macro_xs");
	status = clEnqueueTask(queues[K_ACCU_MACRO_XS], kernels[K_ACCU_MACRO_XS], 0, NULL, NULL);
	checkError(status, "Failed to launch kernel accu_macro_xs");
	
  	for(int i=0; i<K_NUM_KERNELS; ++i) {
		status = clFinish(queues[i]);
		checkError(status, "Failed to finish (%d: %s)", i, kernel_names[i]);
	}
	printf("Reading The Timers\n");
#if NUM_DEBUG_POINTS > 0
	//Read timer output from device
	read_debug_all_buffers(context,program,debug_kernel,debug_queue,&time_stamp);
	print_debug(time_stamp);
	reset_debug_all_buffers(debug_kernel,debug_queue);
#endif //NUM_DEBUG_POINTS

	// Record execution time
	time = getCurrentTimestamp() - time;

	printf("\n" );
	printf("Simulation complete.\n" );

	// =====================================================================
	// Output Results & Finalize
	// =====================================================================
	status = clEnqueueReadBuffer(queues[K_ACCU_MACRO_XS], d_vhash, CL_TRUE, 0, sizeof(unsigned long), vhash, 0, NULL, NULL);
	checkError(status, "Failed to read buffer from kernel cal_vhash");

	// Final Hash Step
	*vhash = *vhash % 1000000;

	// Print / Save Results and Exit
	print_results( in, 0, time, 1, (unsigned long long)*vhash );

	#ifdef VERIFICATION
	printf("\nVerifying\n");
	unsigned long vhash_verify = 0;
	run_event_based_simulation(in, energy_grid, nuclide_grids, num_nucs, mats, concs, 0, &vhash_verify);
	vhash_verify = vhash_verify % 1000000;
	if(*vhash == vhash_verify)
		printf("Verification PASS.\n");
	else {
		printf("Verification FAIL.\n");
		printf("vhash_verify: %ld, vhash: %ld\n", vhash_verify, *vhash);
	}
	#endif

	printf("\nProcessing time = %.4fms\n", (float)(time * 1E3));
}

// Set up the context, device, kernels, and buffers...
bool init()
{
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
 // platform = findPlatform("Intel(R) FPGA");
  platform = findPlatform("Altera");
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
void cleanup()
{
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
