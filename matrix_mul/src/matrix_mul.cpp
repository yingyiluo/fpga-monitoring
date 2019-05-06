/*******************************************************************************************
 * Author : Anshuman Verma 
 * Email  : anshuman@vt.edu
 * Date   : June 23rd, 2016 2:23PM EST 
 * Description : Simple Matrix Multiply kernel for checking the verilog dumps for BSPs
 ******************************************************************************************/
#include "matrix_mul.h"

int main(int argc, char *argv[]) { 
    Options cmd_line_opt(argc,argv); //Option Container
    string  aocx_file_name;          //AOCX File
    cl_int  row_a,col_a, col_b;

    //Usual suspects in OpenCL 
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_uint          num_devices;
    cl_kernel        kernel;
    cl_event         event;
    cl_uint          status;
    char             message[100];

    DATA_TYPE       *data_A;
    DATA_TYPE       *data_B;
    DATA_TYPE       *data_C;
    DATA_TYPE       *data_D;
    ulong           *timer_info;
    cl_mem           host_a,host_b,host_c;

//Debug Interface 
    cl_kernel*         debug_kernel;
    cl_command_queue*  debug_queue;
    stamp_t*           time_stamp;
    watch_s*           watch_points;



    //Resets the timer and starts new sample




    if(cmd_line_opt.has("h") | cmd_line_opt.has("help")) {
        help(argv[0]); 
        return __LINE__;
    }
    
    aocx_file_name = cmd_line_opt.get("x");
    row_a         = cmd_line_opt.get<cl_uint> ("m");
    col_a         = cmd_line_opt.get<cl_uint> ("n");
    col_b         = cmd_line_opt.get<cl_uint> ("p");

    PRINT_INFO("Operation A[" << row_a << "][" << col_a << \
                     "] X B[" << col_a << "][" << col_b << \
                     "] = C[" << row_a << "][" << col_b << "]" )

    PRINT_INFO("AOCX FILE NAME " << aocx_file_name)

    //Allocate Data buffers 
    data_A = (DATA_TYPE *) alignedMalloc(sizeof(DATA_TYPE)*row_a*col_a);
    data_B = (DATA_TYPE *) alignedMalloc(sizeof(DATA_TYPE)*col_a*col_b);
    data_C = (DATA_TYPE *) alignedMalloc(sizeof(DATA_TYPE)*row_a*col_b);
    data_D = (DATA_TYPE *) alignedMalloc(sizeof(DATA_TYPE)*row_a*col_b);
    
    //Fill random data into buffer
    fill_buffer(data_A,row_a*col_a);  
    fill_buffer(data_B,col_a*col_b);
    
    if(init_opencl(aocx_file_name,platform,device,context,program,num_devices) == false)
        return __LINE__;
  
    //Initialize debug
    init_debug(context,program,device,&debug_kernel,&debug_queue);

    if(!create_kernel("matrix_multiply",context,program,device,queue,kernel))
        return __LINE__;

#ifdef EMULATOR 
    cl_kernel        pipe_k;
    cl_command_queue pipe_q;

    if(!create_kernel("r_pipe",context,program,device,pipe_q,pipe_k))
        return __LINE__;

    status = clEnqueueTask(pipe_q,pipe_k,0,NULL,&event);
    sprintf(message,"Could not enqueue Kernel %s", "r_pipe");

    checkError(status,message);

#endif 

    //Copy the data buffers to device
    if(!write_device_buffer(queue,context,host_a,data_A,row_a*col_a,"Buffer A")) return __LINE__;
    if(!write_device_buffer(queue,context,host_b,data_B,col_a*col_b,"Buffer B")) return __LINE__;


    if(!create_or_read_device_buffer(queue,context,host_c,data_C,
                                     sizeof(DATA_TYPE)*row_a*col_b,"Buffer C",true)) return __LINE__;

    //for(int i = 0; i < row_a*col_a; i++)
    //    printf("A[%0d][%0d] = %0d\n",i/col_a,i%col_a,data_A[i]);

    //Wait for data transfer to be completed
    clFinish(queue);
    
    PRINT_INFO("Copied buffer to memory")
    
    //Set the arguments for the kernel
    status  = clSetKernelArg(kernel,0,sizeof(cl_mem),  &host_a);
    status |= clSetKernelArg(kernel,1,sizeof(cl_mem),  &host_b);
    status |= clSetKernelArg(kernel,2,sizeof(cl_mem),  &host_c);
    status |= clSetKernelArg(kernel,3,sizeof(cl_int),  &row_a);
    status |= clSetKernelArg(kernel,4,sizeof(cl_int),  &col_a);
    status |= clSetKernelArg(kernel,5,sizeof(cl_int),  &col_b);

    PRINT_INFO("Completed setting the Args")

    for(int repeat_count = 0; repeat_count < 1; repeat_count++) { 
        //Launch the kernel for computation
        status = clEnqueueTask(queue,kernel,0,NULL,&event);
        sprintf(message,"Could not enqueue Kernel %s", "matrix_multiply");
        checkError(status,message); 
        clFinish(queue);
        
        PRINT_INFO("Reading The Timers")
#if NUM_DEBUG_POINTS > 0
        //Read timer output from device
        PRINT_INFO("Read the timers")
        PRINT_INFO("main, num_debug_points");
        PRINT_INFO( NUM_DEBUG_POINTS);
        read_debug_all_buffers(context,program,debug_kernel,debug_queue,&time_stamp);
        print_debug(time_stamp);
        reset_debug_all_buffers(debug_kernel,debug_queue);
#endif //NUM_DEBUG_POINTS

#if NUM_WATCH_POINTS > 0
        PRINT_INFO("Read The Watch")
        read_watch_all_buffers(context,debug_kernel,debug_queue,&watch_points);
        print_watch(watch_points);
#endif 
    }

    //Read the results from the device back to host
    if(!create_or_read_device_buffer(queue,context,host_c,data_C,
                                     sizeof(DATA_TYPE)*row_a*col_b,"Buffer C",false)) return __LINE__;
    clFinish(queue);

    matrix_multiply(data_A,data_B,data_D,row_a,col_a,col_b);
    //Do a serial multiplication
    if(!compare_result(data_C,data_D,row_a*col_b)) {
        PRINT_ERROR("Compare Failed")
        return __LINE__;
    }
    else 
        PRINT_INFO("Result from Accelerator is correct")
    


    return 0;

}

bool compare_result(DATA_TYPE *data_acc
                   ,DATA_TYPE *data_host
                   ,cl_uint    size) { 

    bool result = true;
    for(cl_uint i = 0; i < size; i++) { 
        if(data_acc[i] != data_host[i]) {
            PRINT_ERROR("Data Mismatch EXP: " << data_host[i] << " ACT: " << data_acc[i] << " Location : " << i)
            result = false;
        }
    }
    return result;
}

bool write_device_buffer(cl_command_queue queue
                        ,cl_context       context
                        ,cl_mem          &host_buffer
                        ,DATA_TYPE       *device_buffer
                        ,cl_uint          size
                        ,const char      *info) { 
    cl_int status;
    char   message[100];
    host_buffer = clCreateBuffer(context
                                ,CL_MEM_WRITE_ONLY
                                ,sizeof(DATA_TYPE)*size
                                ,NULL
                                ,&status);

    sprintf(message,"Could not create buffer %s", info);
    checkError(status,message); 

    status = clEnqueueWriteBuffer(queue
                                 ,host_buffer
                                 ,CL_FALSE   //FIXME
                                 ,0
                                 ,sizeof(DATA_TYPE)*size
                                 ,device_buffer
                                 ,0,NULL,NULL);

    sprintf(message,"Could not write buffer %s", info);
    checkError(status,message); 
    return status == CL_SUCCESS;
}

bool create_or_read_device_buffer(
                         cl_command_queue queue
                        ,cl_context       context
                        ,cl_mem          &host_buffer
                        ,void            *device_buffer
                        ,size_t           size
                        ,const char      *info
                        ,bool             create_not_write=false) { 
    cl_int status;
    char   message[100];
    if(create_not_write) {
        host_buffer = clCreateBuffer(context
                                    ,CL_MEM_READ_ONLY
                                    ,size
                                    ,NULL
                                    ,&status);

        sprintf(message,"Could not create buffer %s", info);
        checkError(status,message); 
    }
    else {
        status = clEnqueueReadBuffer (queue
                                     ,host_buffer
                                     ,CL_FALSE   //FIXME
                                     ,0
                                     ,size
                                     ,device_buffer
                                     ,0,NULL,NULL);

        sprintf(message,"Could not write buffer %s", info);
        checkError(status,message); 
    }
    return status == CL_SUCCESS;
}

void fill_buffer(DATA_TYPE *buffer
                ,cl_uint    size) {
    for(cl_uint i = 0 ; i < size; i++) 
            buffer[i] = i;

}

bool init_opencl(const std::string &aocx 
                ,cl_platform_id    &platform
                ,cl_device_id      &dev
                ,cl_context        &context
                ,cl_program        &program
                ,cl_uint           &num_devices ) {

    cl_int          status;
    char            platform_name[20]="Altera";
    std::string     board_binary;
    cl_device_id   *device;

    if(setCwdToExeDir() == false)
        return false;

    platform = findPlatform(platform_name);

    if(platform == NULL) {
        PRINT_ERROR("Could NOT find the platform " << platform_name)
        return false;
    }
    else 
        PRINT_INFO("Building for " << getPlatformName(platform))

    device = getDevices(platform,CL_DEVICE_TYPE_ALL,&num_devices);
    PRINT_INFO("Found " << num_devices << " device(s) in " << platform_name << " listed below ")

    for(cl_uint no_dev = 0; no_dev < num_devices; no_dev++) {
        std::cout << "\t\t |- Device: [" << no_dev << "]  = " << getDeviceName(device[no_dev]) << std::endl;
    }

    if(!num_devices)
        PRINT_ERROR("No devices found: Check whether  machine has boards installed")

    board_binary = getBoardBinaryFile(aocx.c_str(),device[0]);
    PRINT_INFO("Using binary file " << board_binary)

    context = clCreateContext(NULL
                             ,num_devices
                             ,device
                             ,NULL
                             ,NULL
                             ,&status);
	checkError(status, "Could not create OpenCL context");
    
    program = createProgramFromBinary(context
                                     ,board_binary.c_str()
                                     ,device
                                     ,num_devices);

    status = clBuildProgram(program
                           ,num_devices
                           ,device
                           ,"",NULL,NULL);
	checkError(status, "Could not build Program");
    dev = device[0];
    return true;
    

}

bool create_kernel(const std::string  name
                  ,cl_context         context
                  ,cl_program         program
                  ,cl_device_id       device
                  ,cl_command_queue  &queue
                  ,cl_kernel         &kernel) {
    cl_int status;
    char   message[100];
    
    PRINT_INFO("Creating Kernel " << name )
    queue = clCreateCommandQueue(context
                                ,device
                                ,CL_QUEUE_PROFILING_ENABLE
                                ,&status);
    sprintf(message,"Could not create command queue for Kernel %s", name.c_str());
    checkError(status,message); 

#ifdef DEBUG
    //Temporary Experiment: Builds all the kernels in the file, and prints out the name of kernels
    //in the aocx file and number of arguments each kernel has. It can return a handle to pointer of
    //kernels. Neat way to build all the kernels in file.
    cl_uint   number_of_kernels;
    cl_kernel *kernels;
    status = clCreateKernelsInProgram(program,0,NULL,&number_of_kernels);
    sprintf(message,"Could not find any kernels in %s", name.c_str());
    checkError(status,message); 
    
    PRINT_INFO("Available Kernels in Binary : " << number_of_kernels)
    
    kernels = (cl_kernel *) malloc(sizeof(cl_kernel)*number_of_kernels);
    status = clCreateKernelsInProgram(program
                                     ,number_of_kernels
                                     ,kernels
                                     ,NULL);
    sprintf(message,"Could not build kernels in %s", name.c_str());
    checkError(status,message); 

    for(cl_uint i = 0; i < number_of_kernels ; i++) { 
    char   *kernel_name;
    size_t  kernel_name_size;
    cl_uint kernel_arg_size;
        status = clGetKernelInfo(kernels[i]
                                ,CL_KERNEL_FUNCTION_NAME ,0 , NULL, &kernel_name_size);

        sprintf(message,"Could not get info on kernels in %s", name.c_str());
        checkError(status,message); 
        
        kernel_name = (char *) malloc(sizeof(char) * kernel_name_size);

        status = clGetKernelInfo(kernels[i]
                                ,CL_KERNEL_FUNCTION_NAME 
                                ,sizeof(char)*kernel_name_size
                                ,kernel_name
                                ,NULL );
        sprintf(message,"Could not get info on kernels in %s", name.c_str());
        checkError(status,message); 


        status = clGetKernelInfo(kernels[i]
                                ,CL_KERNEL_NUM_ARGS 
                                ,sizeof(cl_uint)
                                ,&kernel_arg_size
                                ,NULL );
        sprintf(message,"Could not get info on kernels in %s", name.c_str());
        checkError(status,message); 

        PRINT_INFO("KERNEL_NAME : " << kernel_name << " has " << kernel_arg_size << " arguments")


    }

#endif //DEBUG
    kernel = clCreateKernel(program
                           ,name.c_str()
                           ,&status);
    sprintf(message,"Could not create  Kernel %s", name.c_str());
    checkError(status,message); 

    return true;

}
    
void matrix_multiply(DATA_TYPE*  data_a
                    ,DATA_TYPE*  data_b
                    ,DATA_TYPE*  data_c
                    ,int row_a
                    ,int col_a
                    ,int col_b
                    ) {

    for(int i = 0; i < row_a ; i++) { 
        for(int j = 0; j < col_b; j++) { 
         DATA_TYPE acc=0;
            for(int k = 0; k < col_a; k++) { 
                acc += data_a[i*col_a + k] * data_b[k*col_b + j];
            }
            data_c[i*col_b + j] = acc;
        }
    }
}

void help(char *argv) { 
    printf("\n\n\n\t%s -x <aocx file name> -h/help\n\n\n", argv);
}
