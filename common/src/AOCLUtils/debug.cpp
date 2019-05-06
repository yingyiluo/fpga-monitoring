#ifndef DEBUG_CPP
#define DEBUG_CPP

#include "AOCLUtils/debug.h"
#include "CL/opencl.h"

void init_debug(const cl_context         context
               ,const cl_program         program
               ,const cl_device_id       device
               ,      cl_kernel**        kernel
               ,      cl_command_queue** queue) {

    int    no_of_kernels;
    cl_int status;
#ifdef EMULATOR 
    #pragma message "compiling debug for emulator"
    char debug_kernel_names[ NUM_WATCH_POINTS + NUM_DEBUG_POINTS + 2 ][30];

    sprintf(debug_kernel_names[0],"read_stamp");
    sprintf(debug_kernel_names[1],"read_watch");
    for(int i = 2; i < NUM_DEBUG_POINTS + 2; i++) 
        sprintf(debug_kernel_names[i], "sample_stamp_%0d", i-2); 

    for(int i = NUM_DEBUG_POINTS + 2; i < 2 + NUM_DEBUG_POINTS + NUM_WATCH_POINTS ; i++) 
        sprintf(debug_kernel_names[i], "watch_addr_%0d", i - 2 - NUM_DEBUG_POINTS); 

    no_of_kernels = 2 + NUM_DEBUG_POINTS + NUM_WATCH_POINTS;
#else 
    const char* debug_kernel_names[] = {"read_stamp","read_watch"};
    no_of_kernels = 2;
#endif 
    *kernel =  (cl_kernel *)         malloc(sizeof(cl_kernel)        * no_of_kernels);
    *queue  =  (cl_command_queue *)  malloc(sizeof(cl_command_queue) * no_of_kernels);

#if NUM_DEBUG_POINTS > 0
    (*queue)[0] = clCreateCommandQueue(context,device,CL_QUEUE_PROFILING_ENABLE,&status);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not create queue in debug %0d\n", 0);
    }
    (*kernel)[0] = clCreateKernel(program,debug_kernel_names[0],&status);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not create kernel in debug %0d\n", debug_kernel_names[0]);
    }
#endif 

#if NUM_WATCH_POINTS > 0
    (*queue)[1] = clCreateCommandQueue(context,device,CL_QUEUE_PROFILING_ENABLE,&status);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not create queue in debug %1d\n", 1);
    }
    (*kernel)[1] = clCreateKernel(program,debug_kernel_names[1],&status);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not create kernel in debug %1d\n", debug_kernel_names[1]);
    }
#endif

#ifdef EMULATOR 
    for(int i = 2; i < 2 + NUM_DEBUG_POINTS + NUM_WATCH_POINTS ; i++) { 
        (*queue)[i] = clCreateCommandQueue(context,device,CL_QUEUE_PROFILING_ENABLE,&status);
        if(status != CL_SUCCESS) { 
            printf("-ERROR- Could not create queue in debug %0d\n", i);
        }
        (*kernel)[i] = clCreateKernel(program,debug_kernel_names[i],&status);
        if(status != CL_SUCCESS) { 
            printf("-ERROR- Could not create kernel in debug %0d\n", debug_kernel_names[i]);
        }
        status = clEnqueueTask((*queue)[i],(*kernel)[i],0,NULL,NULL);
        if(status != CL_SUCCESS) { 
            printf("-ERROR- Could not enqueue kernels in debug %s\n", debug_kernel_names[i]);
        }
        else 
            printf("-INFO- : Launched Kernel %s\n", debug_kernel_names[i]);
    }
#endif 

}

void reset_debug_all_buffers(const cl_kernel*        kernel
                            ,const cl_command_queue* queue) {

    for(int trace_buffer_id = 0; trace_buffer_id <  NUM_DEBUG_POINTS ; trace_buffer_id++) { 
        reset_debug(kernel,queue,trace_buffer_id);
    }
}

void control_buffer(const cl_kernel*        kernel
                   ,const cl_command_queue* queue
                   ,const cl_int            buffer_id
                   ,const debug_command_e   cmd) {

    cl_int status;
    cl_int cmd_in = (cl_int) cmd;

    status  = clSetKernelArg(kernel[0],0,sizeof(cl_int),&cmd_in);
    status |= clSetKernelArg(kernel[0],1,sizeof(cl_mem),NULL);
    status |= clSetKernelArg(kernel[0],2,sizeof(cl_int),&buffer_id); 

    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not set the kernel arguments");
    }

    status = clEnqueueTask(queue[0],kernel[0],0,NULL,NULL);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not Enqueue Kernel");
    }
    clFinish(queue[0]);
}


void reset_debug(const cl_kernel*          kernel
                ,const cl_command_queue*   queue
                ,const cl_int              buffer_id) { 

    control_buffer(kernel,queue,buffer_id,D_RESET);

}

void stop_debug(const cl_kernel*          kernel
               ,const cl_command_queue*   queue
               ,const cl_int              buffer_id) { 

    control_buffer(kernel,queue,buffer_id,D_STOP);

}

void start_debug(const cl_kernel*          kernel
                ,const cl_command_queue*   queue
                ,const cl_int              buffer_id) { 

    control_buffer(kernel,queue,buffer_id,D_RECORD);
}

void start_cyclic_debug(const cl_kernel*          kernel
                       ,const cl_command_queue*   queue
                       ,const cl_int              buffer_id) { 

    control_buffer(kernel,queue,buffer_id,D_RECORD_CYCLIC);
}


void read_debug(const cl_context         context
               ,const cl_program         program
               ,const cl_kernel*         kernel
               ,const cl_command_queue*  queue
               ,const cl_int             buffer_id
               ,stamp_t**                time_stamp) {

    cl_mem read_buffer;
    cl_int status;

    const cl_int mem_size = DEBUG_SAMPLE_DEPTH + 1;

    posix_memalign ((void **) time_stamp,64,sizeof(stamp_t)*mem_size);

    for(cl_uint i = 0; i < 1 ; i++) { 
    char   *kernel_name;
    size_t  kernel_name_size;
    cl_uint kernel_arg_size;
    char    message[200];
        status = clGetKernelInfo(kernel[i]
                                ,CL_KERNEL_FUNCTION_NAME ,0 , NULL, &kernel_name_size);

        sprintf(message,"Could not get info on kernel in %s", "read_stamp");
        if(status != CL_SUCCESS) printf("%s %d",message,status);   
     
        kernel_name = (char *) malloc(sizeof(char) * kernel_name_size);

        status = clGetKernelInfo(kernel[i]
                                ,CL_KERNEL_FUNCTION_NAME 
                                ,sizeof(char)*kernel_name_size
                                ,kernel_name
                                ,NULL );
        sprintf(message,"Could not get info on kernel in %s", kernel_name);
        if(status != CL_SUCCESS) printf("%s %d",message,status);   
        fflush(stdout);

        status = clGetKernelInfo(kernel[i]
                                ,CL_KERNEL_NUM_ARGS 
                                ,sizeof(cl_uint)
                                ,&kernel_arg_size
                                ,NULL );
        sprintf(message,"Could not get info on kernel in %s", kernel_name);
        if(status != CL_SUCCESS) printf("%s %d",message,status);   

        printf("Kernel %s has %0d args\n", kernel_name, kernel_arg_size);
        fflush(stdout);
    }


    read_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(stamp_t)*mem_size, NULL, &status);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not create read buffer in rest_debug\n");
    }
    
    status = clSetKernelArg(kernel[0],1,sizeof(cl_mem) ,&read_buffer);

    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not set argument[1] for  Kernel %s %d\n","read_stamp", status);
        fflush(stdout);
    }

    cl_int cmd_in = D_READ;
    status = clSetKernelArg(kernel[0],0,sizeof(cl_int) ,&cmd_in);

    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not set argument[0] for  Kernel %s %d\n","read_stamp", status);
        fflush(stdout);
    }
    clSetKernelArg(kernel[0],2,sizeof(cl_int),&buffer_id);

    status = clEnqueueTask(queue[0],kernel[0],0,NULL,NULL);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not Enqueue Kernel %s\n","read_stamp");
        fflush(stdout);
    }
    clFinish(queue[0]);

	status = clEnqueueReadBuffer(queue[0],
                                 read_buffer,
                                 CL_TRUE,
                                 0,
			                     sizeof(stamp_t) * mem_size,
                                 *time_stamp,
                                 0,
                                 NULL,
                                 NULL);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not read buffer from Kernel\n");
    }
    clFinish(queue[0]);
    fflush(stdout);
}

void read_debug_all_buffers(const cl_context         context
                           ,const cl_program         program
                           ,const cl_kernel*         kernel
                           ,const cl_command_queue*  queue
                           ,stamp_t**                time_stamp) {
    
    printf("read_debug_all_buffers, num_debug_points: %d\n", NUM_DEBUG_POINTS);
    fflush(stdout);
    const cl_int mem_size = NUM_DEBUG_POINTS*(DEBUG_SAMPLE_DEPTH+1);

    posix_memalign ((void **) time_stamp,64,sizeof(stamp_t)*mem_size);
    for(int trace_buffer = 0; trace_buffer < NUM_DEBUG_POINTS ; trace_buffer++) { 
    stamp_t *time_stamp_buffer;

        //printf("-INFO- Reading TraceBuffer[%0d]\n", trace_buffer);
        //fflush(stdout);
        read_debug( context ,program ,kernel ,queue ,trace_buffer ,&time_stamp_buffer);
        for(int sample_id = 0; sample_id <= DEBUG_SAMPLE_DEPTH; sample_id++) { 
            (*time_stamp)[trace_buffer*(DEBUG_SAMPLE_DEPTH+1) + sample_id] = time_stamp_buffer[sample_id];
            //printf("-INFO- Reading TraceBuffer[%0d]\n", trace_buffer);
            //fflush(stdout);
        }
        free(time_stamp_buffer);
    }
}

void print_debug(const stamp_t* time_stamp) { 
    int valid_samples[NUM_DEBUG_POINTS];
    for(int i = 0; i <= DEBUG_SAMPLE_DEPTH; i++) { 
        printf("TIME %5d",i);
        for(int j = 0; j < NUM_DEBUG_POINTS; j++) { 
            if(i == 0) { 
                valid_samples[j] = time_stamp[j*(DEBUG_SAMPLE_DEPTH+1)+i]; 
            }
            else { 
                if(valid_samples[j]) {
                    printf(": %10ld :", time_stamp[j*(DEBUG_SAMPLE_DEPTH+1)+i]);
                    valid_samples[j]--;
                }
                else 
                    printf(": %s :", "----------");
            }
        }
        if(i)printf("%4d ",(int) (time_stamp[1+DEBUG_SAMPLE_DEPTH+i] - time_stamp[i]));
        printf("\n");
    }
}


void read_watch(const cl_context         context
               ,const cl_kernel*         kernel
               ,const cl_command_queue*  queue
               ,const cl_int             watch_id
               ,watch_s**                watch_point) {

    cl_mem read_buffer;
    cl_int status;

    const cl_int mem_size = WATCH_SAMPLE_DEPTH+2;

    posix_memalign ((void **) watch_point,64,sizeof(watch_s)*mem_size);

    read_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(watch_s)*mem_size, NULL, &status);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not create read buffer in rest_debug\n");
    }
    
    cl_int cmd_in = D_READ;

    status = clSetKernelArg(kernel[1],0,sizeof(cl_int) ,&cmd_in);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not set argument[0] for  Kernel %s %d\n","read_watch", status);
    }

    status = clSetKernelArg(kernel[1],1,sizeof(cl_mem) ,&read_buffer);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not set argument[1] for  Kernel %s %d\n","read_watch", status);
    }

    clSetKernelArg(kernel[1],2,sizeof(cl_int),&watch_id);

    status = clEnqueueTask(queue[1],kernel[1],0,NULL,NULL);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not Enqueue Kernel %s\n","read_watch");
    }

    clFinish(queue[1]);


	status = clEnqueueReadBuffer(queue[1],
                                 read_buffer,
                                 CL_TRUE,
                                 0,
			                     sizeof(watch_s) * mem_size,
                                 *watch_point,
                                 0,
                                 NULL,
                                 NULL);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not read buffer from Kernel");
    }
    clFinish(queue[1]);
    
}

void read_watch_all_buffers(const cl_context         context
                           ,const cl_kernel*         kernel
                           ,const cl_command_queue*  queue
                           ,watch_s**                watch_point) {

    const cl_int mem_size = NUM_WATCH_POINTS*(WATCH_SAMPLE_DEPTH+2);

    posix_memalign ((void **) watch_point,64,sizeof(watch_s)*mem_size);

    for(cl_int watch_id = 0; watch_id < NUM_WATCH_POINTS ; watch_id++) { 
    watch_s *watch_buffer;

        printf("-INFO- Reading WatchBUffer[%0d]\n", watch_id);
        read_watch( context ,kernel ,queue ,watch_id ,&watch_buffer);
        for(int sample_id = 0; sample_id <= WATCH_SAMPLE_DEPTH+1; sample_id++) { 
            (*watch_point)[watch_id*(WATCH_SAMPLE_DEPTH+2) + sample_id] = watch_buffer[sample_id];
            printf("watch[%0d]  = %0d\n", watch_id , watch_buffer[sample_id].addr);
        }
        free(watch_buffer);
    }
}

void print_watch(const watch_s* watch_point) { 
    for(int i = 0; i <= WATCH_SAMPLE_DEPTH+1; i++) { 
        printf("WATCH %5d",i);
        for(int j = 0; j < NUM_WATCH_POINTS; j++) { 
            if(i>1) printf(": (%8d : %4d) ", watch_point[j*(WATCH_SAMPLE_DEPTH+2)+i].addr , watch_point[j*(WATCH_SAMPLE_DEPTH+2)+i].tag );
            else if (i==1)
                printf(": (Valid : %0d) ", watch_point[j*(WATCH_SAMPLE_DEPTH+2)+i].tag );

            else  printf(": 0x%8X :", watch_point[j*(WATCH_SAMPLE_DEPTH+2)+i].addr);
        }
        printf("\n");
    }
}


void reset_watch_all(const cl_kernel*        kernel
                    ,const cl_command_queue* queue) {

    for(int watch_id = 0; watch_id <  NUM_WATCH_POINTS ; watch_id++) { 
        reset_watch(kernel,queue,watch_id);
    }
}

void control_watch(const cl_kernel*        kernel
                  ,const cl_command_queue* queue
                  ,const cl_int            watch_id
                  ,const debug_command_e   cmd) {

    cl_int status;
    cl_int cmd_in = (cl_int) cmd;

    status  = clSetKernelArg(kernel[1],0,sizeof(cl_int),&cmd_in);
    status |= clSetKernelArg(kernel[1],1,sizeof(cl_mem),NULL);
    status |= clSetKernelArg(kernel[1],2,sizeof(cl_int),&watch_id); 

    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not set the kernel arguments");
    }

    status = clEnqueueTask(queue[1],kernel[1],0,NULL,NULL);
    if(status != CL_SUCCESS) { 
        printf("-ERROR- Could not Enqueue Kernel");
    }
    clFinish(queue[1]);
}


void reset_watch(const cl_kernel*          kernel
                ,const cl_command_queue*   queue
                ,const cl_int              watch_id) { 

    control_watch(kernel,queue,watch_id,D_RESET);

}

void stop_watch(const cl_kernel*          kernel
               ,const cl_command_queue*   queue
               ,const cl_int              watch_id) { 

    control_watch(kernel,queue,watch_id,D_STOP);

}

void start_watch(const cl_kernel*          kernel
                ,const cl_command_queue*   queue
                ,const cl_int              watch_id) { 

    control_watch(kernel,queue,watch_id,D_RECORD);
}

void start_cyclic_watch(const cl_kernel*          kernel
                       ,const cl_command_queue*   queue
                       ,const cl_int              watch_id) { 

    control_watch(kernel,queue,watch_id,D_RECORD_CYCLIC);
}



#endif //DEBUG_CPP
