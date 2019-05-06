#ifndef MATRIX_MUL__H
#define MATRIX_MUL__H

#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <vector>
//#include <pcap.h>
#include <stdlib.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <math.h>

#include "aocl_utils.h"
#include "common_defines.h"
#include "debug.h"

using namespace std;
using namespace aocl_utils;

#define PRINT_INFO(m) \
    std::cout << "-INFO- : " << m << std::endl; 

#define PRINT_ERROR(m) \
    std::cout << "-ERROR- : " << m << std::endl; 


int main(int argc, char *argv[]);

void help(char *argv);


bool init_opencl(const std::string &aocx 
                ,cl_platform_id    &platform
                ,cl_device_id      &device
                ,cl_context        &context
                ,cl_program        &program
                ,cl_uint           &num_devices );


bool create_kernel(const std::string      name
                  , cl_context            context
                  , cl_program            program
                  , cl_device_id          device
                  ,cl_command_queue       &queue
                  ,cl_kernel              &kernel); 

void fill_buffer(DATA_TYPE *buffer
                ,cl_uint    size);

bool write_device_buffer(cl_command_queue queue
                        ,cl_context       context
                        ,cl_mem          &host_buffer
                        ,DATA_TYPE       *device_buffer
                        ,cl_uint          size
                        ,const char      *info); 

bool create_or_read_device_buffer(
                         cl_command_queue queue
                        ,cl_context       context
                        ,cl_mem          &host_buffer
                        ,void            *device_buffer
                        ,size_t           size
                        ,const char      *info
                        ,bool             create_not_write) ; 
 
void matrix_multiply(DATA_TYPE*  data_a
                    ,DATA_TYPE*  data_b
                    ,DATA_TYPE*  data_c
                    ,int row_a
                    ,int col_a
                    ,int col_b); 


bool compare_result(DATA_TYPE *data_acc
                   ,DATA_TYPE *data_host
                   ,cl_uint    size); 
#endif //MATRIX_MUL__H 
