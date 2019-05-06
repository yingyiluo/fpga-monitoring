#ifndef DEBUG_DEFINES__H
#define DEBUG_DEFINES__H


#pragma OPENCL EXTENSION cl_altera_channels : enable channel

#define USE_OPENCL


/* Number of debug channels for taking snapshot in time. 
 * Set the variable to zero if you don't want to enable trace buffer. Ensure that you have as many 
 * instances of take_snapshot as tracebuffers. An un-connected channel would result into compilation 
 * error 
 */
#define NUM_DEBUG_POINTS 1

/* Number of watch points for address monitoring. 
 * Set the variable to zero if you do not want a watch point. Ensure that you add as many instances of 
 * add_watch and monitor_address as the number of watch points declared here 
 */
#define NUM_WATCH_POINTS 0

/* All the sampled stamp would be stored in a local memory which can be read 
 * at the end of test. This would ensure minimal interaction with the existing design and 
 * will not affect the memory bandwidth while the design is running 
 */
#define SAMPLE_DEPTH 2048

/* Number of tags and timestamps stored for a watch point */
#define WATCH_SAMPLE_DEPTH 8

/* Keeping the depth of channels as 16,however, this will be optimized by AOCL */
#define DEBUG_CHANNEL_DEPTH 4

/* Start sampling automatically, default state after reset can be set here */
#define DEFAULT_SAMPLING_SCHEME D_RECORD

/* Start sampling automatically, default state after reset can be set here */
#define DEFAULT_WATCH_SAMPLING_SCHEME D_RECORD


#define PACKED_AGGREGATE __attribute__((aligned (1))) __attribute__((packed))

/* Time stamp data type, unsigned long (64-bit counter is sufficient to sample time) */
typedef ulong stamp_t;

/* Commands to pass to autorun kernel collecting the sample point */
typedef enum {D_RESET,        //Reset the trace buffer
              D_RECORD,        //Start storing the data in trace buffer, stop after depth : FIXME 
              D_RECORD_CYCLIC, //Store the data in trace buffer in a cyclic buffer
              D_STOP,         //Stop Storing the data     
              D_READ          //Read the data from trace buffer to host
             } debug_command_e; 

/* Not all the 64-bits of timer need to be stored, making it a 40-bit storage  would save M20 blocks */
typedef struct PACKED_AGGREGATE
               { unsigned char hi; 
                 unsigned int  lo; 
               } buffer_s;

typedef struct PACKED_AGGREGATE 
               { unsigned int   addr; 
                 unsigned short tag; 
               } watch_s;

#if SAMPLE_DEPTH <= 512  
#define DEBUG_SAMPLE_DEPTH 512
#elif SAMPLE_DEPTH <= 1024
#define DEBUG_SAMPLE_DEPTH 1024
#elif SAMPLE_DEPTH <= 1536
#define DEBUG_SAMPLE_DEPTH 1536
#elif SAMPLE_DEPTH <= 2048
#define DEBUG_SAMPLE_DEPTH 2048
#endif

/* Writes data channel without blocking, this channel is provided to a trace-buffer, which samples  
 * a free running counter and stores the data into a local memory. This memory can be read by host 
 * by launching the kernel read_stamp 
 *  
 * void take_snapshot(uint id, stamp_t in); 
 */

/* Add a watchpoint for the given address. Adress is written into a channel without blocking, which is 
 * provided to a watch buffer. This buffer conitnously monitors the adress provided by monitor_address 
 * and stores the time stamp and a tag provided by user, into a local memory. 
 * This local memory can be read by launching the kernel read_watch 
 *
 * void add_watch(uint id, size_t address); 
 *
 * void monitor_address(uint id, size_t addr, ushort tag); 
 */



#endif //DEBUG_DEFINES__H
