    //buffer_s        debug_memory[DEBUG_SAMPLE_DEPTH];
    stamp_t         debug_memory[DEBUG_SAMPLE_DEPTH];
    int             sampled_point;
    debug_command_e state        = D_RESET;
    int             read_count;
    #pragma acc kernels loop independent
    for(ulong infinite_counter = 0; infinite_counter < ULONG_MAX ; infinite_counter++) { 
    debug_command_e next_state;
    bool            rvalid;
    bool            read_valid;
    short           take_stamp;
    bool            overflow;    
    stamp_t         time_stamp;
        take_stamp = read_channel_nb_altera(time_in_c[buffer_id], &read_valid);
        next_state = read_channel_nb_altera(command_c[buffer_id], &rvalid);
        time_stamp = get_time((stamp_t)take_stamp);  
        if(rvalid) {
            switch(next_state) {
                case D_RESET: 
                case D_STOP : 
                case D_READ : 
                    state = next_state;
                    break;
                case D_RECORD_CYCLIC: 
                case D_RECORD: 
                    state = state == D_STOP ? next_state : state; 
                    break;
                default : 
                    break;
            } 
            //printf("Read Command %0d\n", next_state);
        }
        switch(state) { 
            case D_RESET: {
                sampled_point = 0;
                read_count    = 0;
                overflow      = 0;
                state         = DEFAULT_SAMPLING_SCHEME;
                //printf("-INFO- DEBUG RESET COMPLETE \n");
                break;
            }
            case D_RECORD: 
            case D_RECORD_CYCLIC: { 
                if(read_valid & (sampled_point < DEBUG_SAMPLE_DEPTH)) { 
                buffer_s stamp_value;

                    //stamp_value.lo = time_stamp & 0xFFFFFFFF;
                    //stamp_value.hi = (time_stamp >> 32) & 0xFF;
                    //printf("Sampled [%0d][%0d] = %0lx\n",channel_no,index,stamp_value);
                    //debug_memory[sampled_point] = stamp_value;
                    debug_memory[sampled_point] = time_stamp;
                    sampled_point++;
                    if((state == D_RECORD_CYCLIC) & 
                       (sampled_point == DEBUG_SAMPLE_DEPTH)) { 
                        sampled_point = 0;
                        overflow      = true;
                    }
                        
                }
                break;
            }
            case D_READ: {
                if(read_count <= DEBUG_SAMPLE_DEPTH) { 
                //buffer_s ts_out;
                stamp_t ts_out;
                bool     write_valid;
                    if(read_count == 0) 
                        //ts_out.lo = overflow ? DEBUG_SAMPLE_DEPTH : sampled_point;
                        ts_out = overflow ? DEBUG_SAMPLE_DEPTH : sampled_point;
                    else
                        ts_out  = overflow ? debug_memory[(sampled_point + read_count - 1)%DEBUG_SAMPLE_DEPTH] : debug_memory[read_count - 1];
                    //printf("Read TIme Stamp %0lu\n", ts_out);
                    write_valid = write_channel_nb_altera(read_stamp_c[buffer_id], ts_out);
                    mem_fence(CLK_CHANNEL_MEM_FENCE);
                    if(write_valid) { 
                        read_count++;
                    }
                }
                else  {  
                    state      = D_STOP;
                    read_count = 0;
                }
                break;
            }
            default: break;
        }
    }

