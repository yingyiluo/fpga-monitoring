    debug_command_e state   = D_RESET;
    bool            watch_enable = false;
    uchar           index=0;
    uchar           read_pointer=0;
    watch_s         first_read;
    watch_s         watch_mem[WATCH_SAMPLE_DEPTH];
    bool            overflow;
    #pragma acc kernels loop independent
    for(ulong infinite_counter = 0; infinite_counter < ULONG_MAX ; infinite_counter++) { 
    watch_s    data_in;
    bool       rvalid_addr;
    bool       rvalid_cmd;
    bool       rvalid_watch;
    debug_command_e next_state;
    watch_s     addr_read;
        addr_read.addr  = read_channel_nb_altera(addr_in_c[watch_id],   &rvalid_addr);
        next_state      = read_channel_nb_altera(watch_cmd_c[watch_id], &rvalid_cmd);
        data_in         = read_channel_nb_altera(watch_in_c[watch_id],  &rvalid_watch);

        if(rvalid_addr) {
            watch_enable = true;
            first_read   = addr_read;
        }

        if(rvalid_cmd) { 
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
        }
        

        switch(state) { 
            case D_RESET : 
                watch_enable = false;
                state = DEFAULT_WATCH_SAMPLING_SCHEME;
                index = 0;
                read_pointer = 0;
                overflow = 0;
                break;
            case D_RECORD : 
            case D_RECORD_CYCLIC : 
                if(rvalid_watch & watch_enable & (index < WATCH_SAMPLE_DEPTH)) { 
                    if(data_in.addr == first_read.addr) { 
                        watch_mem[index].addr = get_time(data_in.tag);
                        watch_mem[index].tag  = data_in.tag;
                        index++;
                        if(state == D_RECORD_CYCLIC & index == WATCH_SAMPLE_DEPTH) {
                            index    = 0;
                            overflow = 0;
                        }
                    }
                }
                break;
            case D_READ : { 
                if(read_pointer <= WATCH_SAMPLE_DEPTH+1) { 
                    bool     wvalid;
                    watch_s out_data;
                    watch_s depth;
                    depth.tag = overflow ? WATCH_SAMPLE_DEPTH : index;
                    out_data = read_pointer == 0 ? first_read : 
                               read_pointer == 1 ? depth : 
                               (overflow ? watch_mem[(index + read_pointer - 2)%WATCH_SAMPLE_DEPTH] : watch_mem[read_pointer - 2]); 
                    wvalid = write_channel_nb_altera(watch_out_c[watch_id],out_data); 
                    if(wvalid) 
                        read_pointer++;
                }
                else {
                    state = D_STOP;
                    read_pointer = 0; 
                }
                break;
            }
            default : break;
        }
    }

