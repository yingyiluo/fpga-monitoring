library verilog;
use verilog.vl_types.all;
entity get_time is
    generic(
        WIDTH           : integer := 64
    );
    port(
        clock           : in     vl_logic;
        resetn          : in     vl_logic;
        ivalid          : in     vl_logic;
        oready          : in     vl_logic;
        command         : in     vl_logic_vector;
        iready          : out    vl_logic;
        ovalid          : out    vl_logic;
        curr_time       : out    vl_logic_vector
    );
end get_time;
