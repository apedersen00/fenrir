import conv_pkg::*;

interface arbiter_if #(
    parameter int COORD_BITS = DEFAULT_BITS_PER_COORDINATE_IN,
    parameter int CHANNELS = DEFAULT_OUT_CHANNELS,
    parameter int BITS_PER_CHANNEL = DEFAULT_BITS_PER_NEURON
);
    typedef conv_pkg::feature_map_t fm_array_t; // Use feature map type for the convolutions
    
    // Read signals
    vec2_t coord_get;        // coordinate to read from
    fm_array_t data_out;     // data from arbiter to conv/pool module  
    logic read_req;          // conv/pool requests read
    logic read_ready;        // arbiter ready for read request

    // Write signals  
    vec2_t coord_wtr;        // coordinate to write to
    fm_array_t data_in;      // data from conv/pool module to arbiter
    logic write_req;         // conv/pool requests write
    logic write_ready;       // arbiter ready for write request

    // Read port for conv and pool modules
    modport read_port(
        output coord_get,
        input  data_out,
        output read_req,
        input  read_ready
    );

    // Write port for conv and pool modules
    modport write_port(
        output coord_wtr,
        output data_in,
        output write_req,
        input  write_ready
    );

    // For the arbiter module itself
    modport arbiter(
        input  coord_get,
        output data_out,
        input  read_req,
        output read_ready,
        input  coord_wtr,
        input  data_in,
        input  write_req,
        output write_ready
    );

    // Monitor for debugging
    modport monitor(
        input coord_get,
        input data_out,
        input read_req,
        input read_ready,
        input coord_wtr,
        input data_in,
        input write_req,
        input write_ready
    );
endinterface