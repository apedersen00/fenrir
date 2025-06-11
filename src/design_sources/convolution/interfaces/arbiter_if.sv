import conv_pkg::*;

interface arbiter_if #(
    parameter int BITS_PER_COORDINATE,
    parameter int OUT_CHANNELS,
    parameter int BITS_PER_NEURON
);
    typedef logic signed [BITS_PER_NEURON-1:0] feature_map_t [0:OUT_CHANNELS-1];
    typedef struct packed {
        logic [BITS_PER_COORDINATE-1:0] x;
        logic [BITS_PER_COORDINATE-1:0] y;
    } vec2_t;
    // Read signals
    vec2_t coord_get;        // coordinate to read from
    feature_map_t data_out;  // data from arbiter to conv/pool module
    logic read_req;          // conv/pool requests read

    // Write signals  
    vec2_t coord_wtr;        // coordinate to write to
    feature_map_t data_in;      // data from conv/pool module to arbiter
    logic write_req;         // conv/pool requests write

    // Read port for conv and pool modules
    modport read_port(
        output coord_get,
        input  data_out,
        output read_req
    );

    // Write port for conv and pool modules
    modport write_port(
        output coord_wtr,
        output data_in,
        output write_req
    );

    // For the arbiter module itself
    modport arbiter(
        input  coord_get,
        output data_out,
        input  read_req,
        input  coord_wtr,
        input  data_in,
        input  write_req
    );
endinterface