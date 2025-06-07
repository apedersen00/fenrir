// =============================================================================
// File: interfaces/arbiter_if.sv
// Description: Arbiter interface with separated read/write ports and simplified handshaking
// =============================================================================

import snn_interfaces_pkg::*;

interface arbiter_if #(
    parameter int COORD_BITS = DEFAULT_COORD_BITS,
    parameter int CHANNELS = DEFAULT_CHANNELS,
    parameter int BITS_PER_CHANNEL = DEFAULT_NEURON_BITS
);

    // Use simple unpacked array type - much cleaner
    typedef snn_interfaces_pkg::feature_map_t fm_array_t;
    
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

    // Debug functions
    function automatic string read_status();
        return $sformatf("READ: coord=(%0d,%0d) req=%b ready=%b", 
                        coord_get.x, coord_get.y, read_req, read_ready);
    endfunction

    function automatic string write_status();
        return $sformatf("WRITE: coord=(%0d,%0d) req=%b ready=%b", 
                        coord_wtr.x, coord_wtr.y, write_req, write_ready);
    endfunction

    function automatic string data_brief();
        return $sformatf("data_out=[0x%h,0x%h,...] data_in=[0x%h,0x%h,...]", 
                        data_out[0], data_out[1], data_in[0], data_in[1]);
    endfunction

endinterface