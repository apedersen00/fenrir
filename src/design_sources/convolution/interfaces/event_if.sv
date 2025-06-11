interface event_if #(
    parameter int BITS_PER_COORDINATE,
    parameter int IN_CHANNELS
)(
    input logic clk,
    input logic rst_n
);

    typedef struct packed {
        logic timestep;
        logic [BITS_PER_COORDINATE-1:0] x;
        logic [BITS_PER_COORDINATE-1:0] y;
        logic [IN_CHANNELS-1:0] spikes; // Spike vector for input channels
    } event_t;

    event_t event_data;
    logic event_valid;
    logic conv_ready;
    logic conv_ack;

    modport capture(
        output event_data,
        output event_valid,
        input  conv_ready,
        input  conv_ack,
        input  clk,
        input  rst_n
    );

    // Convolution module modport (receives structured events from capture)  
    modport convolution(
        input  event_data,
        input  event_valid,
        output conv_ready,
        output conv_ack
    );

endinterface