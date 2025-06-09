import snn_interfaces_pkg::*;
module Convolution2d #(
    parameter int COORD_BITS = DEFAULT_COORD_BITS,
    parameter int IN_CHANNELS = DEFAULT_IN_CHANNELS,
    parameter int OUT_CHANNELS = DEFAULT_OUT_CHANNELS,
    parameter int IMG_WIDTH = DEFAULT_IMG_WIDTH,
    parameter int IMG_HEIGHT = DEFAULT_IMG_HEIGHT,
    parameter int BITS_PER_NEURON = DEFAULT_NEURON_BITS,
    parameter int BITS_PER_KERNEL_WEIGHT = DEFAULT_KERNEL_BITS,
    parameter int KERNEL_SIZE = 3
)(
    input logic clk,
    input logic rst_n,
    
    kernel_bram_if.conv_module bram_port,

    // temporary fake interface for events
    input output_vector_t event_in //TODO : replace with interface for events (capture event)
);

    typedef enum logic [1:0] {
        IDLE,
        PREPARE_PROCESSING,
        PROCESSING
    } state_t;

    state_t state, next_state = IDLE;




    // Obligatory state machine logic
    always_comb begin

        case (state)
            IDLE: begin
                next_state = PREPARE_PROCESSING;
            end

            PREPARE_PROCESSING: begin
                next_state = PROCESSING;
            end

            PROCESSING: begin
                next_state = IDLE;
            end
        endcase
    end 

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

endmodule