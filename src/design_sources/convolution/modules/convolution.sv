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
    
    kernel_bram_if.conv_module mem_kernel,
    arbiter_if.read_port mem_read,
    arbiter_if.write_port mem_write,
    // temporary fake interface for events
    input output_vector_t event_in, //TODO : replace with interface for events (capture event)
    input logic event_valid, //TODO : replace with interface for events (capture event)
    output logic event_ack //TODO : replace with interface for events (capture event)
);

    localparam int KERNEL_OFFSET = KERNEL_SIZE / 2;
    localparam int MAX_COORDS_TO_UPDATE = KERNEL_SIZE * KERNEL_SIZE;

    typedef enum logic [1:0] {
        IDLE,
        PREPARE_PROCESSING,
        PROCESSING
    } state_t;

    state_t state, next_state = IDLE;

    // registers for event storage
    vec2_t event_coord;
    spike_vector_t event_spikes;
    logic event_stored = 0; 

    // registers for coordinates to update and kernel positions
    vec2_t coords_to_update [0:MAX_COORDS_TO_UPDATE-1];
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] coords_count = 0; // length of current coords list
    logic coord_list_ready = 1'b0;
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] kernel_idx [0:KERNEL_SIZE*KERNEL_SIZE-1];

    // registers for controlling which channels to process
    logic [$clog2(IN_CHANNELS)-1:0] channels_to_process_list [0:IN_CHANNELS-1];
    logic [$clog2(IN_CHANNELS):0] channels_count = 0;
    
    // Counter registers and conv status
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] conv_counter = 0; // Counter for kernel positions
    logic [$clog2(IN_CHANNELS)-1:0] channel_counter = 0; // Current channel being processed
    logic conv_active = 0; // Indicates if convolution is active
    // ==================================================================
    // Counters
    // ==================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
        end else begin

            case (state)
                PREPARE_PROCESSING: begin
                    // set first channel as the current channel
                    conv_counter <= 0;
                    conv_active <= 1; // Start convolution processing
                end

                PROCESSING: begin
                    if (conv_counter == (KERNEL_SIZE ** 2) && channel_counter == (channels_count - 1)) begin
                        // done condition
                        conv_active <= 0; // End convolution processing
                        conv_counter <= 0; // Reset counter for next event
                        channel_counter <= 0; // Reset channel counter
                    end else if (conv_counter == (KERNEL_SIZE ** 2)) begin
                        // Move to the next channel
                        channel_counter <= channel_counter + 1; // Increment channel counter
                        conv_counter <= 0; // Reset kernel position counter for the next channel

                    end else begin
                        conv_counter <= conv_counter + 1; // Increment kernel position counter
                    end

                end
            endcase 

        end
    end

    // ==================================================================
    // MEMORY DRIVERS
    // ==================================================================
    always_comb begin

        case (state)
        PROCESSING: begin
            
            if (conv_counter != (KERNEL_SIZE ** 2)) begin
                mem_read.coord_get = coords_to_update[conv_counter];
                mem_read.read_req = 1; // Request read from memory
            end else begin
                mem_read.read_req = 0; // No read request on last iteration
                mem_read.coord_get = {'0, '0}; // Reset coordinate to avoid invalid reads
            end

            if (conv_counter > 0) begin
                mem_write.coord_wtr = coords_to_update[conv_counter - 1];
                mem_write.write_req = 1;
                //mem_write.data_in = '0;
            end else begin
                mem_write.write_req = 0; // No write request on first iteration
                mem_write.coord_wtr = {'0, '0}; // Reset coordinate to avoid invalid writes
                //mem_write.data_in = '0; // Reset data input
            end
            

        end
        endcase


    end

    // ==================================================================
    // Calculate the coordinates and kernel positions for the convolution
    // ==================================================================
    always_comb begin
        case (state)
            PREPARE_PROCESSING: begin

                coords_count = 0;
                for (int dy = -KERNEL_OFFSET; dy <= KERNEL_OFFSET; dy++) begin
                    for (int dx = -KERNEL_OFFSET; dx <= KERNEL_OFFSET; dx++) begin

                        automatic int _x = event_coord.x + dx;
                        automatic int _y = event_coord.y + dy;
                        automatic int flat_idx = (dy + KERNEL_OFFSET) + (dx + KERNEL_OFFSET) * KERNEL_SIZE;
                        if (   _x >= 0 
                            && _x < IMG_WIDTH
                            && _y >= 0
                            && _y < IMG_HEIGHT
                        ) begin
                            
                            coords_to_update[coords_count] = '{_x, _y};
                            kernel_idx[coords_count] = flat_idx;
                            coords_count++;

                        end
                    end
                end 
                channels_count = 0;
                for (int i = 0; i < IN_CHANNELS; i++) begin
                    if (event_spikes[i]) begin
                        channels_to_process_list[channels_count] = i;
                        channels_count++;
                    end
                end
                coord_list_ready = (coords_count > 0);
            end

            PROCESSING: begin end // lol just keeping the stuff alive

            default: begin
                coords_count = 0;
                coord_list_ready = 0;
                for (int i = 0; i < MAX_COORDS_TO_UPDATE; i++) begin
                    coords_to_update[i] = '0; // Reset coordinates
                    kernel_idx[i] = '0; // Reset kernel indices
                end
                for (int i = 0; i < IN_CHANNELS; i++) begin
                    channels_to_process_list[i] = '0; // Reset channel list
                end
                channels_count = 0; // Reset channel count
            end
        endcase
    end

    // ==================================================================
    // State machine logic
    // ==================================================================
    always_comb begin

        case (state)
            IDLE: begin
                if (event_valid && !event_stored) begin
                    next_state = PREPARE_PROCESSING;
                    event_coord.x = event_in.x;
                    event_coord.y = event_in.y;
                    event_spikes = event_in.spikes;
                    event_stored = 1;
                end else begin
                    next_state = IDLE;
                    event_ack = 0; // Do not acknowledge if no event is stored
                    event_coord = '0; // Reset coordinates
                    event_spikes = '0; // Reset spikes
                end   
            end

            PREPARE_PROCESSING: begin
                next_state = PROCESSING;
                event_ack = 1; // Acknowledge the event
            end

            PROCESSING: begin
                if (conv_active) begin
                    next_state = PROCESSING; // Continue processing
                end else begin
                    next_state = IDLE; // Go back to IDLE after processing
                end
                event_ack = 0; // Reset acknowledgment after processing
                event_stored = 1;
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