import conv_pkg::*; // Import the kernel BRAM interface

module Convolution2d #(
    parameter int BITS_PER_COORDINATE,
    parameter int IN_CHANNELS,
    parameter int OUT_CHANNELS,
    parameter int IMG_WIDTH,
    parameter int IMG_HEIGHT,
    parameter int BITS_PER_NEURON,
    parameter int BITS_PER_KERNEL_WEIGHT,
    parameter int KERNEL_SIZE
)(
    input logic clk,
    input logic rst_n,
    
    kernel_bram_if.conv_module mem_kernel,
    arbiter_if.read_port mem_read,
    arbiter_if.write_port mem_write,
    event_if.convolution capture

);

    localparam int KERNEL_OFFSET = KERNEL_SIZE / 2;
    localparam int MAX_COORDS_TO_UPDATE = KERNEL_SIZE * KERNEL_SIZE;

    typedef struct packed {
        logic timestep;
        logic [BITS_PER_COORDINATE-1:0] x;
        logic [BITS_PER_COORDINATE-1:0] y;
        logic [IN_CHANNELS-1:0] spikes; // Spike vector for input channels
    } event_t;

    typedef struct packed {
        logic [BITS_PER_COORDINATE-1:0] x;
        logic [BITS_PER_COORDINATE-1:0] y;
    } vec2_t;

    typedef enum logic [1:0] {
        IDLE,
        PREPARE_PROCESSING,
        PROCESSING
    } state_t;

    typedef logic signed [BITS_PER_KERNEL_WEIGHT-1:0] kernel_weight_vector_t [0:OUT_CHANNELS-1];
    typedef logic signed [BITS_PER_NEURON-1:0] feature_map_t [0:OUT_CHANNELS-1];

    state_t state, next_state = IDLE;

    // registers for event storage
    vec2_t event_coord;
    logic[IN_CHANNELS-1:0] event_spikes;
    logic event_stored = 0; 

    // TEMP DEBUG
    kernel_weight_vector_t kernel_weights; // Temporary storage for kernel weights


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

                IDLE: begin
                    if (next_state == PREPARE_PROCESSING) begin
                        conv_active <= 1;
                    end else begin
                        conv_active <= 0; // Reset convolution active state
                    end
                end

                PREPARE_PROCESSING: begin
                    // set first channel as the current channel
                    conv_counter <= 0;
                    conv_active <= 1; // Start convolution processing
                end

                PROCESSING: begin
                    if (conv_counter == coords_count && channel_counter == (channels_count - 1)) begin
                        // done condition
                        conv_active <= 0; // End convolution processing
                        conv_counter <= 0; // Reset counter for next event
                        channel_counter <= 0; // Reset channel counter
                    end else if (conv_counter == coords_count) begin
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
            
            if (conv_counter != coords_count && conv_active) begin
                mem_read.coord_get = coords_to_update[conv_counter];
                mem_read.read_req = 1; // Request read from memory

                // Read kernel
                mem_kernel.addr = channels_to_process_list[channel_counter] * KERNEL_SIZE**2 + kernel_idx[conv_counter];             
                mem_kernel.en = 1; // Enable kernel BRAM

            end else begin
                mem_read.read_req = 0; // No read request on last iteration
                mem_read.coord_get = {'0, '0}; // Reset coordinate to avoid invalid reads
                mem_kernel.addr = '0;
                mem_kernel.en = 0; // Disable kernel BRAM
            end

            if (conv_counter > 0 && conv_active) begin
                mem_write.coord_wtr = coords_to_update[conv_counter - 1];
                mem_write.write_req = 1;
                kernel_weights = bram_to_kernel_weight_vector(mem_kernel.data_out); // Read kernel weights from BRAM
                mem_write.data_in = add_kernel_weights_to_feature_map(mem_read.data_out, kernel_weights); // Add kernel weights to feature map
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
                        automatic int flat_idx = (dy + KERNEL_OFFSET)  * KERNEL_SIZE + (dx + KERNEL_OFFSET);
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
                if (capture.event_valid && !event_stored) begin
                    next_state = PREPARE_PROCESSING;
                    event_coord.x = capture.event_data.x;
                    event_coord.y = capture.event_data.y;
                    event_spikes = capture.event_data.spikes;
                    event_stored = 1;
                end else begin
                    next_state = IDLE;
                    capture.conv_ack = 0; 
                    capture.conv_ready = 1;
                    event_coord = '0; 
                    event_spikes = '0; 
                end   
            end

            PREPARE_PROCESSING: begin
                next_state = PROCESSING;
                capture.conv_ack = 1; // Acknowledge the event
                capture.conv_ready = 0; // Indicate that convolution is not ready yet
            end

            PROCESSING: begin
                if (conv_active) begin
                    next_state = PROCESSING; // Continue processing
                    event_stored = 1;
                    capture.conv_ready = 0; // Indicate that convolution is not ready yet
                end else begin
                    next_state = IDLE; // Go back to IDLE after processing
                    event_stored = 0; // Reset event storage
                    capture.conv_ready = 1; // Indicate that convolution is ready for the next event
                end
                capture.conv_ack = 0; // Reset acknowledgment after processing
                
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

    // ==================================================================
    // Functions
    // ==================================================================
    function automatic kernel_weight_vector_t bram_to_kernel_weight_vector(
        input logic [OUT_CHANNELS * BITS_PER_KERNEL_WEIGHT - 1:0] bram_data
    );

        kernel_weight_vector_t result;

        for (int channel = 0; channel < OUT_CHANNELS; channel++) begin
            result[channel] = bram_data[channel * BITS_PER_KERNEL_WEIGHT +: BITS_PER_KERNEL_WEIGHT];
        end

        return result;

    endfunction

    function automatic feature_map_t add_kernel_weights_to_feature_map(
        input feature_map_t fm,
        input kernel_weight_vector_t kernel_weights
    );
        feature_map_t result;

        for (int channel = 0; channel < OUT_CHANNELS; channel++) begin

            automatic logic signed [BITS_PER_NEURON:0] result_before_clamp;
            result_before_clamp = fm[channel] + kernel_weights[channel];

            if (result_before_clamp > $signed({1'b0, {BITS_PER_NEURON - 1{1'b1}}})) begin
                result[channel] = {1'b0, {BITS_PER_NEURON - 1{1'b1}}}; // Clamp to max value
            end else if (result_before_clamp < $signed({1'b1, {BITS_PER_NEURON - 1{1'b0}}})) begin
                result[channel] = {1'b1, {BITS_PER_NEURON - 1{1'b0}}}; // Clamp to min value
            end else begin
                result[channel] = result_before_clamp; // No clamping needed
            end
        end

        return result;
    endfunction


endmodule