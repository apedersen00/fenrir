import snn_interfaces_pkg::*;

module sum_pooling #(
    parameter int CHANNELS = DEFAULT_CHANNELS,
    parameter int BITS_PER_CHANNEL = DEFAULT_NEURON_BITS,
    parameter int IMG_WIDTH = DEFAULT_IMG_WIDTH,
    parameter int IMG_HEIGHT = DEFAULT_IMG_HEIGHT,
    parameter int OUT_VECTOR_WIDTH = (DEFAULT_COORD_BITS - 1) * 2 + CHANNELS + 1
)(
    snn_control_if.pooling ctrl_port,
    arbiter_if.read_port mem_read,
    arbiter_if.write_port mem_write,

    output logic output_valid,
    output logic [OUT_VECTOR_WIDTH-1:0] output_vector
);

    typedef enum logic [2:0] {
        IDLE,
        PAUSE,
        PROCESSING
    } state_t;

    state_t state = IDLE;
    state_t next_state;

    localparam int no_of_coords = IMG_WIDTH * IMG_HEIGHT;
    localparam int no_of_windows = no_of_coords / 4; 
    localparam int no_of_windows_one_axis = IMG_WIDTH / 2;
    
    assign ctrl_port.active = (state == PROCESSING); // or maybe PAUSE also?
    assign ctrl_port.done = (next_state == IDLE);

    vec2_t coord_start = '{0, 0};
    vec2_t previous_coord = '{0, 0};
    vec2_t window_coord = '{0, 0};
    vec2_t previous_coord_anchor = '{0, 0};

    logic [1:0] coord_counter = 0;
    logic [$clog2(no_of_coords)-1:0] total_count = 0;
    logic [$clog2(IMG_HEIGHT)-1:0] row_counter = 0;
    logic [$clog2(IMG_WIDTH)-1:0] col_counter = 0;

    logic done = 1'b0; // indicates if the pooling operation is done

    // temporary variables for memory operations
    feature_map_t read_minus_decay;
    output_vector_t last_output_vector;
    feature_map_t pooled_maps;
    

always_ff @(posedge ctrl_port.clk or negedge ctrl_port.reset) begin
    if (!ctrl_port.reset) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

always_ff @(posedge ctrl_port.clk or negedge ctrl_port.reset) begin
    // drive counters
    if (!ctrl_port.reset) begin
        row_counter <= 0;
        col_counter <= 0;
        coord_counter <= 0;
        total_count <= 0;
        done <= 1'b0; 
    end else if (state == PROCESSING && total_count < no_of_coords) begin

        if (coord_counter == 3) begin
            
            automatic int tmp_row_counter = 0;
            automatic int tmp_col_counter = 0;

            if (col_counter + 2 >= IMG_WIDTH) begin
                row_counter <= row_counter + 2;
                col_counter <= 0;
                tmp_row_counter = row_counter + 2;
                tmp_col_counter = 0;

            end else begin
                tmp_row_counter = row_counter;
                tmp_col_counter = col_counter + 2;
                col_counter <= col_counter + 2;

            end

            coord_start = '{tmp_col_counter, tmp_row_counter};
            
        end

        coord_counter <= coord_counter + 1; // overflow doesnt matter
        total_count <= total_count + 1;
        
    end else if (total_count + 1 >= no_of_coords) begin
        done <= 1'b1; // set done when all coordinates are processed
        coord_counter <= 0;
        total_count <= 0;
        row_counter <= 0;
        col_counter <= 0;
    end else begin
        done <= 1'b0; // reset done if not processing
    end
end

// Keep window coordinates and update pooled maps
always_ff @(posedge ctrl_port.clk or negedge ctrl_port.reset) begin
    if (!ctrl_port.reset) begin
        window_coord <= '{0, 0};
        previous_coord_anchor <= '{0, 0};
        last_output_vector <= '{default: 0};
        output_valid <= 1'b0; // reset output valid
    end else if (state == PROCESSING) begin
        previous_coord <= window_coord;
        if (coord_counter == 2) begin
            previous_coord_anchor <= coord_start; // save the anchor for the next window
        end
        

        if (coord_counter == 0 && total_count == 0) begin
            pooled_maps = '{default: 0}; // reset all channels to zero

        end else if (coord_counter == 0) begin
            
            automatic logic [DEFAULT_COORD_BITS-2:0] x = previous_coord_anchor.x >> 1;
            automatic logic [DEFAULT_COORD_BITS-2:0] y = previous_coord_anchor.y >> 1;
            automatic output_vector_t temp_spike_vector = '{default: 0};
            pooled_maps <= combine_feature_maps(
                pooled_maps,
                read_minus_decay
            );

            temp_spike_vector = create_output_spike_vector(
                pooled_maps,
                threshold_vector,
                x,
                y,
                0
            );

            if (total_count == no_of_coords) begin
                temp_spike_vector.timestep = 1'b1; // last event
            end

            if (|temp_spike_vector.spikes) begin
                
                last_output_vector <= temp_spike_vector;
                output_valid <= 1'b1; // signal that output is valid
                output_vector <= pack_output_vector(last_output_vector);
                
            end else begin
                last_output_vector <= '{default: 0}; // reset output vector if no spikes
                output_valid <= 1'b0; // signal that output is not valid
            end

            /// reset pooled_maps for the next window
            pooled_maps <= '{default: 0}; // reset all channels to zero

        end else begin
            pooled_maps <= combine_feature_maps(
                pooled_maps,
                read_minus_decay
            );
        end

    end
end

// lets drive som memory
always_comb begin
    
        case (state)

            PROCESSING: begin
                // figure out coordinates to read
                automatic vec2_t window = coord_start;
                window.x = coord_start.x + coord_counter[0];
                window.y = coord_start.y + coord_counter[1];
                window_coord = window;
                
                mem_read.coord_get = window;
                mem_read.read_req = 1'b1; 

                // 
                read_minus_decay = combine_feature_maps(
                    mem_read.data_out, // assuming data_out is the read data
                    decay_vector
                );
                
                // write to memory
                mem_write.coord_wtr = previous_coord;
                mem_write.data_in = reset_to_zero_if_above_threshold(
                    read_minus_decay,
                    threshold_vector
                );

                if (total_count > 0) begin // dont need to write on first read
                    mem_write.write_req = 1'b1;
                end
            end

            PAUSE: begin
                if (next_state == PAUSE) begin
                end else if (next_state == PROCESSING) begin
                end
            end
        
            default: begin
                mem_read.read_req = 1'b0;
                mem_write.write_req = 1'b0;
                mem_read.coord_get = '{0, 0};
                mem_write.coord_wtr = '{0, 0};
            end

        endcase
    
end


always_comb begin


        case (state)

            IDLE: begin
                if (ctrl_port.enable) begin
                    next_state = PROCESSING;
                end else begin
                    next_state = IDLE; // stay in idle state
                end
            end
            PROCESSING: begin
                if (done) begin
                    next_state = IDLE;
                end else if (!ctrl_port.enable) begin
                    next_state = PAUSE;
                end else begin
                    next_state = PROCESSING;
                end
            end
            PAUSE: begin
                if (ctrl_port.enable) begin
                    next_state = PROCESSING;
                end else begin
                    next_state = PAUSE; // stay in pause state
                end
            end
        endcase

    

end
endmodule