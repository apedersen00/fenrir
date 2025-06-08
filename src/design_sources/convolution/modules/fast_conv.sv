import snn_interfaces_pkg::*;

module fast_conv #(
    parameter int COORD_BITS = DEFAULT_COORD_BITS,
    parameter int CHANNELS = DEFAULT_CHANNELS,
    parameter int BITS_PER_CHANNEL = DEFAULT_NEURON_BITS,
    parameter int IMG_WIDTH = DEFAULT_IMG_WIDTH,
    parameter int IMG_HEIGHT = DEFAULT_IMG_HEIGHT,

    parameter int KERNEL_SIZE = 3
)(
    snn_control_if.convolution ctrl_port,
    snn_event_if.convolution event_port,
    arbiter_if.read_port mem_read,
    arbiter_if.write_port mem_write
);

    // State machine states
    typedef enum logic [1:0] {
        IDLE,
        PREPARE_PROCESSING,
        PROCESSING
    } state_t;

    // State variable
    state_t state = IDLE;
    state_t next_state;

    //event storage from capture
    vec2_t event_coord;
    logic event_stored;

    localparam int KERNEL_OFFSET = KERNEL_SIZE / 2;
    localparam int MAX_COORDS_TO_UPDATE = KERNEL_SIZE * KERNEL_SIZE;

    //Array to temporaliy store which coordinates to update
    vec2_t coords_list [0:MAX_COORDS_TO_UPDATE-1];
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] coords_count;
    logic coord_list_ready = 1'b0;

    logic conv_active; // Indicates if convolution is active
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] conv_counter = 0;
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] conv_coord_counter;
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] kernel_idx [0:KERNEL_SIZE*KERNEL_SIZE-1];
    // Control signals
    assign ctrl_port.active = (state != IDLE);
    assign ctrl_port.ready = (state == IDLE);
    assign event_port.event_ready = (state == IDLE);
    
    // go to next state
    always_ff @(posedge ctrl_port.clk or negedge ctrl_port.reset) begin
        if (!ctrl_port.reset) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    //update next state
    always_comb begin
        // drives interfaces event_port and ctrl_port
        

            case (state)

                IDLE: begin
                    if (event_port.event_valid && event_stored == '0) begin
                        
                        event_coord = event_port.event_coord;
                        event_stored = 1'b1;
                        next_state = PREPARE_PROCESSING;
                    end else begin
                        event_stored = 1'b0;
                        event_port.event_ack = 1'b0;
                        next_state = IDLE;
                    end
                end

                PREPARE_PROCESSING: begin
                    event_port.event_ack = 1'b1;
                    next_state = PROCESSING;
                end

                PROCESSING: begin
                    event_port.event_ack = 1'b0;
                    if (conv_counter == coords_count - 1) begin
                        next_state = IDLE;
                    end else begin
                        next_state = PROCESSING;
                    end
                end

                default: begin
                end

            endcase
        
    end

always_ff @(posedge ctrl_port.clk) begin
    
    if (state == PROCESSING && next_state==PROCESSING) begin
        conv_counter++;
    end else begin
        conv_counter <= 0;
    end

end

// Drive read an mem ports
always_comb begin
    
        case (state)
        
            PREPARE_PROCESSING: begin
                if (coord_list_ready) begin
                    mem_read.coord_get = coords_list[0];
                    mem_read.read_req = 1'b1;
                end
            end

            PROCESSING: begin
                automatic logic [6:0] read_counter = conv_counter+1;
                if (read_counter < coords_count) begin
                    mem_read.read_req = 1'b1;
                    mem_read.coord_get = coords_list[read_counter];
                end else begin
                    mem_read.read_req = 1'b0;
                    mem_read.coord_get = '{0, 0};
                end

                // write back the data
                if (conv_counter < coords_count) begin

                mem_write.write_req = 1'b1;
                mem_write.coord_wtr = coords_list[conv_counter];
                // Data use function from package
                mem_write.data_in = apply_kernel_weights(
                    mem_read.data_out,
                    kernel_idx[conv_counter]
                );
                end
                
            end

            default: begin
                mem_read.coord_get = '{0, 0};
                mem_read.read_req = 1'b0;
                mem_write.coord_wtr = '{0, 0};
                mem_write.write_req = 1'b0;
                for (int i = 0; i < CHANNELS; i++) begin
                    mem_write.data_in[i] = '0;
                end
            end

        endcase
    
end


always_comb begin
    // Drives following signals: coords_list, coords_count, coord_list_ready, conv_active, conv_coord_counter
    
        case (state)
            //Calculate the coordinates used for the convolution
            PREPARE_PROCESSING: begin
                coords_count = 0;
                for (int dy = -KERNEL_OFFSET; dy <= KERNEL_OFFSET; dy++) begin
                    for (int dx = -KERNEL_OFFSET; dx <= KERNEL_OFFSET; dx++) begin
                        
                        automatic int _x = event_coord.x + dx;
                        automatic int _y = event_coord.y + dy;
                        automatic int flat_idx = (dy + KERNEL_OFFSET)  + (dx + KERNEL_OFFSET)* KERNEL_SIZE;

                        if (_x >= 0 && _x < IMG_WIDTH && _y >= 0 && _y < IMG_HEIGHT) begin
                            coords_list[coords_count] = '{_x, _y};
                            kernel_idx[coords_count] = flat_idx;
                            coords_count++;
                        end

                    end 
                end
                coord_list_ready = (coords_count > 0);
                
            end
            //During processing continue driving the signals
            PROCESSING: begin
            end

            //In idle state reset the signals
            default: begin
                coords_count = 0;
                for (int i = 0; i < MAX_COORDS_TO_UPDATE; i++) begin
                    coords_list[i] = '{0, 0};
                    kernel_idx[i] = 0;
                end
                coord_list_ready = 1'b0;
            
            end
        endcase
    
end


endmodule