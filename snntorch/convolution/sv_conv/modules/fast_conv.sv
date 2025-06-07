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
        EVENT_TRANSACTION,
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
    logic coord_list_ready;

    logic conv_active; // Indicates if convolution is active
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] conv_coord_counter;
    logic [$clog2(MAX_COORDS_TO_UPDATE)-1:0] kernel_idx [0:KERNEL_SIZE*KERNEL_SIZE-1];
    // Control signals
    assign ctrl_port.active = (state != IDLE);
    assign event_port.event_ready = (state == IDLE);
    assign event_port.event_ack = (state == PREPARE_PROCESSING) && event_stored;
    
    always_ff @(posedge ctrl_port.clk or negedge ctrl_port.reset) begin
    
        if (ctrl_port.reset) begin
            event_stored <= 1'b0;
            event_coord <= '0;
        end else begin

            case (state)

                IDLE: begin
                    event_stored <= 1'b0;
                end

                EVENT_TRANSACTION: begin
                    if (event_port.event_valid && ctrl_port.enable) begin
                    
                        event_coord <= event_port.event_coord;
                        event_stored <= 1'b1;
                    
                    end
                end

                PREPARE_PROCESSING, PROCESSING: begin
                    event_stored <= 1'b1;
                end

                default: begin
                    event_stored <= 1'b0;
                end

            endcase

        end

    end

    always_ff @(posedge ctrl_port.clk or negedge ctrl_port.reset) begin
        if (ctrl_port.reset) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        case (state)
            IDLE: begin
                // Move to EVENT_TRANSACTION when we have a valid event and we're enabled
                if (ctrl_port.enable && event_port.event_valid) begin
                    next_state = EVENT_TRANSACTION;
                end else begin
                    next_state = IDLE;
                end
            end
            
            EVENT_TRANSACTION: begin
                // After acknowledging the event, prepare for processing
                next_state = PREPARE_PROCESSING;
            end
            
            PREPARE_PROCESSING: begin
                if (coord_list_ready) begin
                    // If we have coordinates ready, move to PROCESSING
                    next_state = PROCESSING;
                end else begin
                    // Otherwise, stay in PREPARE_PROCESSING until we have coordinates
                    next_state = PREPARE_PROCESSING;
                end
            end
            
            PROCESSING: begin
                
                if (conv_active) begin
                    // If convolution is active, stay in PROCESSING
                    next_state = PROCESSING;
                end else begin
                    // If convolution is not active, go back to IDLE
                    next_state = IDLE;
                end

            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end

    always_ff @(posedge ctrl_port.clk or negedge ctrl_port.reset) begin
        if (ctrl_port.reset) begin
            conv_active <= 1'b0;
        end else begin
            if (
                state == PREPARE_PROCESSING
                && next_state == PROCESSING 
                && coord_list_ready
            ) begin

                conv_active <= 1'b1;
                mem_read.coord_get = coords_list[0];
                mem_read.read_req = 1'b1;
                conv_coord_counter <= 0;

            end else if (
                state == PROCESSING
                && conv_coord_counter < coords_count
                ) begin

                conv_active <= 1'b1;
                
                if (conv_coord_counter > 0) begin

                mem_write.coord_wtr = coords_list[conv_coord_counter - 1];
                mem_write.write_req = 1'b1;

                end
                mem_write.data_in = apply_kernel_weights(
                    mem_read.data_out,
                    kernel_idx[conv_coord_counter - 1]
                );

                mem_read.coord_get = coords_list[conv_coord_counter];
                mem_read.read_req = 1'b1;

                conv_coord_counter <= conv_coord_counter + 1;

            end else if (
                state == PROCESSING 
                && conv_coord_counter == coords_count
                ) begin
                // Last write
                mem_write.data_in = apply_kernel_weights(
                    mem_read.data_out,
                    kernel_idx[conv_coord_counter - 1]
                );
                mem_write.coord_wtr = coords_list[conv_coord_counter - 1];
                mem_write.write_req = 1'b1;

                mem_read.read_req = 1'b0;
                mem_read.coord_get = '0;
                
                conv_coord_counter <= conv_coord_counter + 1;
            end else begin
                conv_active <= 1'b0;
                mem_read.read_req = 1'b0;
                mem_write.write_req = 1'b0;
                mem_read.coord_get = '0;
                mem_write.coord_wtr = '0;
                for (int i = 0; i < MAX_COORDS_TO_UPDATE; i++) begin
                    mem_write.data_in[i] = '0; // Reset data_in
                end
            end
        end
    end

    always_comb begin
        if (ctrl_port.reset) begin
            coords_count <= 0;
            coord_list_ready <= 1'b0;
        end else if (state == EVENT_TRANSACTION) begin
            for (int i = 0; i < MAX_COORDS_TO_UPDATE; i++) begin
                coords_list[i] = '0; 
                kernel_idx[i] = '0; 
            end
            coords_count <= 0; 
        end else if (state == PREPARE_PROCESSING && event_stored) begin
            automatic byte coord_counter = 0;
            automatic logic [3:0] kernel_idx_counter = 0;
            for (byte dx = -KERNEL_OFFSET; dx <= KERNEL_OFFSET; dx++) begin
                for (byte dy = -KERNEL_OFFSET; dy <= KERNEL_OFFSET; dy++) begin
                    
                    automatic byte x_coord = event_coord.x + dx;
                    automatic byte y_coord = event_coord.y + dy;

                    if (x_coord < IMG_WIDTH 
                        && x_coord >= 0 
                        && y_coord < IMG_HEIGHT
                        && y_coord >= 0) 
                    begin
                        automatic vec2_t new_coord = '{x_coord, y_coord};
                        coords_list[coord_counter] = new_coord;
                        kernel_idx[coord_counter] = kernel_idx_counter;
                        coord_counter++;
                    end
                    kernel_idx_counter++;
                end
            end
            coords_count <= coord_counter;
            coord_list_ready <= (coord_counter > 0);
            conv_coord_counter <= 0; 
        end 
    end

    `ifndef SYNTHESIS
    // Print control signals every cycle
    always @(posedge ctrl_port.clk) begin
        $display("T=%0t: FAST_CONV Debug: enable=%b, rst_n=%b, state=%s, event_valid=%b, event_ready=%b", 
                 $time, ctrl_port.enable, ctrl_port.reset, state.name(), 
                 event_port.event_valid, event_port.event_ready);
    end
    
    // Print state changes
    always @(posedge ctrl_port.clk) begin
        if (state != next_state) begin
            $display("T=%0t: FAST_CONV State: %s → %s", $time, state.name(), next_state.name());
        end
    end
    
    // Print all control signals to find the undefined one
    always @(posedge ctrl_port.clk) begin
        $display("T=%0t: FAST_CONV Debug: clk=%b, rst_n=%b, ctrl_enable=%b, ctrl_reset=%b", 
                 $time, ctrl_port.clk, ctrl_port.reset, ctrl_port.enable, ctrl_port.reset);
        $display("    → event_ready=%b, state=%s, event_valid=%b", 
                 event_port.event_ready, state.name(), event_port.event_valid);
    end
    `endif

endmodule