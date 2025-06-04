// =============================================================================
// File: modules/capture_event.sv
// Description: Event capture module - reads from FIFO and converts to structured events
// =============================================================================

import snn_interfaces_pkg::*;

module capture_event #(
    parameter int DATA_WIDTH = DEFAULT_COORD_BITS * 2,  // For packed coordinates
    parameter int IMG_HEIGHT = DEFAULT_IMG_HEIGHT,
    parameter int IMG_WIDTH  = DEFAULT_IMG_WIDTH
)(
    fifo_if.consumer        fifo_port,
    snn_event_if.capture    conv_port,
    snn_control_if.capture  ctrl_port
);

    // Extract clock from control interface
    logic clk;
    logic rst_n;
    assign clk = ctrl_port.clk;
    assign rst_n = !ctrl_port.reset;  // Convert active-high reset to active-low

    // Internal state
    typedef enum logic [1:0] {
        IDLE,
        READING,
        OUTPUTTING
    } state_t;
    
    state_t current_state, next_state;
    logic read_cycle_count;  // Track cycles in READING state

    // Capture condition logic
    logic can_capture;
    assign can_capture = ctrl_port.enable &&
                        !ctrl_port.reset &&
                        conv_port.event_ready &&
                        !fifo_port.empty &&
                        (current_state == IDLE);

    // State machine for proper handshaking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            read_cycle_count <= 1'b0;
        end else begin
            current_state <= next_state;
            
            // Count cycles in READING state
            if (current_state == READING) begin
                read_cycle_count <= ~read_cycle_count;  // Toggle: 0 → 1 → 0
            end else begin
                read_cycle_count <= 1'b0;
            end
        end
    end
    
    // Next state logic
    always_comb begin
        case (current_state)
            IDLE: begin
                if (can_capture) begin
                    next_state = READING;
                end else begin
                    next_state = IDLE;
                end
            end
            
            READING: begin
                if (read_cycle_count == 1'b1) begin
                    // Second cycle in READING - data is now valid, go to OUTPUTTING
                    next_state = OUTPUTTING;
                end else begin
                    // First cycle in READING - stay here for FIFO data to appear
                    next_state = READING;
                end
            end
            
            OUTPUTTING: begin
                if (conv_port.event_valid && conv_port.event_ack) begin
                    next_state = IDLE;
                end else begin
                    next_state = OUTPUTTING;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end

    // FIFO read control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_port.read_en <= 1'b0;
        end else begin
            if (current_state == IDLE && can_capture) begin
                fifo_port.read_en <= 1'b1;
            end else if (current_state == READING && read_cycle_count == 1'b0) begin
                // Keep read_en high for first cycle only
                fifo_port.read_en <= 1'b0;
            end else begin
                fifo_port.read_en <= 1'b0;
            end
        end
    end

    // Data conversion and output
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_port.event_coord <= '0;
            conv_port.event_valid <= 1'b0;
        end else begin
            case (current_state)
                IDLE: begin
                    conv_port.event_valid <= 1'b0;
                end
                
                READING: begin
                    if (read_cycle_count == 1'b1) begin
                        // Second cycle of READING - prepare data AND assert valid
                        // This ensures event_valid goes high when we enter OUTPUTTING
                        conv_port.event_coord <= unpack_coordinates(fifo_port.read_data);
                        conv_port.event_valid <= 1'b1;
                    end else begin
                        // First cycle of READING - keep valid low
                        conv_port.event_valid <= 1'b0;
                    end
                end
                
                OUTPUTTING: begin
                    // Keep valid high until acknowledgment received
                    if (conv_port.event_ack) begin
                        conv_port.event_valid <= 1'b0;  // Immediately go low when ack received
                    end else begin
                        conv_port.event_valid <= 1'b1;  // Stay high until ack
                    end
                end
                
                default: begin
                    conv_port.event_valid <= 1'b0;
                end
            endcase
        end
    end

    // Status reporting
    assign ctrl_port.active = (current_state != IDLE);

endmodule