// =============================================================================
// File: testbenches/tb_event_to_conv.sv
// Description: Test the event transaction between capture_event and fast_conv
// =============================================================================

import snn_interfaces_pkg::*;

module tb_event_to_conv;

    // Testbench parameters
    localparam int DATA_WIDTH = 16;  // 2 * DEFAULT_COORD_BITS for packed coordinates
    localparam int ADDR_WIDTH = 3;   // Small FIFO for testing
    
    // Clock and reset
    logic clk = 0;
    logic rst_n;
    
    // Generate clock - 10ns period
    always #5 clk = ~clk;
    
    // =========================================================================
    // Interface Instances
    // =========================================================================
    
    // FIFO interface
    fifo_if #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) 
        fifo_bus (.clk(clk), .rst_n(rst_n));
    
    // Event interface (connects capture to fast_conv)
    snn_event_if event_bus ();
    
    // Control interface for capture_event
    snn_control_if capture_ctrl ();
    
    // Control interface for fast_conv
    snn_control_if conv_ctrl ();
    
    // =========================================================================
    // Module Instances
    // =========================================================================
    
    // FIFO module
    fifo #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) 
        fifo_inst (.fifo_port(fifo_bus.fifo_module));
    
    // Capture event module
    capture_event #(
        .DATA_WIDTH(DATA_WIDTH),
        .IMG_HEIGHT(DEFAULT_IMG_HEIGHT),
        .IMG_WIDTH(DEFAULT_IMG_WIDTH)
    ) capture_inst (
        .fifo_port(fifo_bus.consumer),
        .conv_port(event_bus.capture),
        .ctrl_port(capture_ctrl.capture)
    );
    
    // Fast convolution module
    fast_conv #(
        .COORD_BITS(DEFAULT_COORD_BITS),
        .CHANNELS(DEFAULT_CHANNELS),
        .BITS_PER_CHANNEL(DEFAULT_NEURON_BITS),
        .IMG_WIDTH(DEFAULT_IMG_WIDTH),
        .IMG_HEIGHT(DEFAULT_IMG_HEIGHT),
        .KERNEL_SIZE(3)
    ) conv_inst (
        .ctrl_port(conv_ctrl.convolution),
        .event_port(event_bus.convolution)
    );
    
    // =========================================================================
    // Connect Control Signals
    // =========================================================================
    
    // Connect clocks
    assign capture_ctrl.clk = clk;
    assign conv_ctrl.clk = clk;
    
    // We control enable and reset for both modules
    logic test_enable;
    logic test_reset;
    
    assign capture_ctrl.enable = test_enable;
    assign capture_ctrl.reset = test_reset;
    assign conv_ctrl.enable = test_enable;
    assign conv_ctrl.reset = test_reset;
    
    // =========================================================================
    // Test Procedure
    // =========================================================================
    
    initial begin
        $display("=== Event Transaction Test: capture_event → fast_conv ===");
        
        // Initialize signals
        rst_n = 0;
        test_enable = 0;
        test_reset = 1;
        
        // FIFO producer signals
        fifo_bus.write_en = 0;
        fifo_bus.write_data = 0;
        
        // Reset sequence
        #20;
        rst_n = 1;
        test_reset = 0;
        #10;
        
        $display("T=%0t: Reset complete", $time);
        
        // =====================================================================
        // Test 1: Write event data to FIFO
        // =====================================================================
        $display("\n=== Test 1: Loading Event Data into FIFO ===");
        
        // Write some test coordinates to FIFO
        // Event 1: (5, 10) - Fix: use proper vec2_t struct
        @(posedge clk);
        fifo_bus.write_data = pack_coordinates('{x: 8'd5, y: 8'd10});
        fifo_bus.write_en = 1;
        $display("T=%0t: Writing event coord (5,10) = 0x%04h to FIFO", 
                 $time, fifo_bus.write_data);
        
        @(posedge clk);
        fifo_bus.write_en = 0;
        
        // Event 2: (15, 20) - Fix: use proper vec2_t struct
        @(posedge clk);
        fifo_bus.write_data = pack_coordinates('{x: 8'd15, y: 8'd20});
        fifo_bus.write_en = 1;
        $display("T=%0t: Writing event coord (15,20) = 0x%04h to FIFO", 
                 $time, fifo_bus.write_data);
        
        @(posedge clk);
        fifo_bus.write_en = 0;
        
        $display("T=%0t: FIFO loading complete. FIFO empty=%b, full=%b", 
                 $time, fifo_bus.empty, fifo_bus.full);
        
        // =====================================================================
        // Test 2: Enable modules and observe event flow
        // =====================================================================
        $display("\n=== Test 2: Enable Modules and Watch Event Transaction ===");
        
        @(posedge clk);
        test_enable = 1;
        $display("T=%0t: Modules enabled", $time);
        
        // Monitor for several cycles to see the complete transaction
        for (int i = 0; i < 20; i++) begin
            @(posedge clk);
            
            // Print detailed status each cycle - Add fast_conv state info
            $display("T=%0t: Cycle %2d | Capture: active=%b | Conv: state=%s, active=%b, ready=%b | Event: valid=%b, ready=%b, ack=%b", 
                     $time, i,
                     capture_ctrl.active,
                     conv_inst.state.name(), conv_ctrl.active, conv_ctrl.ready,
                     event_bus.event_valid, event_bus.event_ready, event_bus.event_ack);
            
            // Show event coordinates when valid
            if (event_bus.event_valid) begin
                $display("         → Event coordinates: (%0d,%0d)", 
                         event_bus.event_coord.x, event_bus.event_coord.y);
            end
            
            // Show FIFO status and capture_event internal state
            $display("         → FIFO: empty=%b, full=%b, read_en=%b", 
                     fifo_bus.empty, fifo_bus.full, fifo_bus.read_en);
            $display("         → Capture can_capture=%b, state=%s", 
                     capture_inst.can_capture, capture_inst.current_state.name());
        end
        
        // =====================================================================
        // Test 3: Send another event after first one is processed
        // =====================================================================
        $display("\n=== Test 3: Second Event Transaction ===");
        
        // Wait a few cycles, then check if second event gets processed
        repeat(10) @(posedge clk);
        
        $display("T=%0t: Checking for second event processing...", $time);
        
        // Monitor for another 15 cycles
        for (int i = 0; i < 15; i++) begin
            @(posedge clk);
            
            if (event_bus.event_valid) begin
                $display("T=%0t: Second event! coord=(%0d,%0d), ack=%b", 
                         $time, event_bus.event_coord.x, event_bus.event_coord.y, 
                         event_bus.event_ack);
            end
        end
        
        // =====================================================================
        // Test 4: Test with no more events (empty FIFO)
        // =====================================================================
        $display("\n=== Test 4: No More Events (Empty FIFO) ===");
        
        repeat(10) @(posedge clk);
        
        $display("T=%0t: Final status:", $time);
        $display("  FIFO empty=%b, Capture active=%b, Conv active=%b", 
                 fifo_bus.empty, capture_ctrl.active, conv_ctrl.active);
        $display("  Event interface: valid=%b, ready=%b, ack=%b", 
                 event_bus.event_valid, event_bus.event_ready, event_bus.event_ack);
        
        #50;
        $display("\n=== Event Transaction Test Complete ===");
        $finish;
    end
    
    // =========================================================================
    // Monitors for Key Transactions
    // =========================================================================
    
    // Monitor event transactions
    always @(posedge clk) begin
        if (event_bus.event_valid && event_bus.event_ack) begin
            $display("*** EVENT TRANSACTION: coord=(%0d,%0d) transferred from capture to conv", 
                     event_bus.event_coord.x, event_bus.event_coord.y);
        end
    end
    
    // Monitor when fast_conv stores events
    always @(posedge clk) begin
        if (conv_inst.event_stored && conv_inst.state == conv_inst.EVENT_TRANSACTION) begin
            $display("*** CONV STORED EVENT: coord=(%0d,%0d) stored in fast_conv", 
                     conv_inst.event_coord.x, conv_inst.event_coord.y);
        end
    end

endmodule