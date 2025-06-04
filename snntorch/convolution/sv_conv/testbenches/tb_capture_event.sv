// =============================================================================
// File: testbenches/tb_fifo_capture.sv
// Description: Integration testbench for FIFO + capture_event modules
// =============================================================================

import snn_interfaces_pkg::*;

module tb_fifo_capture;

    // Testbench parameters
    localparam int DATA_WIDTH = 16;  // 2 * DEFAULT_COORD_BITS for packed coordinates
    localparam int ADDR_WIDTH = 3;   // FIFO depth = 8
    localparam int FIFO_DEPTH = 2**ADDR_WIDTH;
    
    // Clock and reset
    logic clk = 0;
    logic rst_n;
    
    // Generate clock - 10ns period
    always #5 clk = ~clk;
    
    // Interface instances
    fifo_if #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) 
        fifo_bus (.clk(clk), .rst_n(rst_n));
    
    snn_event_if event_bus ();
    snn_control_if ctrl_bus ();
    
    // Module instances
    fifo #(.DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) 
        fifo_dut (.fifo_port(fifo_bus.fifo_module));
        
    capture_event #(
        .DATA_WIDTH(DATA_WIDTH),
        .IMG_HEIGHT(DEFAULT_IMG_HEIGHT),
        .IMG_WIDTH(DEFAULT_IMG_WIDTH)
    ) capture_dut (
        .fifo_port(fifo_bus.consumer),
        .conv_port(event_bus.capture),
        .ctrl_port(ctrl_bus.capture)
    );
    
    // Connect control signals
    assign ctrl_bus.clk = clk;
    
    // Simulate convolution module behavior
    logic conv_ready;
    logic conv_ack;
    assign event_bus.event_ready = conv_ready;
    assign event_bus.event_ack = conv_ack;
    
    // Verification task for coordinate conversion
    task verify_coordinates(input logic [15:0] packed_data, input vec2_t output_coord);
        vec2_t expected;
        expected = unpack_coordinates(packed_data);
        if (output_coord.x == expected.x && output_coord.y == expected.y) begin
            $display("T=%0t: ✓ Coordinate conversion correct: (%0d,%0d)", 
                     $time, output_coord.x, output_coord.y);
        end else begin
            $display("T=%0t: ✗ Coordinate conversion error!", $time);
            $display("         Expected: (%0d,%0d), Got: (%0d,%0d)", 
                     expected.x, expected.y, output_coord.x, output_coord.y);
        end
    endtask
    
    // Test procedure
    initial begin
        // Declare test variables
        logic [15:0] written_data;
        
        $display("=== FIFO + Capture Event Integration Test ===");
        $display("FIFO: DATA_WIDTH=%0d, DEPTH=%0d", DATA_WIDTH, FIFO_DEPTH);
        
        // Initialize signals
        rst_n = 0;
        ctrl_bus.enable = 0;
        ctrl_bus.reset = 1;  // Active high reset
        conv_ready = 0;
        conv_ack = 0;
        
        // FIFO producer signals
        fifo_bus.write_en = 0;
        fifo_bus.write_data = 0;
        
        // Reset sequence
        #20;
        rst_n = 1;
        ctrl_bus.reset = 0;  // Release reset
        #10;
        
        $display("T=%0t: Reset complete", $time);
        
        // =================================================================
        // Test 1: Write data to FIFO, enable capture_event, simulate conv ready
        // =================================================================
        $display("\n=== Test 1: Basic FIFO to Capture Flow ===");
        
        // Write test coordinates to FIFO
        // Test coordinate: x=10, y=20 → packed = {10, 20} = 0x0A14
        written_data = pack_coordinates({8'd10, 8'd20});
        @(posedge clk);
        fifo_bus.write_data = written_data;
        fifo_bus.write_en = 1;
        $display("T=%0t: Writing packed coordinates 0x%04h (x=10, y=20)", 
                 $time, fifo_bus.write_data);
        
        @(posedge clk);
        fifo_bus.write_en = 0;
        $display("T=%0t: FIFO state - empty=%b, full=%b", 
                 $time, fifo_bus.empty, fifo_bus.full);
        
        // Enable capture_event but convolution not ready yet
        @(posedge clk);
        ctrl_bus.enable = 1;
        $display("T=%0t: Enabled capture_event, conv_ready=%b", $time, conv_ready);
        
        // Wait a few cycles - should not read from FIFO yet
        repeat(3) @(posedge clk);
        $display("T=%0t: Capture active=%b (should be 0, conv not ready)", 
                 $time, ctrl_bus.active);
        
        // Now make convolution ready
        @(posedge clk);
        conv_ready = 1;
        $display("T=%0t: Convolution ready, capture should start", $time);
        
        // Monitor for a few cycles to see the data flow
        for (int i = 0; i < 8; i++) begin
            @(posedge clk);
            if (event_bus.event_valid && !conv_ack) begin
                $display("T=%0t: Event output! coord=(%0d,%0d), valid=%b", 
                         $time, 
                         event_bus.event_coord.x, 
                         event_bus.event_coord.y,
                         event_bus.event_valid);
                // Verify the coordinate conversion
                verify_coordinates(written_data, event_bus.event_coord);
                
                // Simulate convolution processing: wait 2 cycles then acknowledge
                repeat(2) @(posedge clk);
                conv_ack = 1;
                $display("T=%0t: Convolution acknowledging data consumption", $time);
                @(posedge clk);
                conv_ack = 0;
                $display("T=%0t: Convolution ack deasserted", $time);
            end
            $display("T=%0t: Capture active=%b, FIFO empty=%b", 
                     $time, ctrl_bus.active, fifo_bus.empty);
        end
        
        // We'll add more tests in next steps...
        
        #100;
        $display("\n=== Integration Test Complete ===");
        $finish;
    end
    
    // Monitor data flow
    always @(posedge clk) begin
        if (fifo_bus.read_en) begin
            $display("T=%0t: FIFO read_en asserted", $time);
        end
        if (event_bus.event_valid) begin
            $display("T=%0t: Event valid - coordinate output: (%0d,%0d)", 
                     $time, event_bus.event_coord.x, event_bus.event_coord.y);
        end
        if (event_bus.event_ack) begin
            $display("T=%0t: Event acknowledged by convolution", $time);
        end
    end
    always @(posedge clk) begin
        if (fifo_bus.read_en) begin
            $display("T=%0t: FIFO read_en asserted", $time);
        end
        if (event_bus.event_valid) begin
            $display("T=%0t: Event valid - coordinate output: (%0d,%0d)", 
                     $time, event_bus.event_coord.x, event_bus.event_coord.y);
        end
        if (event_bus.event_ack) begin
            $display("T=%0t: Event acknowledged by convolution", $time);
        end
    end

endmodule