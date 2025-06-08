// =============================================================================
// File: testbenches/tb_arbiter.sv
// Description: Testbench for arbiter module with pipelined read/write verification
// =============================================================================

import snn_interfaces_pkg::*;

module tb_arbiter;

    // Testbench parameters
    localparam int COORD_BITS = 4;        // Small for testing (16x16 image)
    localparam int CHANNELS = 4;          // 4 feature maps
    localparam int BITS_PER_CHANNEL = 8;  // 8-bit values
    localparam int IMG_WIDTH = 16;
    localparam int IMG_HEIGHT = 16;
    localparam int BRAM_DATA_WIDTH = CHANNELS * BITS_PER_CHANNEL;  // 32 bits total
    localparam int BRAM_ADDR_WIDTH = $clog2(IMG_WIDTH * IMG_HEIGHT);  // 8 bits
    
    // Clock and reset
    logic clk = 0;
    logic rst_n;
    
    // Generate clock - 10ns period
    always #5 clk = ~clk;
    
    // Interface instances
    dp_bram_if #(
        .DATA_WIDTH(BRAM_DATA_WIDTH),
        .ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) bram_bus ();
    
    snn_control_if ctrl_bus ();
    
    // Arbiter interfaces for conv module
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS), 
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) conv_read_bus ();
    
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL) 
    ) conv_write_bus ();
    
    // Arbiter interfaces for pool module
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) pool_read_bus ();
    
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) pool_write_bus ();
    
    // Module instances
    dp_bram #(
        .DATA_WIDTH(BRAM_DATA_WIDTH),
        .ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) bram_inst (
        .bram_port(bram_bus.bram_module)
    );
    
    arbiter #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .BRAM_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BRAM_ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) arbiter_inst (
        .clk(clk),
        .rst_n(rst_n),
        .ctrl_port(ctrl_bus.arbiter),
        .conv_read_port(conv_read_bus.arbiter),
        .conv_write_port(conv_write_bus.arbiter),
        .pool_read_port(pool_read_bus.arbiter),
        .pool_write_port(pool_write_bus.arbiter),
        .bram_port(bram_bus.arbiter)
    );
    
    // Connect control signals
    assign ctrl_bus.clk = clk;
    
    // Type definition for feature map arrays
    typedef logic [BITS_PER_CHANNEL-1:0] fm_array_t [CHANNELS-1:0];
    
    // Test data
    fm_array_t test_input_data [0:8];    // 3x3 kernel input data
    fm_array_t test_output_data [0:8];   // Expected convolution results
    fm_array_t read_result;
    
    // Initialize test data
    initial begin
        // Input pattern for 3x3 kernel at position (5,5)
        // Using simple incremental pattern
        for (int i = 0; i < 9; i++) begin
            for (int ch = 0; ch < CHANNELS; ch++) begin
                test_input_data[i][ch] = 8'h10 + (i * 4) + ch;  // 0x10, 0x11, 0x12, 0x13, 0x14, ...
                test_output_data[i][ch] = 8'h80 + (i * 4) + ch; // Expected conv results
            end
        end
    end
    
    // Task to write initial data to memory
    task automatic write_initial_data();
        $display("\n=== Writing Initial Test Data ===");
        
        // Write test pattern to a 3x3 region starting at (4,4)
        for (int row = 0; row < 3; row++) begin
            for (int col = 0; col < 3; col++) begin
                automatic vec2_t coord = '{x: 4+col, y: 4+row};
                automatic int idx = row * 3 + col;
                
                @(posedge clk);
                conv_write_bus.coord_wtr = coord;
                conv_write_bus.data_in = test_input_data[idx];
                conv_write_bus.write_req = 1'b1;
                
                $display("T=%0t: Writing to (%0d,%0d) data=[%h,%h,%h,%h]", 
                         $time, coord.x, coord.y,
                         test_input_data[idx][0], test_input_data[idx][1], 
                         test_input_data[idx][2], test_input_data[idx][3]);
                
                @(posedge clk);
                conv_write_bus.write_req = 1'b0;
            end
        end
        $display("Initial data write complete");
    endtask
    
    // Task to simulate 3x3 convolution with pipelined read/write
    task automatic test_pipelined_convolution();
        // 3x3 kernel positions relative to center (5,5)
        automatic vec2_t kernel_coords [0:8];
        automatic vec2_t result_coords [0:8];
        
        // Initialize kernel coordinates
        kernel_coords[0] = '{x:4, y:4}; kernel_coords[1] = '{x:5, y:4}; kernel_coords[2] = '{x:6, y:4};
        kernel_coords[3] = '{x:4, y:5}; kernel_coords[4] = '{x:5, y:5}; kernel_coords[5] = '{x:6, y:5};
        kernel_coords[6] = '{x:4, y:6}; kernel_coords[7] = '{x:5, y:6}; kernel_coords[8] = '{x:6, y:6};
        
        // Initialize result coordinates
        result_coords[0] = '{x:8, y:4}; result_coords[1] = '{x:9, y:4}; result_coords[2] = '{x:10, y:4};
        result_coords[3] = '{x:8, y:5}; result_coords[4] = '{x:9, y:5}; result_coords[5] = '{x:10, y:5};
        result_coords[6] = '{x:8, y:6}; result_coords[7] = '{x:9, y:6}; result_coords[8] = '{x:10, y:6};
        
        $display("\n=== Testing Pipelined 3x3 Convolution ===");
        $display("Pattern: Read kernel positions while writing results with 1-cycle offset");
        
        // Initialize request signals
        conv_read_bus.read_req = 1'b0;
        conv_write_bus.write_req = 1'b0;
        
        $display("Starting 10-cycle pipelined convolution...");
        
        // Cycle 1: First read only
        @(posedge clk);
        conv_read_bus.coord_get = kernel_coords[0];
        conv_read_bus.read_req = 1'b1;
        $display("T=%0t: Cycle 1 - Read coord (%0d,%0d)", 
                 $time, kernel_coords[0].x, kernel_coords[0].y);
        
        // Cycles 2-9: Simultaneous read next + write previous result
        for (int i = 1; i < 9; i++) begin
            @(posedge clk);
            
            // Read next position
            conv_read_bus.coord_get = kernel_coords[i];
            conv_read_bus.read_req = 1'b1;
            
            // Write result from previous cycle (offset by 1)
            conv_write_bus.coord_wtr = result_coords[i-1];
            conv_write_bus.data_in = test_output_data[i-1];
            conv_write_bus.write_req = 1'b1;
            
            $display("T=%0t: Cycle %0d - Read (%0d,%0d) + Write (%0d,%0d) data=[%h,%h,%h,%h]", 
                     $time, i+1,
                     kernel_coords[i].x, kernel_coords[i].y,
                     result_coords[i-1].x, result_coords[i-1].y,
                     test_output_data[i-1][0], test_output_data[i-1][1],
                     test_output_data[i-1][2], test_output_data[i-1][3]);
        end
        
        // Cycle 10: Final write only
        @(posedge clk);
        conv_read_bus.read_req = 1'b0;  // Stop reading
        conv_write_bus.coord_wtr = result_coords[8];
        conv_write_bus.data_in = test_output_data[8];
        conv_write_bus.write_req = 1'b1;
        
        $display("T=%0t: Cycle 10 - Final Write (%0d,%0d) data=[%h,%h,%h,%h]", 
                 $time, result_coords[8].x, result_coords[8].y,
                 test_output_data[8][0], test_output_data[8][1],
                 test_output_data[8][2], test_output_data[8][3]);
        
        @(posedge clk);
        conv_write_bus.write_req = 1'b0;
        
        $display("Pipelined convolution complete - 10 cycles total");
    endtask
    
    // Task to verify written results
    task automatic verify_convolution_results();
        automatic vec2_t result_coords [0:8];
        
        // Initialize result coordinates
        result_coords[0] = '{x:8, y:4}; result_coords[1] = '{x:9, y:4}; result_coords[2] = '{x:10, y:4};
        result_coords[3] = '{x:8, y:5}; result_coords[4] = '{x:9, y:5}; result_coords[5] = '{x:10, y:5};
        result_coords[6] = '{x:8, y:6}; result_coords[7] = '{x:9, y:6}; result_coords[8] = '{x:10, y:6};
        
        $display("\n=== Verifying Convolution Results ===");
        
        for (int i = 0; i < 9; i++) begin
            @(posedge clk);
            conv_read_bus.coord_get = result_coords[i];
            conv_read_bus.read_req = 1'b1;
            
            @(posedge clk);
            conv_read_bus.read_req = 1'b0;
            read_result = conv_read_bus.data_out;
            
            $display("T=%0t: Read result (%0d,%0d): [%h,%h,%h,%h] Expected: [%h,%h,%h,%h] %s", 
                     $time, result_coords[i].x, result_coords[i].y,
                     read_result[0], read_result[1], read_result[2], read_result[3],
                     test_output_data[i][0], test_output_data[i][1], 
                     test_output_data[i][2], test_output_data[i][3],
                     (read_result == test_output_data[i]) ? "✓" : "✗");
        end
    endtask
    
    // Task to test pool phase
    task automatic test_pool_phase();
        automatic fm_array_t expected_data = '{8'hAA, 8'hBB, 8'hCC, 8'hDD};
        
        $display("\n=== Testing Pool Phase Access ===");
        
        // Switch to pool phase
        ctrl_bus.conv_or_pool = 1'b0;
        #10;
        $display("T=%0t: Switched to POOL phase", $time);
        
        // Test pool read
        @(posedge clk);
        pool_read_bus.coord_get = '{x: 5, y: 5};
        pool_read_bus.read_req = 1'b1;
        $display("T=%0t: Pool reading from (5,5)", $time);
        
        @(posedge clk);
        pool_read_bus.read_req = 1'b0;
        read_result = pool_read_bus.data_out;
        $display("T=%0t: Pool read result: [%h,%h,%h,%h]", 
                 $time, read_result[0], read_result[1], read_result[2], read_result[3]);
        
        // Test pool write
        @(posedge clk);
        pool_write_bus.coord_wtr = '{x: 12, y: 12};
        pool_write_bus.data_in = expected_data;
        pool_write_bus.write_req = 1'b1;
        $display("T=%0t: Pool writing to (12,12): [AA,BB,CC,DD]", $time);
        
        @(posedge clk);
        pool_write_bus.write_req = 1'b0;
        
        // Verify pool write
        @(posedge clk);
        pool_read_bus.coord_get = '{x: 12, y: 12};
        pool_read_bus.read_req = 1'b1;
        
        @(posedge clk);
        pool_read_bus.read_req = 1'b0;
        read_result = pool_read_bus.data_out;
        
        // Check if data matches expected
        begin
            logic data_match = 1'b1;
            for (int i = 0; i < CHANNELS; i++) begin
                if (read_result[i] !== expected_data[i]) data_match = 1'b0;
            end
            
            $display("T=%0t: Pool verify read: [%h,%h,%h,%h] %s", 
                     $time, read_result[0], read_result[1], read_result[2], read_result[3],
                     data_match ? "✓" : "✗");
        end
    endtask
    
    // Main test procedure
    initial begin
        $display("=== Arbiter Pipelined Read/Write Testbench ===");
        $display("Configuration: %0dx%0d image, %0d channels, %0d bits/channel", 
                 IMG_WIDTH, IMG_HEIGHT, CHANNELS, BITS_PER_CHANNEL);
        
        // Initialize signals
        rst_n = 0;
        ctrl_bus.enable = 0;
        ctrl_bus.reset = 1;
        ctrl_bus.conv_or_pool = 1;  // Start in conv phase
        
        // Initialize interface signals
        conv_read_bus.read_req = 1'b0;
        conv_read_bus.coord_get = '{x: 0, y: 0};
        conv_write_bus.write_req = 1'b0;
        conv_write_bus.coord_wtr = '{x: 0, y: 0};
        conv_write_bus.data_in = '{default: '0};
        pool_read_bus.read_req = 1'b0;
        pool_read_bus.coord_get = '{x: 0, y: 0};
        pool_write_bus.write_req = 1'b0;
        pool_write_bus.coord_wtr = '{x: 0, y: 0};
        pool_write_bus.data_in = '{default: '0};
        
        // Reset sequence
        #20;
        rst_n = 1;
        ctrl_bus.reset = 0;
        ctrl_bus.enable = 1;
        #10;
        
        $display("T=%0t: Reset complete, arbiter enabled in CONV phase", $time);
        
        // Test sequence
        write_initial_data();
        #20;
        
        test_pipelined_convolution();
        #20;
        
        verify_convolution_results();
        #20;
        
        test_pool_phase();
        #50;
        
        $display("\n=== Arbiter Testbench Complete ===");
        $finish;
    end
    
    // Monitor arbiter activity
    always @(posedge clk) begin
        if (ctrl_bus.active) begin
            $display("    → ARBITER ACTIVE in %s phase", 
                     ctrl_bus.conv_or_pool ? "CONV" : "POOL");
        end
    end

endmodule