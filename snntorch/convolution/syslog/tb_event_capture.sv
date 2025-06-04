// File: tb_event_capture.sv (FIXED VERSION)
import conv_pool_pkg::*;

module tb_event_capture;

    // =================================================================
    // TEST PARAMETERS
    // =================================================================
    
    parameter int CLK_PERIOD = 10;  // 10ns = 100MHz
    parameter int COORD_BITS = 8;
    parameter int IMG_WIDTH = 32;
    parameter int IMG_HEIGHT = 32;
    
    // =================================================================
    // DUT SIGNALS (matching actual module ports)
    // =================================================================
    
    // Clock and control
    logic clk = 0;
    logic reset_ni = 0;               // BUG FIX 6: Match module port name
    logic enable_i = 0;
    
    // FIFO interface (matching module port names)
    logic fifo_empty_i = 1;
    logic [2*COORD_BITS-1:0] fifo_data_i = 0;
    logic fifo_read_o;
    
    // Output interface (matching module port names)
    coord_t captured_event_coord_o;
    logic captured_event_valid_o;
    logic captured_event_processed_i = 0;
    
    // =================================================================
    // CLOCK GENERATION
    // =================================================================
    
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // =================================================================
    // DUT INSTANTIATION (BUG FIX 7: Correct port connections)
    // =================================================================
    
    event_capture #(
        .COORD_BITS(COORD_BITS),
        .IMG_WIDTH(IMG_WIDTH), 
        .IMG_HEIGHT(IMG_HEIGHT)
    ) dut (
        .clk(clk),
        .reset_ni(reset_ni),                           // Fixed: correct port name
        .enable_i(enable_i),                           // Fixed: correct port name
        .fifo_empty_i(fifo_empty_i),                   // Fixed: correct port name
        .fifo_data_i(fifo_data_i),                     // Fixed: correct port name
        .fifo_read_o(fifo_read_o),                     // Fixed: correct port name
        .captured_event_coord_o(captured_event_coord_o), // Fixed: correct port name
        .captured_event_valid_o(captured_event_valid_o), // Fixed: correct port name
        .captured_event_processed_i(captured_event_processed_i) // Fixed: correct port name
    );
    
    // =================================================================
    // TEST UTILITIES
    // =================================================================
    
    task automatic send_coordinates(
        input int x, 
        input int y,
        input int wait_cycles = 0
    );
        coord_t coord;
        coord.x = x;
        coord.y = y;
        
        $display("[%t] Sending coordinates (%0d,%0d)", $time, x, y);
        
        // Step 1: Put data on FIFO bus (like real FIFO would)
        
        fifo_empty_i = 0;
        $display("[%t] FIFO: Data ready on bus", $time);
        
        // Step 2: Wait for DUT to request read
        wait(fifo_read_o == 1);
        $display("[%t] FIFO: Read request detected", $time);
        
        // Step 3: FIFO advances on NEXT clock edge (realistic timing)
        @(posedge clk);
        $display("[%t] FIFO: Clock edge - processing read", $time);
        
        // Step 4: FIFO state changes AFTER the clock edge
        #1;  // Small delay to avoid race conditions
        fifo_empty_i = 1;           // FIFO becomes empty
        fifo_data_i = pack_coordinates(coord);
        $display("[%t] FIFO: Advanced (now empty)", $time);
        
        if (wait_cycles > 0) begin
            repeat(wait_cycles) @(posedge clk);
        end
    endtask

// ALSO REPLACE your reset_dut task for better timing:

    task automatic reset_dut();
        $display("[%t] Resetting DUT...", $time);
        reset_ni = 0;
        enable_i = 0;
        fifo_empty_i = 1;
        fifo_data_i = 16'h0000;
        captured_event_processed_i = 0;

        // Use clock-based timing instead of #delay
        repeat(3) @(posedge clk);   // Hold reset for 3 clock cycles
        reset_ni = 1;
        @(posedge clk);             // Wait one more cycle
        $display("[%t] Reset complete", $time);
    endtask
    
    task automatic consume_data(output coord_t received_coord);
        $display("[%t] Waiting for valid data...", $time);
        
        // Wait for data_valid to go high
        wait(captured_event_valid_o == 1);      // BUG FIX 13: Use correct signal name
        received_coord = captured_event_coord_o; // BUG FIX 14: Use correct signal name
        
        $display("[%t] Received valid data: %s", 
                $time, coord_to_string(received_coord));
        
        #(CLK_PERIOD);
        // Acknowledge consumption
        captured_event_processed_i = 1;         // BUG FIX 15: Use correct signal name
        #(CLK_PERIOD);
        captured_event_processed_i = 0;
        #(CLK_PERIOD);
        
        // Wait for valid to go low
        wait(captured_event_valid_o == 0);      // BUG FIX 16: Use correct signal name
        $display("[%t] Data consumed", $time);
    endtask
    
    function automatic void assert_coord_equal(
        input coord_t expected,
        input coord_t actual,
        input string test_name
    );
        if (expected.x == actual.x && expected.y == actual.y) begin
            $display("[%t] PASS: %s - Expected %s, Got %s", 
                    $time, test_name, 
                    coord_to_string(expected), 
                    coord_to_string(actual));
        end else begin
            $error("[%t] FAIL: %s - Expected %s, Got %s", 
                   $time, test_name,
                   coord_to_string(expected),
                   coord_to_string(actual));
        end
    endfunction
    
    // =================================================================
    // TEST CASES
    // =================================================================
    
    task automatic test_basic_capture();
        coord_t sent_coord, received_coord;
        
        $display("\n=== TEST: Basic Coordinate Capture ===");
        
        reset_dut();
        enable_i = 1;                           // BUG FIX 17: Use correct signal name
        
        sent_coord.x = 10;
        sent_coord.y = 15;
        
        fork
            send_coordinates(sent_coord.x, sent_coord.y);
            consume_data(received_coord);
        join
        
        assert_coord_equal(sent_coord, received_coord, "Basic Capture");
    endtask
    
    task automatic test_boundary_coordinates();
        coord_t test_coords[] = '{
            '{x: 0, y: 0},
            '{x: IMG_WIDTH-1, y: IMG_HEIGHT-1},
            '{x: 15, y: 31}
        };
        coord_t received_coord;
        
        $display("\n=== TEST: Boundary Coordinates ===");
        
        reset_dut();
        enable_i = 1;
        
        foreach (test_coords[i]) begin
            $display("[%t] Testing boundary coord %0d: %s", 
                    $time, i, coord_to_string(test_coords[i]));
            
            fork
                send_coordinates(test_coords[i].x, test_coords[i].y);
                consume_data(received_coord);
            join
            
            assert_coord_equal(test_coords[i], received_coord, 
                             $sformatf("Boundary Test %0d", i));
        end
    endtask
    
    task automatic test_invalid_coordinates();
        coord_t invalid_coords[] = '{
            '{x: IMG_WIDTH, y: 10},
            '{x: 10, y: IMG_HEIGHT},
            '{x: 255, y: 255}
        };
        
        $display("\n=== TEST: Invalid Coordinates (Should be Rejected) ===");
        
        reset_dut();
        enable_i = 1;
        
        foreach (invalid_coords[i]) begin
            $display("[%t] Testing invalid coord %0d: %s", 
                    $time, i, coord_to_string(invalid_coords[i]));
            
            send_coordinates(invalid_coords[i].x, invalid_coords[i].y, 5);
            
            // Should NOT receive valid data
            if (captured_event_valid_o) begin       // BUG FIX 18: Use correct signal name
                $error("[%t] FAIL: Invalid coordinate was accepted!", $time);
            end else begin
                $display("[%t] PASS: Invalid coordinate correctly rejected", $time);
            end
        end
    endtask
    
    task automatic test_enable_disable();
        coord_t test_coord = '{x: 20, y: 25};
        coord_t received_coord;   // ✅ FIX 1: Declare variable at top with semicolon
        
        $display("\n=== TEST: Enable/Disable Functionality ===");
        
        reset_dut();
        enable_i = 0;  // Start disabled
        
        send_coordinates(test_coord.x, test_coord.y, 3);
        
        if (fifo_read_o) begin
            $error("[%t] FAIL: DUT reading FIFO while disabled!", $time);
        end else begin
            $display("[%t] PASS: DUT correctly ignores FIFO when disabled", $time);
        end
        
        enable_i = 1;
        #(CLK_PERIOD * 2);
        
        fork
            send_coordinates(test_coord.x, test_coord.y);
            consume_data(received_coord);    // ✅ FIX 2: Added semicolon
        join
        
        assert_coord_equal(test_coord, received_coord, "Enable/Disable Test");
    endtask
    
    // =================================================================
    // MAIN TEST SEQUENCE
    // =================================================================
    
    initial begin : main_test_sequence
        $display("=================================================");
        $display("Starting Event Capture Testbench");
        $display("=================================================");
        
        test_basic_capture();
        test_boundary_coordinates();
        test_invalid_coordinates();
        test_enable_disable();
        
        $display("\n=================================================");
        $display("All tests completed!");
        $display("=================================================");
        
        #(CLK_PERIOD * 10);
        $finish;
    end
    
    // Timeout watchdog
    initial begin : timeout_watchdog
        #(CLK_PERIOD * 1000);
        $fatal("Simulation timeout! Tests took too long.");
    end
    
    // Waveform dumping
    initial begin : waveform_dump
        $dumpfile("event_capture.vcd");
        $dumpvars(0, tb_event_capture);
    end

endmodule