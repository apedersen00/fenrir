// Step 1: Basic testbench structure

module fifo_tb;

    // Step 1a: Testbench parameters - match our FIFO configuration
    localparam int DATA_WIDTH = 8;
    localparam int ADDR_WIDTH = 3;  // Small for easier testing (depth = 8)
    localparam int FIFO_DEPTH = 2**ADDR_WIDTH;
    
    // Step 1b: Clock and reset signals
    logic clk = 0;
    logic rst_n;
    
    // Step 1c: Generate clock - 10ns period (100MHz)
    always #5 clk = ~clk;
    
    // Step 1d: Instantiate the FIFO interface (match your actual interface)
    fifo_if #(
        .DATA_WIDTH(DATA_WIDTH)
        // No ADDR_WIDTH parameter in your interface
    ) fifo_bus (.clk(clk), .rst_n(rst_n));
    
    // Step 1e: Instantiate the FIFO module under test
    fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .fifo_port(fifo_bus.fifo_module)
    );
    
    // Step 1f: Test procedure starts here
    initial begin
        // Initialize all signals (match your interface signal names)
        rst_n = 0;
        fifo_bus.write_en = 0;       // Your interface uses 'write_en'
        fifo_bus.read_en = 0;        // Your interface uses 'read_en' 
        fifo_bus.write_data = 0;
        
        // Reset sequence
        $display("=== FIFO Testbench Started ===");
        $display("FIFO Configuration: DATA_WIDTH=%0d, DEPTH=%0d", DATA_WIDTH, FIFO_DEPTH);
        
        #20;  // Hold reset for 20ns
        rst_n = 1;
        #10;  // Wait after reset release
        
        $display("Reset complete - FIFO ready");
        $display("Initial state: empty=%b, full=%b", fifo_bus.empty, fifo_bus.full);
        
        // =====================================================================
        // Test Scenario 1: Single Write
        // =====================================================================
        $display("\n=== Test 1: Single Write ===");
        
        // Write single data value
        @(posedge clk);
        fifo_bus.write_data = 8'hAA;  // Write 0xAA
        fifo_bus.write_en = 1;        // Use 'write_en'
        $display("T=%0t: Writing 0x%02h", $time, fifo_bus.write_data);
        
        @(posedge clk);
        fifo_bus.write_en = 0;        // Deassert write enable
        $display("T=%0t: After write - empty=%b, full=%b", $time, fifo_bus.empty, fifo_bus.full);
        
        // =====================================================================
        // Test Scenario 2: Single Read
        // =====================================================================
        $display("\n=== Test 2: Single Read ===");
        
        // Read the data we just wrote
        @(posedge clk);
        fifo_bus.read_en = 1;         // Use 'read_en'
        $display("T=%0t: Asserting read_en", $time);
        
        @(posedge clk);
        fifo_bus.read_en = 0;         // Deassert read enable
        $display("T=%0t: Read data = 0x%02h, empty=%b, full=%b", 
                 $time, fifo_bus.read_data, fifo_bus.empty, fifo_bus.full);
        
        // =====================================================================
        // Test Scenario 3: Multiple Writes
        // =====================================================================
        $display("\n=== Test 3: Multiple Writes ===");
        
        // Write sequence: 0x11, 0x22, 0x33
        for (int i = 0; i < 3; i++) begin
            @(posedge clk);
            fifo_bus.write_data = 8'h11 + (i * 8'h11);  // 0x11, 0x22, 0x33
            fifo_bus.write_en = 1;
            $display("T=%0t: Writing 0x%02h (item %0d)", $time, fifo_bus.write_data, i+1);
            
            @(posedge clk);
            fifo_bus.write_en = 0;
            $display("T=%0t: After write %0d - empty=%b, full=%b", 
                     $time, i+1, fifo_bus.empty, fifo_bus.full);
        end
        
        // =====================================================================
        // Test Scenario 4: Multiple Reads (Burst Reading)
        // =====================================================================
        $display("\n=== Test 4: Multiple Reads ===");
        
        // Read back the sequence: should get 0x11, 0x22, 0x33
        for (int i = 0; i < 3; i++) begin
            @(posedge clk);
            fifo_bus.read_en = 1;
            $display("T=%0t: Asserting read_en (read %0d)", $time, i+1);
            
            @(posedge clk);
            fifo_bus.read_en = 0;
            $display("T=%0t: Read data = 0x%02h (read %0d), empty=%b, full=%b", 
                     $time, fifo_bus.read_data, i+1, fifo_bus.empty, fifo_bus.full);
        end
        
        // =====================================================================
        // Test Scenario 5: Write, Wait, Write and Read (Mixed Operations)
        // =====================================================================
        $display("\n=== Test 5: Write, Wait, Write and Read ===");
        
        // First write
        @(posedge clk);
        fifo_bus.write_data = 8'hAA;
        fifo_bus.write_en = 1;
        $display("T=%0t: Writing 0x%02h", $time, fifo_bus.write_data);
        
        @(posedge clk);
        fifo_bus.write_en = 0;
        $display("T=%0t: After first write - empty=%b, full=%b", $time, fifo_bus.empty, fifo_bus.full);
        
        // Wait period (simulate processing delay)
        repeat(5) @(posedge clk);
        $display("T=%0t: After wait period - empty=%b, full=%b", $time, fifo_bus.empty, fifo_bus.full);
        
        // Second write
        @(posedge clk);
        fifo_bus.write_data = 8'hBB;
        fifo_bus.write_en = 1;
        $display("T=%0t: Writing 0x%02h", $time, fifo_bus.write_data);
        
        @(posedge clk);
        fifo_bus.write_en = 0;
        $display("T=%0t: After second write - empty=%b, full=%b", $time, fifo_bus.empty, fifo_bus.full);
        
        // Read first item
        @(posedge clk);
        fifo_bus.read_en = 1;
        $display("T=%0t: Reading first item", $time);
        
        @(posedge clk);
        fifo_bus.read_en = 0;
        $display("T=%0t: Read data = 0x%02h, empty=%b, full=%b", 
                 $time, fifo_bus.read_data, fifo_bus.empty, fifo_bus.full);
        
        // Test simultaneous read and write
        $display("T=%0t: Testing simultaneous read and write", $time);
        @(posedge clk);
        fifo_bus.write_data = 8'hCC;
        fifo_bus.write_en = 1;
        fifo_bus.read_en = 1;  // Read and write simultaneously!
        $display("T=%0t: Simultaneous: writing 0x%02h and reading", $time, fifo_bus.write_data);
        
        @(posedge clk);
        fifo_bus.write_en = 0;
        fifo_bus.read_en = 0;
        $display("T=%0t: After simultaneous op: read_data=0x%02h, empty=%b, full=%b", 
                 $time, fifo_bus.read_data, fifo_bus.empty, fifo_bus.full);
        
        // =====================================================================
        // Test Scenario 6: Fill to Full, then Drain to Empty
        // =====================================================================
        $display("\n=== Test 6: Fill to Full, then Drain to Empty ===");
        
        // First, read remaining item to start with empty FIFO
        @(posedge clk);
        fifo_bus.read_en = 1;
        @(posedge clk);
        fifo_bus.read_en = 0;
        $display("T=%0t: Cleared FIFO - empty=%b", $time, fifo_bus.empty);
        
        // Fill FIFO to capacity (8 items for ADDR_WIDTH=3)
        $display("T=%0t: Filling FIFO to capacity...", $time);
        for (int i = 0; i < FIFO_DEPTH; i++) begin
            @(posedge clk);
            fifo_bus.write_data = 8'h10 + i;  // 0x10, 0x11, 0x12, ..., 0x17
            fifo_bus.write_en = 1;
            
            @(posedge clk);
            fifo_bus.write_en = 0;
            $display("T=%0t: Wrote 0x%02h (item %0d/%0d) - full=%b", 
                     $time, 8'h10 + i, i+1, FIFO_DEPTH, fifo_bus.full);
        end
        
        // Try to write when full (should be ignored)
        @(posedge clk);
        fifo_bus.write_data = 8'hFF;
        fifo_bus.write_en = 1;
        $display("T=%0t: Attempting write when full (should be ignored)", $time);
        
        @(posedge clk);
        fifo_bus.write_en = 0;
        $display("T=%0t: After attempted write - full=%b", $time, fifo_bus.full);
        
        // Drain FIFO completely
        $display("T=%0t: Draining FIFO completely...", $time);
        for (int i = 0; i < FIFO_DEPTH; i++) begin
            @(posedge clk);
            fifo_bus.read_en = 1;
            
            @(posedge clk);
            fifo_bus.read_en = 0;
            $display("T=%0t: Read 0x%02h (item %0d/%0d) - empty=%b", 
                     $time, fifo_bus.read_data, i+1, FIFO_DEPTH, fifo_bus.empty);
        end
        
        // Try to read when empty (should be ignored)  
        @(posedge clk);
        fifo_bus.read_en = 1;
        $display("T=%0t: Attempting read when empty (should be ignored)", $time);
        
        @(posedge clk);
        fifo_bus.read_en = 0;
        $display("T=%0t: After attempted read - empty=%b", $time, fifo_bus.empty);
        
        // =====================================================================
        // Test Scenario 7: Some Writes and Reset
        // =====================================================================
        $display("\n=== Test 7: Some Writes and Reset ===");
        
        // Write a few items first
        $display("T=%0t: Writing some data before reset...", $time);
        for (int i = 0; i < 3; i++) begin
            @(posedge clk);
            fifo_bus.write_data = 8'hA0 + i;  // 0xA0, 0xA1, 0xA2
            fifo_bus.write_en = 1;
            
            @(posedge clk);
            fifo_bus.write_en = 0;
            $display("T=%0t: Wrote 0x%02h - empty=%b, full=%b", 
                     $time, 8'hA0 + i, fifo_bus.empty, fifo_bus.full);
        end
        
        $display("T=%0t: FIFO state before reset - empty=%b, full=%b", 
                 $time, fifo_bus.empty, fifo_bus.full);
        
        // Assert reset while FIFO has data
        @(posedge clk);
        rst_n = 0;
        $display("T=%0t: Reset asserted (FIFO should clear)", $time);
        
        // Hold reset for a few cycles
        repeat(3) @(posedge clk);
        $display("T=%0t: During reset - empty=%b, full=%b", $time, fifo_bus.empty, fifo_bus.full);
        
        // Release reset
        rst_n = 1;
        @(posedge clk);
        $display("T=%0t: Reset released - empty=%b, full=%b", $time, fifo_bus.empty, fifo_bus.full);
        
        // Test that FIFO works normally after reset
        $display("T=%0t: Testing normal operation after reset...", $time);
        @(posedge clk);
        fifo_bus.write_data = 8'hDD;
        fifo_bus.write_en = 1;
        
        @(posedge clk);
        fifo_bus.write_en = 0;
        $display("T=%0t: Post-reset write 0x%02h - empty=%b, full=%b", 
                 $time, fifo_bus.write_data, fifo_bus.empty, fifo_bus.full);
        
        @(posedge clk);
        fifo_bus.read_en = 1;
        
        @(posedge clk);
        fifo_bus.read_en = 0;
        $display("T=%0t: Post-reset read 0x%02h - empty=%b, full=%b", 
                 $time, fifo_bus.read_data, fifo_bus.empty, fifo_bus.full);
        
        #100;
        $display("=== Testbench Complete ===");
        $finish;
    end

endmodule