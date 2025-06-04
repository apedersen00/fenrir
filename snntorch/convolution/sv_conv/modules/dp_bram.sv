// =============================================================================
// File: modules/dp_bram.sv
// Description: Dual Port Block RAM Module - Updated for new interface
// =============================================================================

module dp_bram #(
    parameter int DATA_WIDTH = 36,
    parameter int ADDR_WIDTH = 11,
    parameter string INIT_FILE = ""  // Optional initialization file
)(
    dp_bram_if.bram_module bram_port
);

    // Calculate memory depth
    localparam int MEM_DEPTH = 2**ADDR_WIDTH;

    // Memory array - use synthesis attributes for FPGA BRAM inference
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] memory [0:MEM_DEPTH-1];

    // Optional memory initialization
    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, memory);
            $display("DP_BRAM: Initialized from file %s", INIT_FILE);
        end else begin
            // Initialize to zero (optional - FPGA BRAMs usually start at 0)
            for (int i = 0; i < MEM_DEPTH; i++) begin
                memory[i] = '0;
            end
            $display("DP_BRAM: Initialized to zeros, depth=%0d, width=%0d", MEM_DEPTH, DATA_WIDTH);
        end
    end

    // Port A: Read/Write operations
    always_ff @(posedge bram_port.clk) begin
        if (bram_port.en_a) begin
            // Read operation (always happens when enabled)
            bram_port.data_out_a <= memory[bram_port.addr_a];
            
            // Write operation (only when write enable is asserted)
            if (bram_port.we_a) begin
                memory[bram_port.addr_a] <= bram_port.data_in_a;
            end
        end
    end

    // Port B: Read/Write operations  
    always_ff @(posedge bram_port.clk) begin
        if (bram_port.en_b) begin
            // Read operation (always happens when enabled)
            bram_port.data_out_b <= memory[bram_port.addr_b];
            
            // Write operation (only when write enable is asserted)
            if (bram_port.we_b) begin
                memory[bram_port.addr_b] <= bram_port.data_in_b;
            end
        end
    end

    // Collision detection and warnings (for debugging)
    always_ff @(posedge bram_port.clk) begin
        if (bram_port.en_a && bram_port.en_b && 
            bram_port.addr_a == bram_port.addr_b &&
            (bram_port.we_a || bram_port.we_b)) begin
            
            $warning("DP_BRAM: Address collision at 0x%h - PortA(we=%b) PortB(we=%b)", 
                    bram_port.addr_a, bram_port.we_a, bram_port.we_b);
        end
    end

    // Optional: Performance monitoring (can be disabled in synthesis)
    `ifndef SYNTHESIS
    int read_count_a = 0;
    int write_count_a = 0; 
    int read_count_b = 0;
    int write_count_b = 0;

    always_ff @(posedge bram_port.clk) begin
        if (bram_port.en_a) begin
            read_count_a <= read_count_a + 1;
            if (bram_port.we_a) write_count_a <= write_count_a + 1;
        end
        if (bram_port.en_b) begin
            read_count_b <= read_count_b + 1;
            if (bram_port.we_b) write_count_b <= write_count_b + 1;
        end
    end

    // Debug task to print memory contents
    task automatic print_memory_range(input int start_addr, input int end_addr);
        $display("=== DP_BRAM Memory Contents [0x%h:0x%h] ===", start_addr, end_addr);
        for (int i = start_addr; i <= end_addr && i < MEM_DEPTH; i++) begin
            $display("  [0x%h] = 0x%h", i, memory[i]);
        end
    endtask

    // Debug task to print statistics
    task automatic print_stats();
        $display("=== DP_BRAM Usage Statistics ===");
        $display("  Port A: %0d reads, %0d writes", read_count_a, write_count_a);
        $display("  Port B: %0d reads, %0d writes", read_count_b, write_count_b);
    endtask
    `endif

endmodule