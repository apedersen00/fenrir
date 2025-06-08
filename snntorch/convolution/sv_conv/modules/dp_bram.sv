module dp_bram #(
    parameter int DATA_WIDTH = 36,
    parameter int ADDR_WIDTH = 11,
    parameter string INIT_FILE = ""  // Optional initialization file
)(
    dp_bram_if.bram_module bram_port
);

    localparam int MEM_DEPTH = 2**ADDR_WIDTH;

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

endmodule