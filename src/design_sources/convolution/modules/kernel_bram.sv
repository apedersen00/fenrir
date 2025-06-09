module kernel_bram #(
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
            $display("KERNEL_BRAM: Initialized from file %s", INIT_FILE);
        end else begin
            // Initialize to zero (optional - FPGA BRAMs usually start at 0)
            for (int i = 0; i < MEM_DEPTH; i++) begin
                memory[i] = '0;
            end
            $display("KERNEL_BRAM: Initialized to zeros, depth=%0d, width=%0d", MEM_DEPTH, DATA_WIDTH);
        end
    end

    // single port BRAM: Read/Write operations
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

endmodule