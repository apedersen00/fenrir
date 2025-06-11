module kernel_bram #(
    parameter int BITS_PER_KERNEL_WEIGHT,
    parameter int KERNEL_SIZE,
    parameter int IN_CHANNELS,
    parameter int OUT_CHANNELS,
    
    parameter int DATA_WIDTH = BITS_PER_KERNEL_WEIGHT * OUT_CHANNELS,
    // figure out how many kernel weights total
    parameter int TOTAL_KERNEL_POSITIONS = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE,
    parameter int ADDR_WIDTH = $clog2(TOTAL_KERNEL_POSITIONS),
    parameter string INIT_FILE = ""  // Optional initialization file
)(
    input logic clk,
    input logic rst_n,
    kernel_bram_if.bram_module bram_port
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
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // optional: handle sync reset if desired
        end else if (bram_port.en) begin
            // always read when enabled
            bram_port.data_out <= memory[bram_port.addr];

            // write when WE is asserted
            if (bram_port.we) begin
                memory[bram_port.addr] <= bram_port.data_in;
            end
        end
    end

endmodule