interface kernel_bram_if #(
    parameter int KERNEL_WEIGHT_BITS = 6,
    parameter int KERNEL_SIZE = 3,
    parameter int IN_CHANNELS = 6,
    parameter int OUT_CHANNELS = 6,
    
    parameter int DATA_WIDTH = KERNEL_WEIGHT_BITS * OUT_CHANNELS,
    // figure out how many kernel weights total
    parameter int TOTAL_KERNEL_POSITIONS = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE,
    parameter int ADDR_WIDTH = $clog2(TOTAL_KERNEL_POSITIONS)
);
    // Single port BRAM interface for kernel weights
    logic [ADDR_WIDTH-1:0] addr;      // Address for kernel weights
    logic [DATA_WIDTH-1:0] data_in;   // Data input for kernel weights
    logic [DATA_WIDTH-1:0] data_out;  // Data output from kernel weights
    logic                  we;        // Write enable for kernel weights
    logic                  en;        // Enable signal for kernel weights

    modport bram_module(
        input addr,
        input data_in,
        output data_out,
        input we,
        input en
    );

    modport conv_module(
        output en,
        output addr,
        input data_out
    );

endinterface