interface kernel_bram_if #(
    parameter int KERNEL_WEIGHT_BITS = 6,
    parameter int KERNEL_SIZE = 3,
    parameter int IN_CHANNELS = 6,
    parameter int OUT_CHANNELS = 6,
    
    parameter int DATA_WIDTH = KERNEL_WEIGHT_BITS * OUT_CHANNELS,
    // figure out how many kernel weights total
    parameter int TOTAL_KERNEL_POSITIONS = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
    parameter int ADDR_WIDTH = $clog2(TOTAL_KERNEL_POSITIONS)
);
    // Single port BRAM interface for kernel weights

    logic clk;
    logic rst_n;

    logic []