interface fifo_if #(
    parameter int DATA_WIDTH = 8,
    parameter int ADDR_WIDTH = 4
)(
    input logic clk,
    input logic rst_n
);

    // Data signals
    logic [DATA_WIDTH-1:0] read_data;
    logic [DATA_WIDTH-1:0] write_data;

    // Control signals
    logic empty;     // No data in the FIFO
    logic full;      // FIFO is full
    logic read_en;   // Read enable signal
    logic write_en;  // Write enable signal

    // Consumer modport (reads from FIFO)
    modport consumer(
        input  read_data,
        input  empty,
        output read_en,
        input  full
    );

    // Producer modport (writes to FIFO)
    modport producer(
        output write_data,
        output write_en,
        input  empty,
        input  full
    );

    // FIFO module modport - MUST have clk/rst_n
    modport fifo_module(
        input  clk,
        input  rst_n,
        output read_data,
        output full,
        output empty,
        input  write_data,
        input  read_en,
        input  write_en
    );

    // Debug/monitor modport - useful for timing analysis
    modport monitor(
        input clk,
        input rst_n,
        input read_data,
        input write_data,
        input full,
        input read_en,
        input write_en,
        input empty
    );

endinterface