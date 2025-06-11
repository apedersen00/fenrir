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
    logic empty;        // No data in the FIFO
    logic full;         // FIFO is full
    logic read_en;      // Read enable signal
    logic write_en;     // Write enable signal
    logic full_next;    // One sample before full
    logic almost_empty; // One sample before empty
    
    modport consumer(
        input  read_data,
        input  empty,
        input  almost_empty,
        output read_en
    );

    // FIFO module modport
    modport fifo_module(
        input  clk,
        input  rst_n,
        output read_data,
        output full,
        output full_next,
        output empty,
        output almost_empty,
        input  write_data,
        input  read_en,
        input  write_en
    );

endinterface