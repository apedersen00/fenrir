interface dp_bram_if #(
    parameter int DATA_WIDTH,
    parameter int ADDR_WIDTH
);

    // Clock and reset - driven by arbiter
    logic clk;
    logic rst_n;

    // Port A signals
    logic [ADDR_WIDTH-1:0] addr_a;      // Address for port A
    logic [DATA_WIDTH-1:0] data_in_a;   // Data input for port A
    logic [DATA_WIDTH-1:0] data_out_a;  // Data output from port A
    logic                  we_a;        // Write enable for port A
    logic                  en_a;        // Enable signal for port A
    
    // Port B signals  
    logic [ADDR_WIDTH-1:0] addr_b;      // Address for port B
    logic [DATA_WIDTH-1:0] data_in_b;   // Data input for port B
    logic [DATA_WIDTH-1:0] data_out_b;  // Data output from port B
    logic                  we_b;        // Write enable for port B
    logic                  en_b;        // Enable signal for port B

    // BRAM module modport - the actual memory block
    modport bram_module(
        input  clk,
        input  rst_n,
        input  addr_a,
        input  data_in_a,
        output data_out_a,
        input  we_a,
        input  en_a,
        input  addr_b,
        input  data_in_b,
        output data_out_b,
        input  we_b,
        input  en_b
    );

    // Arbiter modport - arbiter drives clock and controls both ports
    modport arbiter(
        output clk,
        output rst_n,
        output addr_a,
        output data_in_a,
        input  data_out_a,
        output we_a,
        output en_a,
        output addr_b,
        output data_in_b,
        input  data_out_b,
        output we_b,
        output en_b
    );

endinterface