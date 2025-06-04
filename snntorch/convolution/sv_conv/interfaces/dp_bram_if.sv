// =============================================================================
// File: interfaces/dp_bram_if.sv
// Description: Dual Port Block RAM Interface - Complete for Arbiter Use
// =============================================================================

interface dp_bram_if #(
    parameter int DATA_WIDTH = 16,    // Width of data (for feature map arrays)
    parameter int ADDR_WIDTH = 10     // Address width (1024 locations default)
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

    // Monitor modport for debugging and verification
    modport monitor(
        input clk,
        input rst_n,
        input addr_a,
        input data_in_a,
        input data_out_a,
        input we_a,
        input en_a,
        input addr_b,
        input data_in_b,
        input data_out_b,
        input we_b,
        input en_b
    );

    // Convenience functions for debugging
    function automatic string port_a_status();
        return $sformatf("PortA: en=%b we=%b addr=0x%h din=0x%h dout=0x%h", 
                        en_a, we_a, addr_a, data_in_a, data_out_a);
    endfunction

    function automatic string port_b_status();
        return $sformatf("PortB: en=%b we=%b addr=0x%h din=0x%h dout=0x%h", 
                        en_b, we_b, addr_b, data_in_b, data_out_b);
    endfunction

    function automatic string bram_status();
        return $sformatf("BRAM: PortA(en=%b,we=%b) PortB(en=%b,we=%b)", 
                        en_a, we_a, en_b, we_b);
    endfunction

endinterface