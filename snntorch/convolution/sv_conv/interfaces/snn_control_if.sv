interface snn_control_if;

    logic pause;
    logic enable;
    logic reset;
    logic clk;
    logic active;
    logic ready;
    logic conv_or_pool; // 1 for conv, 0 for pool

    modport top(
        output enable,
        output reset,
        output clk,
        input active,
        input ready
    );

    modport capture(
        input enable,
        input reset,
        input clk,
        output active
    );

    modport convolution(
        input enable,
        input reset,
        input clk,
        output active,
        output ready
    );

    modport monitor(
        input enable,
        input reset,
        input clk,	
        input active,
        input ready
    );

    modport arbiter(
        input enable,
        input reset,
        input conv_or_pool,
        output active,
        output ready
    );

endinterface