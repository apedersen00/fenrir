import snn_interfaces_pkg::*;

module tb_conv_multichannel;
    // parameters
    localparam int IN_CHANNELS  = 2,
                   OUT_CHANNELS = 2,
                   KERNEL_SIZE  = 3;

    // clock/reset for DUT
    logic clk = 1;
    logic rst_n = 0;
    always #5 clk = ~clk;  // 10 ns period

    logic event_valid = 0;
    logic event_ack = 0;

    output_vector_t test_event = 
    '{
        0, // timestep
        5, // x coordinate
        3, // y coordinate
        'b11 // spikes for each channel (2 channels, both active)
    };

    // instantiate the bus interface (no ports)
    kernel_bram_if #(
        .IN_CHANNELS  (IN_CHANNELS),
        .OUT_CHANNELS (OUT_CHANNELS),
        .KERNEL_SIZE  (KERNEL_SIZE)
    ) kernel_bram_bus ();

    // instantiate DUT: separate clk/rst, plus bus
    kernel_bram #(
        .KERNEL_WEIGHT_BITS(6),
        .KERNEL_SIZE       (KERNEL_SIZE),
        .IN_CHANNELS       (IN_CHANNELS),
        .OUT_CHANNELS      (OUT_CHANNELS)
    ) mem_kernel (
        .clk      (clk),
        .rst_n    (rst_n),
        .bram_port(kernel_bram_bus.bram_module)
    );

    // add convolution module to the testbench
    Convolution2d #(
        .IN_CHANNELS  (IN_CHANNELS),
        .OUT_CHANNELS (OUT_CHANNELS),
        .KERNEL_SIZE  (KERNEL_SIZE)
    ) conv_module (
        .clk      (clk),
        .rst_n    (rst_n),
        .bram_port(kernel_bram_bus.conv_module),
        .event_in (test_event),
        .event_valid(event_valid),
        .event_ack (event_ack)
    );

    initial begin
        // reset pulse
        rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;

        // wait for reset to settle
        repeat (2) @(posedge clk);
        event_valid = 1; // signal that an event is valid
        @(posedge event_ack);
        $display("Event acknowledged");
        event_valid = 0; // reset event valid
        // read it back
        kernel_bram_bus.we = 0;
        #10;
        $display("Read back: %h", kernel_bram_bus.data_out);

        #200;

        $finish;
    end
endmodule
