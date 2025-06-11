

module tb_conv_multichannel;
    // parameters
    localparam int  IN_CHANNELS  = 2,
                    OUT_CHANNELS = 2,
                    KERNEL_SIZE  = 3;
    localparam int  BITS_PER_CHANNEL = 6;
    localparam int  COORD_BITS = 8; // bits for coordinates
    localparam int  IMG_WIDTH = 8;
    localparam int  IMG_HEIGHT = 8;
    localparam int  BRAM_DATA_WIDTH = OUT_CHANNELS * BITS_PER_CHANNEL;
    localparam int  BRAM_ADDR_WIDTH = $clog2(IMG_WIDTH * IMG_HEIGHT);
                    


    // clock/reset for DUT
    logic clk = 1;
    logic rst_n = 0;
    always #5 clk = ~clk;  // 10 ns period

    logic event_valid = 0;
    logic event_ack = 0;

    input_vector_t test_events [0:3] = '{
        '{timestep: 1'b0, x: 8'd5, y: 8'd3, spikes: 2'b11},  // Event 0
        '{timestep: 1'b0, x: 8'd5, y: 8'd4, spikes: 2'b11},  // Event 1  
        '{timestep: 1'b0, x: 8'd0, y: 8'd0, spikes: 2'b01},
        '{timestep: 1'b0, x: 8'd0, y: 8'd5, spikes: 2'b10}   // Event 2
    };

    logic [1:0] event_phase = 0;  // 2 bits for 3 events (0, 1, 2)

// Select current event based on phase
input_vector_t test_event;
assign test_event = test_events[event_phase];

    logic event_phase = 0;
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
        .OUT_CHANNELS      (OUT_CHANNELS),
        .INIT_FILE         ("C:\\Users\\alext\\fenrir\\python\\notebooks\\kernel_weights.mem")

    ) mem_kernel (
        .clk      (clk),
        .rst_n    (rst_n),
        .bram_port(kernel_bram_bus.bram_module)
    );

    dp_bram_if #(
        .DATA_WIDTH(BRAM_DATA_WIDTH),
        .ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) bram_bus ();
    
    snn_control_if ctrl_bus ();
    
    // Arbiter interfaces for conv module
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(OUT_CHANNELS), 
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) conv_read_bus ();
    
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(OUT_CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL) 
    ) conv_write_bus ();
    
    // Arbiter interfaces for pool module
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(OUT_CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) pool_read_bus ();
    
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(OUT_CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) pool_write_bus ();
    
    // Module instances
    dp_bram #(
        .DATA_WIDTH(BRAM_DATA_WIDTH),
        .ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) bram_inst (
        .bram_port(bram_bus.bram_module)
    );
    
    arbiter #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(OUT_CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .BRAM_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BRAM_ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) arbiter_inst (
        .clk(clk),
        .rst_n(rst_n),
        .ctrl_port(ctrl_bus.arbiter),
        .conv_read_port(conv_read_bus.arbiter),
        .conv_write_port(conv_write_bus.arbiter),
        .pool_read_port(pool_read_bus.arbiter),
        .pool_write_port(pool_write_bus.arbiter),
        .bram_port(bram_bus.arbiter)
    );

    // add convolution module to the testbench
    Convolution2d #(
        .IN_CHANNELS  (IN_CHANNELS),
        .OUT_CHANNELS (OUT_CHANNELS),
        .KERNEL_SIZE  (KERNEL_SIZE)
    ) conv_module (
        .clk      (clk),
        .rst_n    (rst_n),
        .mem_kernel(kernel_bram_bus.conv_module),
        .event_in (test_event),
        .event_valid(event_valid),
        .event_ack (event_ack),
        .mem_read (conv_read_bus.read_port),
        .mem_write(conv_write_bus.write_port)
    );

    assign ctrl_bus.arbiter.conv_or_pool = 1;
    assign ctrl_bus.arbiter.enable = 1;
    assign ctrl_bus.arbiter.reset = 1;
    

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
        
        @(posedge !ctrl_bus.active);
        $writememb("test_mem0.mem", bram_inst.memory);
        event_phase = 1;
        @(posedge clk);
        event_valid = 1; // signal that an event is valid
        @(posedge event_ack);
        event_valid = 0; // reset event valid

        @(posedge !ctrl_bus.active);
        $writememb("test_mem1.mem", bram_inst.memory);
        event_phase = 2;
        @(posedge clk);
        event_valid = 1; // signal that an event is valid
        @(posedge event_ack);
        event_valid = 0; // reset event valid

        @(posedge !ctrl_bus.active);
        $writememb("test_mem2.mem", bram_inst.memory);
        event_phase = 3;
        @(posedge clk);
        event_valid = 1; // signal that an event is valid
        @(posedge event_ack);
        event_valid = 0; // reset event valid


        @(posedge !ctrl_bus.active);
        #(100);
        $writememb("test_mem3.mem", bram_inst.memory);
        $display("Test completed, memory contents written to test_mem.mem");
        $finish;
    end
endmodule
