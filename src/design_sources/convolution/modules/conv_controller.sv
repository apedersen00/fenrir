import conv_pkg::*;

module CONV2D #(

    parameter int KERNEL_SIZE = DEFAULT_KERNEL_SIZE,
    parameter int IN_CHANNELS = DEFAULT_IN_CHANNELS,
    parameter int OUT_CHANNELS = DEFAULT_OUT_CHANNELS,
    parameter int IMG_HEIGHT = DEFAULT_IMG_HEIGHT,
    parameter int IMG_WIDTH = DEFAULT_IMG_WIDTH,

    parameter int BITS_PER_KERNEL_WEIGHT = DEFAULT_BITS_PER_KERNEL_WEIGHT,
    parameter int BITS_PER_NEURON = DEFAULT_BITS_PER_NEURON,
    parameter int INPUT_FIFO_EVENT_CAPACITY = DEFAULT_INPUT_FIFO_EVENT_CAPACITY,
    parameter int BITS_PER_COORDINATE = DEFAULT_BITS_PER_COORDINATE_IN
    
)(

    // Control signals
    input logic clk,
    input logic rst_n,
    input logic enable,

    // Input FIFO
    input logic input_write_enable,
    output logic input_fifo_full_next,
    input logic [BITS_PER_COORDINATE * 2 + IN_CHANNELS : 0] input_fifo_data, // plus one for timestep

    // output FIFO
    output logic [(BITS_PER_COORDINATE - 1) * 2 + OUT_CHANNELS : 0] output_fifo_data,
    output logic output_fifo_write_enable,
    input logic output_fifo_full_next
);


    // Lets calculate BRAM sizes
    localparam int MEM_KERNEL_DEPTH = KERNEL_SIZE * KERNEL_SIZE * IN_CHANNELS;
    localparam int MEM_KERNEL_WIDTH = OUT_CHANNELS * BITS_PER_KERNEL_WEIGHT;
    localparam int MEM_KERNEL_ADDRESS_WIDTH = $clog2(MEM_KERNEL_DEPTH);

    localparam int MEM_FEATURE_MAP_DEPTH = IMG_HEIGHT * IMG_WIDTH;
    localparam int MEM_FEATURE_MAP_WIDTH = OUT_CHANNELS * BITS_PER_NEURON;
    localparam int MEM_FEATURE_MAP_ADDRESS_WIDTH = $clog2(MEM_FEATURE_MAP_DEPTH);

    localparam int IN_FIFO_DATA_WIDTH = BITS_PER_COORDINATE * 2 + IN_CHANNELS + 1; // ts, x, y, spikes
    localparam int IN_FIFO_ADDR_WIDTH = $clog2(INPUT_FIFO_EVENT_CAPACITY);

    // ==========================================================
    // Input FIFO
    // ==========================================================

    
    logic fifo_full;
    logic fifo_empty;
    logic fifo_read_enable;
    logic [IN_FIFO_DATA_WIDTH-1:0] fifo_read_data;

    fifo_if #(
        .DATA_WIDTH(IN_FIFO_DATA_WIDTH),
        .ADDR_WIDTH(IN_FIFO_ADDR_WIDTH)
    ) input_fifo_interface (
        .clk(clk),
        .rst_n(rst_n)
    );
    // Producer side connections (from external)
    assign input_fifo_interface.write_data = input_fifo_data;
    assign input_fifo_interface.write_en = input_write_enable;
    assign input_fifo_full_next = input_fifo_interface.full_next;
    assign fifo_full = input_fifo_interface.full;
    
    fifo #(
        .DATA_WIDTH(IN_FIFO_DATA_WIDTH),
        .ADDR_WIDTH(IN_FIFO_ADDR_WIDTH)
    ) input_fifo_instance (
        .fifo_port(input_fifo_interface.fifo_module)
    );

    logic timestep;

    event_if #(
        .BITS_PER_COORDINATE(BITS_PER_COORDINATE),
        .IN_CHANNELS(IN_CHANNELS)
    ) event_interface (
        .clk(clk),
        .rst_n(rst_n)
    );
    
    capture_event #(
        .DATA_WIDTH(IN_FIFO_DATA_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .IMG_WIDTH(IMG_WIDTH),
        .BITS_PER_COORDINATE(BITS_PER_COORDINATE),
        .IN_CHANNELS(IN_CHANNELS)
    ) capture_event_instance(
        .timestep(timestep),
        .fifo_port(input_fifo_interface.consumer),
        .event_port(event_interface.capture)
    );

    // ==========================================================
    // BRAM For Feauter MAPS
    // ==========================================================

    dp_bram_if #(
        .DATA_WIDTH(MEM_FEATURE_MAP_WIDTH),
        .ADDR_WIDTH(MEM_FEATURE_MAP_ADDRESS_WIDTH)
    ) bram_feature_map_bus ();
    
    dp_bram #(
        .DATA_WIDTH(MEM_FEATURE_MAP_WIDTH),
        .ADDR_WIDTH(MEM_FEATURE_MAP_ADDRESS_WIDTH)
    ) bram_feature_map_instance (
        .bram_port(bram_feature_map_bus.bram_module)
    );

    // ==========================================================
    // ARBITER
    // ==========================================================

    arbiter_mode_t arbiter_mode = CONVOLUTION;

    arbiter_if #(
        .BITS_PER_COORDINATE(BITS_PER_COORDINATE),
        .OUT_CHANNELS(OUT_CHANNELS),
        .BITS_PER_NEURON(BITS_PER_NEURON)
    ) conv_read();
    arbiter_if #(
        .BITS_PER_COORDINATE(BITS_PER_COORDINATE),
        .OUT_CHANNELS(OUT_CHANNELS),
        .BITS_PER_NEURON(BITS_PER_NEURON)
    ) conv_write();
    arbiter_if #(
        .BITS_PER_COORDINATE(BITS_PER_COORDINATE),
        .OUT_CHANNELS(OUT_CHANNELS),
        .BITS_PER_NEURON(BITS_PER_NEURON)
    ) pool_read();
    arbiter_if #(
        .BITS_PER_COORDINATE(BITS_PER_COORDINATE),
        .OUT_CHANNELS(OUT_CHANNELS),
        .BITS_PER_NEURON(BITS_PER_NEURON)
    ) pool_write();

    arbiter #(
        .BITS_PER_COORDINATE(BITS_PER_COORDINATE),
        .OUT_CHANNELS(OUT_CHANNELS),
        .BITS_PER_NEURON(BITS_PER_NEURON),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) arbiter_instance(
        .clk(clk),
        .rst_n(rst_n),
        .mode(arbiter_mode),
        .conv_read_port(conv_read),
        .conv_write_port(conv_write),
        .pool_read_port(pool_read),
        .pool_write_port(pool_write),
        .bram_port(bram_feature_map_bus.arbiter)
    );



endmodule