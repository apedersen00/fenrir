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

    logic input_fifo_full;
    logic input_fifo_empty;

    fifo_if #(
        .DATA_WIDTH(IN_FIFO_DATA_WIDTH),
        .ADDR_WIDTH(IN_FIFO_ADDR_WIDTH)
    ) input_fifo_interface (
        .clk(clk),
        .rst_n(rst_n)
    );
    assign input_fifo_interface.write_data = input_fifo_data;
    assign input_fifo_interface.write_en = input_write_enable;
    assign input_fifo_full_next = input_fifo_interface.full_next;
    
    // Instances
    fifo #(
        .DATA_WIDTH(IN_FIFO_DATA_WIDTH),
        .ADDR_WIDTH(IN_FIFO_ADDR_WIDTH)
    ) input_fifo_instance (
        .fifo_port(input_fifo_interface.fifo_module)
    );


endmodule