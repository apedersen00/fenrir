package conv_pkg;
    // CONVOLUTION PARAMETERS
    //parameter int DEFAULT_KERNEL_SIZE = 3;
    //parameter int DEFAULT_IN_CHANNELS = 2;
    //parameter int DEFAULT_OUT_CHANNELS = 2;
    //parameter int DEFAULT_IMG_HEIGHT = 8;
    //parameter int DEFAULT_IMG_WIDTH = 8;
//
    //// MEMORY PARMETERS
    //parameter int DEFAULT_BITS_PER_KERNEL_WEIGHT = 6;
    //parameter int DEFAULT_BITS_PER_NEURON = 9;
    //parameter int DEFAULT_BITS_PER_COORDINATE_IN = $clog2(DEFAULT_IMG_WIDTH);
    //parameter int DEFAULT_INPUT_FIFO_EVENT_CAPACITY = 1024;
    
    //typedef struct packed {
    //    logic [DEFAULT_BITS_PER_COORDINATE_IN-1:0] x;
    //    logic [DEFAULT_BITS_PER_COORDINATE_IN-1:0] y;
    //} vec2_t;
//
    //typedef logic [DEFAULT_IN_CHANNELS-1:0] spike_vector_in_t;
    //typedef struct packed {
    //    logic timestep;
    //    logic [DEFAULT_BITS_PER_COORDINATE_IN-1:0] x;
    //    logic [DEFAULT_BITS_PER_COORDINATE_IN-1:0] y;
    //    spike_vector_in_t spikes;
    //} input_vector_t;
//
    //typedef logic [DEFAULT_OUT_CHANNELS-1:0] spike_vector_out_t;
    //typedef struct packed {
    //    logic timestep;
    //    logic [DEFAULT_BITS_PER_COORDINATE_IN-2:0] x;
    //    logic [DEFAULT_BITS_PER_COORDINATE_IN-2:0] y;
    //    spike_vector_out_t spikes;
    //} output_vector_t;


    //typedef logic signed [DEFAULT_BITS_PER_NEURON-1:0] feature_map_t [0:DEFAULT_OUT_CHANNELS-1];
    //typedef logic signed [DEFAULT_BITS_PER_KERNEL_WEIGHT-1:0] kernel_weight_vector_t [0:DEFAULT_OUT_CHANNELS-1];

    typedef enum logic{
        CONVOLUTION,
        POOLING
    } arbiter_mode_t;


    
    
    

endpackage