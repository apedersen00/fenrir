package conv_pkg;

    // CONVOLUTION PARAMETERS
    parameter int DEFAULT_KERNEL_SIZE = 3;
    parameter int DEFAULT_IN_CHANNELS = 2;
    parameter int DEFAULT_OUT_CHANNELS = 2;
    parameter int DEFAULT_IMG_HEIGHT = 8;
    parameter int DEFAULT_IMG_WIDTH = 8;

    // MEMORY PARMETERS
    parameter int DEFAULT_BITS_PER_KERNEL_WEIGHT = 6;
    parameter int DEFAULT_BITS_PER_NEURON = 9;
    parameter int DEFAULT_BITS_PER_COORDINATE_IN = $clog2(DEFAULT_IMG_WIDTH);
    parameter int DEFAULT_INPUT_FIFO_EVENT_CAPACITY = 1024;
    
    typedef struct packed {
        logic [DEFAULT_BITS_PER_COORDINATE_IN-1:0] x;
        logic [DEFAULT_BITS_PER_COORDINATE_IN-1:0] y;
    } vec2_t;

    typedef logic [DEFAULT_IN_CHANNELS-1:0] spike_vector_in_t;
    typedef struct packed {
        logic timestep;
        logic [DEFAULT_BITS_PER_COORDINATE_IN-1:0] x;
        logic [DEFAULT_BITS_PER_COORDINATE_IN-1:0] y;
        spike_vector_in_t spikes;
    } input_vector_t;

    typedef logic [DEFAULT_OUT_CHANNELS-1:0] spike_vector_out_t;
    typedef struct packed {
        logic timestep;
        logic [DEFAULT_BITS_PER_COORDINATE_IN-2:0] x;
        logic [DEFAULT_BITS_PER_COORDINATE_IN-2:0] y;
        spike_vector_out_t spikes;
    } output_vector_t;


    typedef logic signed [DEFAULT_BITS_PER_NEURON-1:0] feature_map_t [0:DEFAULT_OUT_CHANNELS-1];
    typedef logic signed [DEFAULT_BITS_PER_KERNEL_WEIGHT-1:0] kernel_weight_vector_t [0:DEFAULT_OUT_CHANNELS-1];

    function automatic feature_map_t add_kernel_weights_to_feature_map(
        input feature_map_t fm,
        input kernel_weight_vector_t kernel_weights
    );
        feature_map_t result;

        for (int channel = 0; channel < DEFAULT_OUT_CHANNELS; channel++) begin

            automatic logic signed [DEFAULT_BITS_PER_NEURON:0] result_before_clamp;
            result_before_clamp = fm[channel] + kernel_weights[channel];

            if (result_before_clamp > $signed({1'b0, {DEFAULT_BITS_PER_NEURON - 1{1'b1}}})) begin
                result[channel] = {1'b0, {DEFAULT_BITS_PER_NEURON - 1{1'b1}}}; // Clamp to max value
            end else if (result_before_clamp < $signed({1'b1, {DEFAULT_BITS_PER_NEURON - 1{1'b0}}})) begin
                result[channel] = {1'b1, {DEFAULT_BITS_PER_NEURON - 1{1'b0}}}; // Clamp to min value
            end else begin
                result[channel] = result_before_clamp; // No clamping needed
            end
        end

        return result;
    endfunction
    
    function automatic kernel_weight_vector_t bram_to_kernel_weight_vector(
        input logic [DEFAULT_OUT_CHANNELS * DEFAULT_BITS_PER_KERNEL_WEIGHT - 1:0] bram_data
    );

        kernel_weight_vector_t result;

        for (int channel = 0; channel < DEFAULT_OUT_CHANNELS; channel++) begin
            result[channel] = bram_data[channel * DEFAULT_BITS_PER_KERNEL_WEIGHT +: DEFAULT_BITS_PER_KERNEL_WEIGHT];
        end

        return result;

    endfunction

endpackage