package snn_interfaces_pkg;

    parameter int DEFAULT_COORD_BITS = 8;
    parameter int DEFAULT_IMG_WIDTH = 32;
    parameter int DEFAULT_IMG_HEIGHT = 32;
    parameter int DEFAULT_NEURON_BITS = 9;
    parameter int DEFAULT_CHANNELS = 2; // refactor later
    parameter int DEFAULT_INPUT_FIFO_EVENT_CAPACITY = 1024;

    parameter int DEFAULT_KERNEL_BITS = 6;
    parameter int DEFAULT_IN_CHANNELS = 6;
    parameter int DEFAULT_OUT_CHANNELS = 6;
    
    //
    parameter int OUTPUT_FIFO_DATA_WIDTH = (DEFAULT_COORD_BITS - 1) * 2 + DEFAULT_CHANNELS + 1;

    typedef struct packed {
        logic [DEFAULT_COORD_BITS-1:0] x;
        logic [DEFAULT_COORD_BITS-1:0] y;
    } vec2_t;
    
    typedef logic signed [DEFAULT_NEURON_BITS-1:0] feature_map_t [0:DEFAULT_CHANNELS-1];


    //only for 3x3 kernels
    parameter logic signed [DEFAULT_KERNEL_BITS-1:0] kernel_weights [0:8][0:DEFAULT_CHANNELS -1] =
    '{
        '{1, -1},//0,  0,  1,  1}, 
        '{2, -1},//0,  0,  1,  2},
        '{1, -1},//0,  0,  1,  1},
        '{ 0, 0},//, -2, -1,  1,  2},
        '{ 0, 0},//,  0,  0,  0,  0},
        '{ 0, 0},//,  2,  1, -1, -2},
        '{-1, 1},//,  0,  0, -1, -1},
        '{-2, 1},//,  0,  0, -1, -2},
        '{-1, 1}//  0,  0, -1, -1, 1}// 
    };

    // Decay vector ( represented as feature maps)
    parameter feature_map_t decay_vector = '{-3, -3};
    
    // threshold vector ( represented as feature maps)
    parameter feature_map_t threshold_vector = '{-12, 20};

    
    typedef logic [DEFAULT_CHANNELS-1:0] spike_vector_t;
    typedef struct packed {
        logic timestep;
        logic [DEFAULT_COORD_BITS-2:0] x;
        logic [DEFAULT_COORD_BITS-2:0] y;
        spike_vector_t spikes;
    } output_vector_t; // TODO invent a generalized vector type for tranmission between layers

    function automatic string coord_to_string(input vec2_t coord);
        return $sformatf("(%0d,%0d)", coord.x, coord.y);
    endfunction

    function automatic vec2_t unpack_coordinates(
        input logic [2 * DEFAULT_COORD_BITS - 1 : 0] packed_coords
    );
        vec2_t result;
        result.x = packed_coords[2 * DEFAULT_COORD_BITS - 1 : DEFAULT_COORD_BITS];
        result.y = packed_coords[DEFAULT_COORD_BITS - 1 : 0];
        return result;
    endfunction

    function automatic logic [2 * DEFAULT_COORD_BITS - 1 : 0] pack_coordinates(
        input vec2_t coord
    );
        logic [2 * DEFAULT_COORD_BITS - 1 : 0] packed_coords;
        packed_coords = {coord.x, coord.y};
        return packed_coords;
    endfunction

    // Pack output vector into a single logic vector
    function automatic logic [OUTPUT_FIFO_DATA_WIDTH - 1: 0] pack_output_vector(
        input output_vector_t output_vector
    );
        return output_vector;
    endfunction


    function automatic feature_map_t apply_kernel_weights(
        input feature_map_t current_feature_map,
        input logic [3:0] kernel_position
    );
        automatic feature_map_t updated_feature_map;
        
        for (int ch = 0; ch < DEFAULT_CHANNELS; ch++) begin

            automatic logic signed [DEFAULT_NEURON_BITS:0] temp_result;
            temp_result = current_feature_map[ch] + kernel_weights[kernel_position][ch];
            
            if (temp_result > $signed({1'b0, {DEFAULT_NEURON_BITS-1{1'b1}}})) begin

                updated_feature_map[ch] = {1'b0, {DEFAULT_NEURON_BITS-1{1'b1}}};  // Max positive value

            end else if (temp_result < $signed({1'b1, {DEFAULT_NEURON_BITS-1{1'b0}}})) begin

                updated_feature_map[ch] = {1'b1, {DEFAULT_NEURON_BITS-1{1'b0}}};  // Max negative value

            end else begin

                updated_feature_map[ch] = temp_result[DEFAULT_NEURON_BITS-1:0];  // Normal case

            end
        end

        return updated_feature_map;

    endfunction

    function automatic feature_map_t combine_feature_maps(
        input feature_map_t fm_a,
        input feature_map_t fm_b
    );
        feature_map_t result;

        for (int ch = 0; ch < DEFAULT_CHANNELS; ch++) begin

            automatic logic signed [DEFAULT_NEURON_BITS:0] result_before_clamp;
            result_before_clamp = fm_a[ch] + fm_b[ch];

            if (result_before_clamp > $signed({1'b0, {DEFAULT_NEURON_BITS-1{1'b1}}})) begin

                result[ch] = {1'b0, {DEFAULT_NEURON_BITS-1{1'b1}}};  // Max positive value

            end else if (result_before_clamp < $signed({1'b1, {DEFAULT_NEURON_BITS-1{1'b0}}})) begin

                result[ch] = {1'b1, {DEFAULT_NEURON_BITS-1{1'b0}}};  // Max negative value

            end else begin

                result[ch] = result_before_clamp[DEFAULT_NEURON_BITS-1:0];  // Normal case

            end

        end 
        return result;

    endfunction

    function automatic feature_map_t reset_to_zero_if_above_threshold(
        input feature_map_t feature_map,
        input feature_map_t threshold_vector
    );

        feature_map_t result;

        for (int ch = 0; ch < DEFAULT_CHANNELS; ch++) begin

            if (feature_map[ch] >= threshold_vector[ch]) begin
                result[ch] = '0;
            end else begin
                result[ch] = feature_map[ch];
            end

        end

        return result;

    endfunction;

    function automatic output_vector_t create_output_spike_vector(
        input feature_map_t pooled_maps,
        input feature_map_t threshold_vector,
        input logic [DEFAULT_COORD_BITS-2:0] x,
        input logic [DEFAULT_COORD_BITS-2:0] y,
        input logic is_last_event
    );

        output_vector_t result;
        result.timestep = is_last_event;
        result.x = x;
        result.y = y;
        result.spikes = '0; // Initialize spikes to zero

        for (int ch = 0; ch < DEFAULT_CHANNELS; ch++) begin

            if (pooled_maps[ch] >= threshold_vector[ch]) begin

                result.spikes[ch] = 1'b1;

            end else begin

                result.spikes[ch] = 1'b0;

            end

        end

        return result;

    endfunction

endpackage