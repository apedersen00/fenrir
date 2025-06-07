package snn_interfaces_pkg;

    parameter int DEFAULT_COORD_BITS = 8;
    parameter int DEFAULT_IMG_WIDTH = 32;
    parameter int DEFAULT_IMG_HEIGHT = 32;
    parameter int DEFAULT_NEURON_BITS = 6;
    parameter int DEFAULT_CHANNELS = 6;
    parameter int DEFAULT_INPUT_FIFO_EVENT_CAPACITY = 1024;
    parameter int DEFAULT_KERNEL_BITS = 6;

    typedef struct packed {
        logic [DEFAULT_COORD_BITS-1:0] x;
        logic [DEFAULT_COORD_BITS-1:0] y;
    } vec2_t;
    
    typedef logic signed [DEFAULT_NEURON_BITS-1:0] feature_map_t [0:DEFAULT_CHANNELS-1];

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

    //only for 3x3 kernels
    parameter logic signed [DEFAULT_KERNEL_BITS-1:0] kernel_weights [0:8][0:DEFAULT_CHANNELS -1] =
    '{
        '{-1, -1,  0,  0,  1,  1}, 
        '{-2, -1,  0,  0,  1,  2},
        '{-1, -1,  0,  0,  1,  1},
        '{ 0,  0, -2, -1,  1,  2},
        '{ 0,  0,  0,  0,  0,  0},
        '{ 0,  0,  2,  1, -1, -2},
        '{ 1,  1,  0,  0, -1, -1},
        '{ 2,  1,  0,  0, -1, -2},
        '{ 1,  1,  0,  0, -1, -1}
    };

    function automatic feature_map_t apply_kernel_weights(
        input feature_map_t current_feature_map,
        input logic [3:0] kernel_position
    );
        automatic feature_map_t updated_feature_map;
        
        for (int ch = 0; ch < DEFAULT_CHANNELS; ch++) begin
            automatic logic signed [DEFAULT_NEURON_BITS:0] temp_result;
            
            // Add kernel weight to current membrane potential (both are signed now)
            temp_result = current_feature_map[ch] + kernel_weights[kernel_position][ch];
            
            // Handle overflow with saturation
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

endpackage