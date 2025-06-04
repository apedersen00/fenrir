package snn_interfaces_pkg;

    parameter int DEFAULT_COORD_BITS = 8;
    parameter int DEFAULT_IMG_WIDTH = 32;
    parameter int DEFAULT_IMG_HEIGHT = 32;
    parameter int DEFAULT_NEURON_BITS = 6;
    parameter int DEFAULT_CHANNELS = 6;

    typedef struct packed {
        logic [DEFAULT_COORD_BITS-1:0] x;
        logic [DEFAULT_COORD_BITS-1:0] y;
    } vec2_t;

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

endpackage