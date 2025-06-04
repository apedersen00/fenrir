package conv_pool_pkg;

    parameter int IMG_WIDTH = 32;
    parameter int IMG_HEIGHT = 32;
    parameter int COORD_BITS = 8;

    typedef struct packed {
        logic [COORD_BITS-1:0] x;
        logic [COORD_BITS-1:0] y;
    } coord_t;

    typedef enum logic [2:0] {
        IDLE = 3'b000,
        READ_REQUEST = 3'b001,
        VALIDATE = 3'b010,        
        DATA_READY = 3'b011,      
        RESET = 3'b111
    } capture_state_t;

    function automatic coord_t unpack_coordinates(
        input logic [2 * COORD_BITS - 1:0] packed_data
    );
        coord_t result;
        result.x = packed_data[2 * COORD_BITS - 1: COORD_BITS];
        result.y = packed_data[COORD_BITS - 1:0];
        return result;
    endfunction

    function automatic logic [2 * COORD_BITS - 1 : 0] pack_coordinates(
        input coord_t coord
    );
        return {coord.x, coord.y};
    endfunction

    function automatic logic is_valid_coord(
        input coord_t coord,
        input int width = IMG_WIDTH,
        input int height = IMG_HEIGHT
    );
        return (coord.x < width) && (coord.y < height);
    endfunction

    function automatic string coord_to_string(input coord_t coord);
    return $sformatf("(%0d,%0d)", coord.x, coord.y);
    endfunction

endpackage