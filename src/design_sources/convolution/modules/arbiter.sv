import conv_pkg::*;

module arbiter #(
    parameter int BITS_PER_COORDINATE,
    parameter int OUT_CHANNELS,
    parameter int BITS_PER_NEURON,
    parameter int IMG_WIDTH,
    parameter int IMG_HEIGHT,
    // BRAM parameters
    parameter int BRAM_DATA_WIDTH = OUT_CHANNELS * BITS_PER_NEURON,  // Total width for all feature maps
    parameter int BRAM_ADDR_WIDTH = $clog2(IMG_WIDTH * IMG_HEIGHT)  // Address space for coordinate grid
)(
    input logic clk,
    input logic rst_n,
    input arbiter_mode_t mode,
    // Conv module interfaces
    arbiter_if.arbiter conv_read_port,
    arbiter_if.arbiter conv_write_port,
    
    // Pool module interfaces  
    arbiter_if.arbiter pool_read_port,
    arbiter_if.arbiter pool_write_port,
    
    // BRAM interface
    dp_bram_if.arbiter bram_port
);

    typedef logic signed [BITS_PER_NEURON-1:0] feature_map_t [0:OUT_CHANNELS-1];
    typedef struct packed {
        logic [BITS_PER_COORDINATE-1:0] x;
        logic [BITS_PER_COORDINATE-1:0] y;
    } vec2_t;

    // Drive clock and reset to BRAM
    assign bram_port.clk = clk;
    assign bram_port.rst_n = rst_n;

    // Coordinate to address conversion function
    function automatic logic [BRAM_ADDR_WIDTH-1:0] coord_to_addr(input vec2_t coord); //TODO might be wrong about which to multiply
        return coord.y + coord.x * IMG_WIDTH;  // Row-major addressing
    endfunction

    // Inline packing/unpacking to avoid type conflicts
    // Pack feature map data to BRAM data width
    logic [BRAM_DATA_WIDTH-1:0] packed_write_data;
    always_comb begin
        if (mode == MUX_CONVOLUTION) begin
            // Pack conv write data
            for (int i = 0; i < OUT_CHANNELS; i++) begin
                packed_write_data[i*BITS_PER_NEURON +: BITS_PER_NEURON] = conv_write_port.data_in[i];
            end
        end else begin
            // Pack pool write data
            for (int i = 0; i < OUT_CHANNELS; i++) begin
                packed_write_data[i*BITS_PER_NEURON +: BITS_PER_NEURON] = pool_write_port.data_in[i];
            end
        end
    end
    
    // Unpack BRAM data to feature map data
    always_comb begin
        if (mode == MUX_CONVOLUTION) begin
            // Unpack to conv read data
            for (int i = 0; i < OUT_CHANNELS; i++) begin
                conv_read_port.data_out[i] = bram_port.data_out_a[i*BITS_PER_NEURON +: BITS_PER_NEURON];
            end
            // Clear pool data
            for (int i = 0; i < OUT_CHANNELS; i++) begin
                pool_read_port.data_out[i] = '0;
            end
        end else begin
            // Unpack to pool read data
            for (int i = 0; i < OUT_CHANNELS; i++) begin
                pool_read_port.data_out[i] = bram_port.data_out_a[i*BITS_PER_NEURON +: BITS_PER_NEURON];
            end
            // Clear conv data
            for (int i = 0; i < OUT_CHANNELS; i++) begin
                conv_read_port.data_out[i] = '0;
            end
        end
    end

    // =========================================================================
    // PORT A: READ PORT (used by active module based on conv_or_pool)
    // =========================================================================
    
    // Mux read requests based on phase
    logic read_req_active;
    vec2_t read_coord;
    
    always_comb begin
        if (mode == MUX_CONVOLUTION) begin
            // Conv phase - conv module reads
            read_req_active = conv_read_port.read_req;
            read_coord = conv_read_port.coord_get;
        end else begin
            // Pool phase - pool module reads  
            read_req_active = pool_read_port.read_req;
            read_coord = pool_read_port.coord_get;
        end
    end
    
    // Port A control (READ)
    assign bram_port.addr_a = coord_to_addr(read_coord);
    assign bram_port.en_a = read_req_active;
    assign bram_port.we_a = 1'b0;  // Port A is always READ-ONLY
    assign bram_port.data_in_a = '0;  // Not used for reads

    // =========================================================================  
    // PORT B: WRITE PORT (used by active module based on conv_or_pool)
    // =========================================================================
    
    // Mux write requests based on phase
    logic write_req_active;
    vec2_t write_coord;
    feature_map_t write_data;
    
    always_comb begin
        if (mode == MUX_CONVOLUTION) begin
            // Conv phase - conv module writes
            write_req_active = conv_write_port.write_req;
            write_coord = conv_write_port.coord_wtr;
            write_data = conv_write_port.data_in;
        end else begin
            // Pool phase - pool module writes
            write_req_active = pool_write_port.write_req;
            write_coord = pool_write_port.coord_wtr;
            write_data = pool_write_port.data_in;
        end
    end
    
    // Port B control (WRITE)
    assign bram_port.addr_b = coord_to_addr(write_coord);
    assign bram_port.en_b = write_req_active;
    assign bram_port.we_b = write_req_active;  // Port B is always WRITE when enabled
    assign bram_port.data_in_b = packed_write_data;  // Use the packed data signal

endmodule