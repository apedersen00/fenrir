// =============================================================================
// File: modules/arbiter.sv
// Description: Transparent memory arbiter with Port A=READ, Port B=WRITE assignment
// =============================================================================

import snn_interfaces_pkg::*;

module arbiter #(
    parameter int COORD_BITS = DEFAULT_COORD_BITS,
    parameter int CHANNELS = DEFAULT_CHANNELS,
    parameter int BITS_PER_CHANNEL = DEFAULT_NEURON_BITS,
    parameter int IMG_WIDTH = DEFAULT_IMG_WIDTH,
    parameter int IMG_HEIGHT = DEFAULT_IMG_HEIGHT,
    // BRAM parameters
    parameter int BRAM_DATA_WIDTH = CHANNELS * BITS_PER_CHANNEL,  // Total width for all feature maps
    parameter int BRAM_ADDR_WIDTH = $clog2(IMG_WIDTH * IMG_HEIGHT)  // Address space for coordinate grid
)(
    input logic clk,
    input logic rst_n,
    
    // Control interface
    snn_control_if.arbiter ctrl_port,
    
    // Conv module interfaces
    arbiter_if.arbiter conv_read_port,
    arbiter_if.arbiter conv_write_port,
    
    // Pool module interfaces  
    arbiter_if.arbiter pool_read_port,
    arbiter_if.arbiter pool_write_port,
    
    // BRAM interface
    dp_bram_if.arbiter bram_port
);

    // Use the interface typedef for consistency
    typedef conv_read_port.fm_array_t fm_array_t;

    // Drive clock and reset to BRAM
    assign bram_port.clk = clk;
    assign bram_port.rst_n = rst_n;

    // Coordinate to address conversion function
    function automatic logic [BRAM_ADDR_WIDTH-1:0] coord_to_addr(input vec2_t coord);
        return coord.y + coord.x * IMG_WIDTH;  // Row-major addressing
    endfunction

    // Inline packing/unpacking to avoid type conflicts
    // Pack feature map data to BRAM data width
    logic [BRAM_DATA_WIDTH-1:0] packed_write_data;
    always_comb begin
        if (ctrl_port.conv_or_pool) begin
            // Pack conv write data
            for (int i = 0; i < CHANNELS; i++) begin
                packed_write_data[i*BITS_PER_CHANNEL +: BITS_PER_CHANNEL] = conv_write_port.data_in[i];
            end
        end else begin
            // Pack pool write data
            for (int i = 0; i < CHANNELS; i++) begin
                packed_write_data[i*BITS_PER_CHANNEL +: BITS_PER_CHANNEL] = pool_write_port.data_in[i];
            end
        end
    end
    
    // Unpack BRAM data to feature map data
    always_comb begin
        if (ctrl_port.conv_or_pool) begin
            // Unpack to conv read data
            for (int i = 0; i < CHANNELS; i++) begin
                conv_read_port.data_out[i] = bram_port.data_out_a[i*BITS_PER_CHANNEL +: BITS_PER_CHANNEL];
            end
            // Clear pool data
            for (int i = 0; i < CHANNELS; i++) begin
                pool_read_port.data_out[i] = '0;
            end
        end else begin
            // Unpack to pool read data
            for (int i = 0; i < CHANNELS; i++) begin
                pool_read_port.data_out[i] = bram_port.data_out_a[i*BITS_PER_CHANNEL +: BITS_PER_CHANNEL];
            end
            // Clear conv data
            for (int i = 0; i < CHANNELS; i++) begin
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
        if (ctrl_port.conv_or_pool) begin
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
    assign bram_port.en_a = read_req_active && ctrl_port.enable;
    assign bram_port.we_a = 1'b0;  // Port A is always READ-ONLY
    assign bram_port.data_in_a = '0;  // Not used for reads

    // =========================================================================  
    // PORT B: WRITE PORT (used by active module based on conv_or_pool)
    // =========================================================================
    
    // Mux write requests based on phase
    logic write_req_active;
    vec2_t write_coord;
    fm_array_t write_data;
    
    always_comb begin
        if (ctrl_port.conv_or_pool) begin
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
    assign bram_port.en_b = write_req_active && ctrl_port.enable;
    assign bram_port.we_b = write_req_active && ctrl_port.enable;  // Port B is always WRITE when enabled
    assign bram_port.data_in_b = packed_write_data;  // Use the packed data signal
    // Note: data_out_b is not used since Port B is write-only

    // =========================================================================
    // Ready Signal Generation
    // =========================================================================
    
    // Ready signals based on phase and enable
    assign conv_read_port.read_ready = ctrl_port.enable && ctrl_port.conv_or_pool;
    assign conv_write_port.write_ready = ctrl_port.enable && ctrl_port.conv_or_pool;
    assign pool_read_port.read_ready = ctrl_port.enable && !ctrl_port.conv_or_pool;
    assign pool_write_port.write_ready = ctrl_port.enable && !ctrl_port.conv_or_pool;

    // =========================================================================
    // Status and Control
    // =========================================================================
    
    // Activity detection
    logic conv_active, pool_active;
    assign conv_active = (conv_read_port.read_req || conv_write_port.write_req) && 
                        ctrl_port.enable && ctrl_port.conv_or_pool;
    assign pool_active = (pool_read_port.read_req || pool_write_port.write_req) && 
                        ctrl_port.enable && !ctrl_port.conv_or_pool;
    
    assign ctrl_port.active = conv_active || pool_active;
    assign ctrl_port.ready = ctrl_port.enable;

    // =========================================================================
    // Debug Support
    // =========================================================================
    
    `ifndef SYNTHESIS
    // Monitor for debugging
    always @(posedge clk) begin
        if (ctrl_port.enable) begin
            if (read_req_active) begin
                $display("T=%0t: ARBITER %s READ coord=(%0d,%0d) addr=0x%h → PortA", 
                        $time, ctrl_port.conv_or_pool ? "CONV" : "POOL",
                        read_coord.x, read_coord.y, coord_to_addr(read_coord));
            end
            
            if (write_req_active) begin
                $display("T=%0t: ARBITER %s WRITE coord=(%0d,%0d) addr=0x%h → PortB", 
                        $time, ctrl_port.conv_or_pool ? "CONV" : "POOL",
                        write_coord.x, write_coord.y, coord_to_addr(write_coord));
            end
        end
    end
    
    // Pipeline efficiency monitoring
    always @(posedge clk) begin
        if (ctrl_port.enable && (read_req_active && write_req_active)) begin
            $display("T=%0t: ARBITER Simultaneous Read/Write - Pipeline active!", $time);
        end
    end
    `endif

endmodule