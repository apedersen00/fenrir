module fifo #(
    parameter int DATA_WIDTH = 8,
    parameter int ADDR_WIDTH = 4
)(
    fifo_if.fifo_module fifo_port
);

    // Calculate FIFO depth
    localparam int FIFO_DEPTH = 2**ADDR_WIDTH;

    // BRAM memory with Vivado-specific attributes
    (* ram_style = "block" *) 
    (* ram_decomp = "power" *)
    logic [DATA_WIDTH-1:0] memory [0:FIFO_DEPTH-1];
    
    // Binary pointers with extra bit for full/empty distinction
    logic [ADDR_WIDTH:0] write_ptr;  
    logic [ADDR_WIDTH:0] read_ptr;   

    // Extract addresses - critical for Vivado BRAM inference
    wire [ADDR_WIDTH-1:0] write_addr = write_ptr[ADDR_WIDTH-1:0];
    wire [ADDR_WIDTH-1:0] read_addr = read_ptr[ADDR_WIDTH-1:0];
    
    // Control signals
    wire write_enable = fifo_port.write_en && !fifo_port.full;
    wire read_enable = fifo_port.read_en && !fifo_port.empty;

    // Status flag generation
    assign fifo_port.empty = (write_ptr == read_ptr);
    assign fifo_port.full = ((write_ptr[ADDR_WIDTH-1:0] == read_ptr[ADDR_WIDTH-1:0]) && 
                             (write_ptr[ADDR_WIDTH] != read_ptr[ADDR_WIDTH]));
    
    // Almost full/empty flags
    wire [ADDR_WIDTH:0] next_write_ptr = write_ptr + 1;
    assign fifo_port.full_next = ((next_write_ptr[ADDR_WIDTH-1:0] == read_ptr[ADDR_WIDTH-1:0]) && 
                                  (next_write_ptr[ADDR_WIDTH] != read_ptr[ADDR_WIDTH]));
    assign fifo_port.almost_empty = (write_ptr == (read_ptr + 1));

    // BRAM Write Port - Vivado prefers this simple pattern
    always_ff @(posedge fifo_port.clk) begin
        if (write_enable) begin
            memory[write_addr] <= fifo_port.write_data;
        end
    end

    // BRAM Read Port - Always read, output register handles control
    // This is the key pattern Vivado recognizes for BRAM
    always_ff @(posedge fifo_port.clk) begin
        fifo_port.read_data <= memory[read_addr];
    end

    // Write pointer management
    always_ff @(posedge fifo_port.clk) begin
        if (!fifo_port.rst_n) begin
            write_ptr <= '0;
        end else if (write_enable) begin
            write_ptr <= write_ptr + 1;
        end
    end

    // Read pointer management
    always_ff @(posedge fifo_port.clk) begin
        if (!fifo_port.rst_n) begin
            read_ptr <= '0;
        end else if (read_enable) begin
            read_ptr <= read_ptr + 1;
        end
    end

endmodule