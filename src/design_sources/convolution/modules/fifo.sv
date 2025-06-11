module fifo #(
    parameter int DATA_WIDTH = 8,
    parameter int ADDR_WIDTH = 4
)(
    fifo_if.fifo_module fifo_port
);

    // Calculate FIFO depth
    localparam int FIFO_DEPTH = 2**ADDR_WIDTH;

    // Internal memory and pointers
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] memory [0:FIFO_DEPTH-1];
    logic [ADDR_WIDTH:0] write_ptr;  // Extra bit to distinguish full from empty
    logic [ADDR_WIDTH:0] read_ptr;   // Extra bit to distinguish full from empty
    
    // Gray code pointers for CDC safety (if needed)
    logic [ADDR_WIDTH:0] write_ptr_gray;
    logic [ADDR_WIDTH:0] read_ptr_gray;

    // Status flag generation
    wire ptr_equal = (write_ptr[ADDR_WIDTH-1:0] == read_ptr[ADDR_WIDTH-1:0]);
    wire ptr_msb_diff = (write_ptr[ADDR_WIDTH] != read_ptr[ADDR_WIDTH]);
    
    assign fifo_port.empty = (write_ptr == read_ptr);
    assign fifo_port.full = ptr_equal && ptr_msb_diff;
    
    // Almost full/empty flags for better flow control
    wire [ADDR_WIDTH:0] next_write_ptr = write_ptr + 1;
    wire [ADDR_WIDTH:0] next_read_ptr = read_ptr + 1;
    
    assign fifo_port.full_next = (next_write_ptr[ADDR_WIDTH-1:0] == read_ptr[ADDR_WIDTH-1:0]) && 
                                 (next_write_ptr[ADDR_WIDTH] != read_ptr[ADDR_WIDTH]);
    assign fifo_port.almost_empty = (write_ptr == next_read_ptr);

    // Memory read - registered output for BRAM compatibility
    always_ff @(posedge fifo_port.clk) begin
        if (fifo_port.read_en && !fifo_port.empty) begin
            fifo_port.read_data <= memory[read_ptr[ADDR_WIDTH-1:0]];
        end
    end

    // Memory write - BRAM-friendly (no reset on memory contents)
    always_ff @(posedge fifo_port.clk) begin
        if (fifo_port.write_en && !fifo_port.full) begin
            memory[write_ptr[ADDR_WIDTH-1:0]] <= fifo_port.write_data;
        end
    end
    
    // Write pointer management
    always_ff @(posedge fifo_port.clk or negedge fifo_port.rst_n) begin
        if (!fifo_port.rst_n) begin
            write_ptr <= '0;
        end else if (fifo_port.write_en && !fifo_port.full) begin
            write_ptr <= write_ptr + 1;
        end
    end

    // Read pointer management
    always_ff @(posedge fifo_port.clk or negedge fifo_port.rst_n) begin
        if (!fifo_port.rst_n) begin
            read_ptr <= '0;
        end else if (fifo_port.read_en && !fifo_port.empty) begin
            read_ptr <= read_ptr + 1;
        end
    end

    // Synthesis attributes for better implementation
    (* ASYNC_REG = "TRUE" *) logic sync_reg1, sync_reg2;
    

endmodule