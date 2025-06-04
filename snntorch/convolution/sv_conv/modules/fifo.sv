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
    logic [ADDR_WIDTH-1:0] write_ptr;
    logic [ADDR_WIDTH-1:0] read_ptr;
    logic [ADDR_WIDTH:0] count_reg;

    // Status flags
    assign fifo_port.empty = (count_reg == 0);
    assign fifo_port.full = (count_reg == FIFO_DEPTH);

    // Clocked read data output - BRAM-friendly
    always_ff @(posedge fifo_port.clk) begin
        if (fifo_port.read_en && !fifo_port.empty) begin
            fifo_port.read_data <= memory[read_ptr];
        end
    end

    // Write process - BRAM-friendly (no reset on memory)
    always_ff @(posedge fifo_port.clk) begin
        if (fifo_port.write_en && !fifo_port.full) begin
            memory[write_ptr] <= fifo_port.write_data;
        end
    end
    
    // Write pointer process  
    always_ff @(posedge fifo_port.clk or negedge fifo_port.rst_n) begin
        if (!fifo_port.rst_n) begin
            write_ptr <= '0;
        end else if (fifo_port.write_en && !fifo_port.full) begin
            write_ptr <= write_ptr + 1;
        end
    end

    // Read pointer process
    always_ff @(posedge fifo_port.clk or negedge fifo_port.rst_n) begin
        if (!fifo_port.rst_n) begin
            read_ptr <= '0;
        end else if (fifo_port.read_en && !fifo_port.empty) begin
            read_ptr <= read_ptr + 1;
        end
    end

    // Count management
    always_ff @(posedge fifo_port.clk or negedge fifo_port.rst_n) begin
        if (!fifo_port.rst_n) begin
            count_reg <= '0;
        end else begin
            case ({fifo_port.write_en && !fifo_port.full, 
                   fifo_port.read_en && !fifo_port.empty})
                2'b10: count_reg <= count_reg + 1;  // Write only
                2'b01: count_reg <= count_reg - 1;  // Read only
                2'b11: count_reg <= count_reg;      // Simultaneous read/write
                2'b00: count_reg <= count_reg;      // No operation
            endcase
        end
    end

endmodule