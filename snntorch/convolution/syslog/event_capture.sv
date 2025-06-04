import conv_pool_pkg::*;

module event_capture 
#(
    parameter int COORD_BITS = 8,
    parameter int IMG_WIDTH = 32,
    parameter int IMG_HEIGHT = 32
)
(
    input   logic clk,
    input   logic reset_ni,
    input   logic enable_i,

    input   logic fifo_empty_i,
    input   logic [2 * COORD_BITS - 1 : 0] fifo_data_i,
    output  logic fifo_read_o,

    output  coord_t captured_event_coord_o,
    output  logic captured_event_valid_o,
    input   logic captured_event_processed_i

);

    capture_state_t state, next_state;
    coord_t captured_event_coord;
    logic valid_reg;
    logic fifo_read_pulse;

    logic captured_event_processed_r;

    assign captured_event_coord_o = captured_event_coord;
    assign captured_event_valid_o = valid_reg;
    assign fifo_read_o = fifo_read_pulse;

    always_comb begin : next_state_logic
        next_state = state;
        if (!enable_i) begin
            next_state = IDLE;
        end else begin
            unique case (state)
                IDLE : begin
                    if (!fifo_empty_i) begin
                        next_state = READ_REQUEST;
                    end
                end

                READ_REQUEST : begin
                    next_state = VALIDATE;
                end

                VALIDATE : begin
                    automatic coord_t temp_coord = unpack_coordinates(fifo_data_i);
                    if (is_valid_coord(temp_coord, IMG_WIDTH, IMG_HEIGHT)) begin
                        next_state = DATA_READY;
                    end else begin
                        next_state = IDLE;
                    end
                end

                DATA_READY : begin
                    if (captured_event_processed_r) begin
                        next_state = IDLE;
                    end
                end

                RESET : begin
                    next_state = IDLE;
                end

            endcase
        end
    end

    always_ff @(posedge clk or negedge reset_ni) begin : register_processed_signal
        if (!reset_ni) begin
            captured_event_processed_r <= 1'b0;
        end else begin
            captured_event_processed_r <= captured_event_processed_i;
        end
    end

    always_ff @(posedge clk or negedge reset_ni) begin : fifo_read_control
        if (!reset_ni) begin
            fifo_read_pulse <= 1'b0;
        end else begin
            fifo_read_pulse <= (state == IDLE) && (next_state == READ_REQUEST) && enable_i;
        end
    end

    always_ff @(posedge clk or negedge reset_ni) begin : state_register
    
        if (!reset_ni) begin
            state <= RESET;
        end else begin
            state <= next_state;
        end 

    end 

    always_ff @(posedge clk or negedge reset_ni) begin : data_capture
        if (!reset_ni) begin
            captured_event_coord <= '{x: '0, y: '0};
            valid_reg <= 1'b0;
        end else begin
            case (state)
                VALIDATE : begin
                    
                    if (next_state == DATA_READY) begin
                        captured_event_coord <= unpack_coordinates(fifo_data_i);
                        valid_reg <= 1'b1; 
                    end
                end
                DATA_READY : begin
                    if (captured_event_processed_r) begin
                        valid_reg <= 1'b0; 
                    end else begin
                        valid_reg <= 1'b1; 
                    end
                end
                default : begin
                    valid_reg <= 1'b0;
                end
            endcase
        end
    end
endmodule