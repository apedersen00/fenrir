import conv_pkg::*;

module capture_event #(

    parameter int DATA_WIDTH,
    parameter int IMG_HEIGHT,
    parameter int IMG_WIDTH,
    parameter int BITS_PER_COORDINATE,
    parameter int IN_CHANNELS

)(
    output logic timestep,
    input logic enable,
    fifo_if.consumer fifo_port,
    event_if.capture event_port
);

typedef enum logic [1:0] {
    IDLE,
    VALIDATE,
    OUTPUTTING
} state_t;

state_t current_state, next_state;

typedef struct packed {
    logic timestep;
    logic [BITS_PER_COORDINATE-1:0] x;
    logic [BITS_PER_COORDINATE-1:0] y;
    logic [IN_CHANNELS-1:0] spikes;
} event_t;

event_t event_data;
assign event_data = event_t'(fifo_port.read_data);
assign timestep = event_data.timestep;

always_ff @(posedge event_port.clk or negedge event_port.rst_n) begin
    if (!event_port.rst_n) begin
        current_state <= IDLE;
    end else begin
        current_state <= next_state;
    end
end

always_comb begin

    case (current_state)

        IDLE: begin
            if (!fifo_port.empty && event_port.conv_ready && enable) begin // remember to check if conv is ready later
                next_state = VALIDATE;
                fifo_port.read_en = 1;
            end else begin
                next_state = IDLE;
                fifo_port.read_en = 0;
            end
        end

        VALIDATE: begin
            fifo_port.read_en = 0; // Stop reading from FIFO
            
            if (event_data.timestep) begin
                next_state = IDLE;
            end else if(event_data.spikes == 0) begin
                next_state = IDLE;
            end else begin
                if (event_data.x < IMG_WIDTH && event_data.y < IMG_HEIGHT) begin
                    event_port.event_data = event_data; // Pass the event data
                    event_port.event_valid = 1; // Indicate valid event
                    next_state = OUTPUTTING;
                end else begin
                    next_state = IDLE; // Invalid coordinates, go back to IDLE
                end
            end
        end

        OUTPUTTING: begin
            if (event_port.conv_ack) begin
                event_port.event_valid = 0; // Acknowledge the event
                next_state = IDLE; // Go back to IDLE after acknowledgment
            end else begin
                next_state = OUTPUTTING; // Wait for acknowledgment
            end
        end

    endcase
end
endmodule