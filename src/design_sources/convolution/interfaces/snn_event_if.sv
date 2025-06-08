interface snn_event_if;

    // Import the package inside the interface
    import snn_interfaces_pkg::*;

    // Event data signals using structured types
    vec2_t event_coord;        // Spike coordinates (x, y) - converted from raw FIFO data
    logic  event_valid;        // Data is valid and ready
    logic  event_ready;        // Convolution ready to accept
    logic  event_ack;          // Acknowledgment from convolution module
    // Capture module modport (sends structured events to convolution)
    modport capture(
        output event_coord,
        output event_valid,
        input  event_ready,
        input  event_ack
    );

    // Convolution module modport (receives structured events from capture)  
    modport convolution(
        input  event_coord,
        input  event_valid,
        output event_ready,
        output event_ack
    );

    // Monitor modport for debugging and testbenches
    modport monitor(
        input  event_coord,
        input  event_valid,
        input  event_ready
    );

endinterface