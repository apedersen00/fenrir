import snn_interfaces_pkg::*;

module tb_fast_conv_controller;
    
    localparam int COORD_BITS = 8;        // Use default 8-bit coordinates
    localparam int IMG_WIDTH = 32;        // 32x32 image
    localparam int IMG_HEIGHT = 32;
    localparam int CHANNELS = 6;          // 6 feature map channels
    localparam int BITS_PER_CHANNEL = 6;  // 6 bits per channel
    localparam int FIFO_DATA_WIDTH = 2 * COORD_BITS;           // 16 bits
    localparam int INPUT_FIFO_EVENT_CAPACITY = 16;             // Small for testing
    localparam int INPUT_FIFO_ADDR_WIDTH = $clog2(INPUT_FIFO_EVENT_CAPACITY);

    typedef enum logic [3:0] {
        TB_RESET,
        TB_IDLE,
        TB_WRITE_EVENTS,
        TB_TEST_TIMESTEP,
        TB_FILL_FIFO,
        TB_TEST_OVERFLOW,
        TB_DRAIN_FIFO,
        TB_TEST_RESET,
        TB_COMPLETE
    } tb_state_t;

    tb_state_t tb_state;

    logic clk = 1;
    logic rst_n;

    always #5 clk = ~clk;  // 100 MHz clocck

    logic sys_enable;
    logic sys_reset;
    logic timestep;
    logic system_active;
    logic fifo_empty;
    logic fifo_full;
    logic [FIFO_DATA_WIDTH-1:0] spike_event;
    logic write_enable;
    logic output_fifo_full; // will be used later

    fast_conv_controller #(
        .COORD_BITS(COORD_BITS),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL),
        .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH),
        .INPUT_FIFO_EVENT_CAPACITY(INPUT_FIFO_EVENT_CAPACITY),
        .INPUT_FIFO_ADDR_WIDTH(INPUT_FIFO_ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .sys_enable(sys_enable),
        .sys_reset(sys_reset),
        .timestep(timestep),
        .system_active(system_active),
        .fifo_empty(fifo_empty),
        .fifo_full(fifo_full),
        .spike_event(spike_event),
        .write_enable(write_enable),
        .output_fifo_full(output_fifo_full)
    );

    // =========================================================================
    // TEST DATA
    // =========================================================================

    logic [FIFO_DATA_WIDTH-1:0] test_events [0:9]; // 10 test events

    initial begin
        // Event list: (x, y) coordinates packed as {x, y}
        test_events[0] = pack_coordinates('{x: 8'd5,  y: 8'd5});    // Center region
        test_events[1] = pack_coordinates('{x: 8'd10, y: 8'd15});   // Various locations
        test_events[2] = pack_coordinates('{x: 8'd20, y: 8'd8});    
        test_events[3] = pack_coordinates('{x: 8'd3,  y: 8'd25});   
        test_events[4] = pack_coordinates('{x: 8'd15, y: 8'd12});   
        test_events[5] = pack_coordinates('{x: 8'd28, y: 8'd7});    // Near edges
        test_events[6] = pack_coordinates('{x: 8'd1,  y: 8'd30});   
        test_events[7] = pack_coordinates('{x: 8'd31, y: 8'd2});    
        test_events[8] = pack_coordinates('{x: 8'd16, y: 8'd16});   // Exact center
        test_events[9] = pack_coordinates('{x: 8'd0,  y: 8'd31});   // Corner
        
        $display("=== Test Events Initialized ===");
        for (int i = 0; i < 10; i++) begin
            automatic vec2_t coord = unpack_coordinates(test_events[i]);
            $display("Event[%0d]: (x=%0d, y=%0d) = 0x%04h", 
                     i, coord.x, coord.y, test_events[i]);
        end
    end

    // =========================================================================
    // TESTBENCH START
    // =========================================================================

    int event_index;
    int cycle_count;

    initial begin

        $display("=== Starting Testbench ===");
        $display("Configuration: %0dx%0d image, FIFO capacity %0d",
                 IMG_WIDTH, IMG_HEIGHT, INPUT_FIFO_EVENT_CAPACITY);
        tb_state = TB_RESET;
        rst_n = 0;
        sys_enable = 0;
        sys_reset = 1;
        timestep = 0;
        spike_event = 0;
        write_enable = 0;
        output_fifo_full = 0;
        event_index = 0;
        cycle_count = 0;

        // =====================================================================
        // Test 1: Reset and Initialization
        // =====================================================================
        $display("\n=== Test 1: Reset and Initialization ===");
        tb_state = TB_RESET;
        
        #30;  // Hold reset for 30ns
        rst_n = 1;
        sys_reset = 0;
        sys_enable = 1;
        #20;  // Wait for system to stabilize
        
        $display("T=%0t: Reset complete - fifo_empty=%b, fifo_full=%b, system_active=%b", 
                 $time, fifo_empty, fifo_full, system_active);                

        // =====================================================================
        // Test 2: Write Events to FIFO
        // =====================================================================

        $display("\n=== Test 2: Write 10 Events to FIFO ===");
        tb_state = TB_WRITE_EVENTS;

        for (event_index = 0; event_index < 10; event_index++) begin
            automatic vec2_t coord = unpack_coordinates(test_events[event_index]);

            @(posedge clk);
            spike_event = test_events[event_index];
            write_enable = 1;
            $display("T=%0t: Writing Event[%0d]: (x=%0d, y=%0d) = 0x%04h", 
                     $time, event_index, coord.x, coord.y, test_events[event_index]);
            $display("         FIFO status before: empty=%b, full=%b", fifo_empty, fifo_full);
        end

        for (event_index = 0; event_index < 10; event_index++) begin
            automatic vec2_t coord = unpack_coordinates(test_events[event_index]);

            @(posedge clk);
            spike_event = test_events[event_index];
            write_enable = 1;
            $display("T=%0t: Writing Event[%0d]: (x=%0d, y=%0d) = 0x%04h", 
                     $time, event_index, coord.x, coord.y, test_events[event_index]);
            $display("         FIFO status before: empty=%b, full=%b", fifo_empty, fifo_full);
        end

        for (event_index = 0; event_index < 10; event_index++) begin
            automatic vec2_t coord = unpack_coordinates(test_events[event_index]);

            @(posedge clk);
            spike_event = test_events[event_index];
            write_enable = 1;
            $display("T=%0t: Writing Event[%0d]: (x=%0d, y=%0d) = 0x%04h", 
                     $time, event_index, coord.x, coord.y, test_events[event_index]);
            $display("         FIFO status before: empty=%b, full=%b", fifo_empty, fifo_full);
        end

        for (event_index = 0; event_index < 10; event_index++) begin
            automatic vec2_t coord = unpack_coordinates(test_events[event_index]);

            @(posedge clk);
            spike_event = test_events[event_index];
            write_enable = 1;
            $display("T=%0t: Writing Event[%0d]: (x=%0d, y=%0d) = 0x%04h", 
                     $time, event_index, coord.x, coord.y, test_events[event_index]);
            $display("         FIFO status before: empty=%b, full=%b", fifo_empty, fifo_full);
        end

    end

    

endmodule