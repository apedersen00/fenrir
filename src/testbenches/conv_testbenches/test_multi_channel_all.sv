`timescale 1ns / 1ps

module snn_testbench;

    // ============================================================
    // CONSTANTS - will be replaced by generated configuration
    // ============================================================
    
    // Core SNN Parameters
    localparam int KERNEL_SIZE = 3;
    localparam int IN_CHANNELS = 2;
    localparam int OUT_CHANNELS = 4;
    localparam int IMG_HEIGHT = 8;
    localparam int IMG_WIDTH = 8;
    
    // Bit widths
    localparam int BITS_PER_KERNEL_WEIGHT = 6;
    localparam int BITS_PER_NEURON = 9;
    localparam int BITS_PER_COORDINATE = 7;
    
    // FIFO and memory parameters
    localparam int INPUT_FIFO_EVENT_CAPACITY = 4096;
    
    // File paths (will be set by configuration)
    localparam string KERNEL_WEIGHTS_INIT_FILE = "";
    localparam string THRESHOLD_VECTOR_FILE = "";
    localparam string DECAY_VECTOR_FILE = "";
    localparam string EVENT_FILE = "snn_test_events.hex";
    
    // ============================================================
    // DERIVED CONSTANTS
    // ============================================================
    
    // Event structure sizing
    localparam int EVENT_WIDTH = 1 + 2 * BITS_PER_COORDINATE + IN_CHANNELS;
    localparam int INPUT_FIFO_DATA_WIDTH = EVENT_WIDTH;
    localparam int OUTPUT_FIFO_DATA_WIDTH = (BITS_PER_COORDINATE - 1) * 2 + OUT_CHANNELS + 1;
    
    // Memory sizing
    localparam int MAX_COORD_VAL = (1 << BITS_PER_COORDINATE) - 1;
    localparam int MAX_EVENTS = 1024;  // Maximum events to load from file
    
    // Timing parameters
    localparam time CLK_PERIOD = 10ns;  // 100MHz clock
    localparam time RESET_TIME = 100ns;
    
    // ============================================================
    // TYPE DEFINITIONS
    // ============================================================
    
    // Test state enumeration for visual inspection
    typedef enum logic [3:0] {
        TEST_IDLE,
        TEST_RESET,
        TEST_INIT,
        TEST_LOAD_EVENTS,
        TEST_FEED_EVENTS,
        TEST_WAIT_PROCESSING,
        TEST_COLLECT_OUTPUTS,
        TEST_VERIFY_RESULTS,
        TEST_CLEANUP,
        TEST_DONE,
        TEST_ERROR
    } test_state_t;
    
    // Event structure matching the SystemVerilog module
    typedef struct packed {
        logic timestep;
        logic [BITS_PER_COORDINATE-1:0] x;
        logic [BITS_PER_COORDINATE-1:0] y;
        logic [IN_CHANNELS-1:0] spikes;
    } event_t;
    
    // Output event structure  
    typedef struct packed {
        logic timestep;
        logic [BITS_PER_COORDINATE-2:0] x;  // Pooled coordinates are smaller
        logic [BITS_PER_COORDINATE-2:0] y;
        logic [OUT_CHANNELS-1:0] spikes;
    } output_event_t;
    
    // ============================================================
    // SIGNAL DECLARATIONS
    // ============================================================
    
    // Clock and Reset
    logic clk;
    logic rst_n;
    logic enable;
    
    // Test state tracking
    test_state_t current_test_state;
    test_state_t next_test_state;
    string test_state_name;
    
    // Input FIFO interface signals
    logic input_write_enable;
    logic input_fifo_full_next;
    logic [INPUT_FIFO_DATA_WIDTH-1:0] input_fifo_data;
    
    // Output FIFO interface signals  
    logic [OUTPUT_FIFO_DATA_WIDTH-1:0] output_fifo_data;
    logic output_fifo_write_enable;
    logic output_fifo_full_next;
    
    // ============================================================
    // TEST STATE MANAGEMENT
    // ============================================================
    
    // Convert test state enum to readable string
    always_comb begin
        case (current_test_state)
            TEST_IDLE:            test_state_name = "IDLE";
            TEST_RESET:           test_state_name = "RESET";
            TEST_INIT:            test_state_name = "INIT";
            TEST_LOAD_EVENTS:     test_state_name = "LOAD_EVENTS";
            TEST_FEED_EVENTS:     test_state_name = "FEED_EVENTS";
            TEST_WAIT_PROCESSING: test_state_name = "WAIT_PROCESSING";
            TEST_COLLECT_OUTPUTS: test_state_name = "COLLECT_OUTPUTS";
            TEST_VERIFY_RESULTS:  test_state_name = "VERIFY_RESULTS";
            TEST_CLEANUP:         test_state_name = "CLEANUP";
            TEST_DONE:            test_state_name = "DONE";
            TEST_ERROR:           test_state_name = "ERROR";
            default:              test_state_name = "UNKNOWN";
        endcase
    end
    
    // State transition logic (placeholder for now)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_test_state <= TEST_RESET;
        end else begin
            current_test_state <= next_test_state;
        end
    end
    
    // Display state changes for visual inspection
    always @(current_test_state) begin
        $display("[%0t] TEST STATE: %s", $time, test_state_name);
    end
    
    // ============================================================
    // DUT INSTANTIATION
    // ============================================================
    
    CONV2D #(
        .KERNEL_SIZE(KERNEL_SIZE),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .IMG_HEIGHT(IMG_HEIGHT),
        .IMG_WIDTH(IMG_WIDTH),
        .BITS_PER_KERNEL_WEIGHT(BITS_PER_KERNEL_WEIGHT),
        .BITS_PER_NEURON(BITS_PER_NEURON),
        .INPUT_FIFO_EVENT_CAPACITY(INPUT_FIFO_EVENT_CAPACITY),
        .BITS_PER_COORDINATE(BITS_PER_COORDINATE),
        .KERNEL_WEIGHTS_INIT_FILE(KERNEL_WEIGHTS_INIT_FILE),
        .THRESHOLD_VECTOR_FILE(THRESHOLD_VECTOR_FILE),
        .DECAY_VECTOR_FILE(DECAY_VECTOR_FILE)
    ) dut (
        // Control signals
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        
        // Input FIFO interface
        .input_write_enable(input_write_enable),
        .input_fifo_full_next(input_fifo_full_next),
        .input_fifo_data(input_fifo_data),
        
        // Output FIFO interface
        .output_fifo_data(output_fifo_data),
        .output_fifo_write_enable(output_fifo_write_enable),
        .output_fifo_full_next(output_fifo_full_next)
    );
    
    // ============================================================
    // DISPLAY CONFIGURATION
    // ============================================================
    
    initial begin
        $display("=== SNN Testbench Configuration ===");
        $display("Core Parameters:");
        $display("  KERNEL_SIZE: %0d", KERNEL_SIZE);
        $display("  IN_CHANNELS: %0d", IN_CHANNELS);
        $display("  OUT_CHANNELS: %0d", OUT_CHANNELS);
        $display("  IMG_SIZE: %0dx%0d", IMG_WIDTH, IMG_HEIGHT);
        $display("");
        $display("Bit Widths:");
        $display("  BITS_PER_COORDINATE: %0d", BITS_PER_COORDINATE);
        $display("  BITS_PER_NEURON: %0d", BITS_PER_NEURON);
        $display("  BITS_PER_KERNEL_WEIGHT: %0d", BITS_PER_KERNEL_WEIGHT);
        $display("");
        $display("Derived Values:");
        $display("  EVENT_WIDTH: %0d bits", EVENT_WIDTH);
        $display("  INPUT_FIFO_DATA_WIDTH: %0d bits", INPUT_FIFO_DATA_WIDTH);
        $display("  OUTPUT_FIFO_DATA_WIDTH: %0d bits", OUTPUT_FIFO_DATA_WIDTH);
        $display("  MAX_COORD_VAL: %0d", MAX_COORD_VAL);
        $display("");
        $display("Files:");
        $display("  EVENT_FILE: %s", EVENT_FILE);
        $display("  KERNEL_WEIGHTS_INIT_FILE: %s", KERNEL_WEIGHTS_INIT_FILE);
        $display("===================================");
        $display("");
    end

endmodule