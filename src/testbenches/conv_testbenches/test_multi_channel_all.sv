`timescale 1ns / 1ps

module snn_testbench;

    // ============================================================
    // CONSTANTS - will be replaced by generated configuration
    // ============================================================
    
    // Core SNN Parameters
    localparam int KERNEL_SIZE = 3;
    localparam int IN_CHANNELS = 4;
    localparam int OUT_CHANNELS = 8;
    localparam int IMG_HEIGHT = 8;
    localparam int IMG_WIDTH = 8;
    
    // Bit widths
    localparam int BITS_PER_KERNEL_WEIGHT = 6;
    localparam int BITS_PER_NEURON = 8;
    localparam int BITS_PER_COORDINATE = 3;
    
    // FIFO and memory parameters
    localparam int INPUT_FIFO_EVENT_CAPACITY = 32;
    
    // File paths (will be set by configuration)
    localparam string KERNEL_WEIGHTS_INIT_FILE = "kernel_weights.mem";
    localparam string THRESHOLD_VECTOR_FILE = "";
    localparam string DECAY_VECTOR_FILE = "";
    localparam string EVENT_FILE = "snn_test_events.mem";
    
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
    
    // Test control signals
    int reset_counter;
    int clock_counter;
    
    // Event memory and control
    logic [EVENT_WIDTH-1:0] event_memory [0:MAX_EVENTS-1];
    int total_events_loaded;
    int current_event_index;
    event_t current_event;
    assign current_event = event_t'(event_memory[current_event_index]);
    // Input FIFO interface signals
    logic input_write_enable;
    logic input_fifo_full_next;
    logic [INPUT_FIFO_DATA_WIDTH-1:0] input_fifo_data;
    
    // Output FIFO interface signals  
    logic [OUTPUT_FIFO_DATA_WIDTH-1:0] output_fifo_data;
    logic output_fifo_write_enable;
    logic output_fifo_full_next;
    

    logic module_active;
    logic module_was_active;
    // ============================================================
    // CLOCK GENERATION
    // ============================================================
    
    // Clock generation - 100MHz (10ns period)
    initial begin
        clk = 1'b1;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ============================================================
    // DUMP FILES
    // ============================================================
    int memory_dump_counter = 0;
    always @(posedge clk) begin
        if(rst_n) begin
        
            module_was_active <= module_active;
            if(!module_active && module_was_active) begin
            
                string dump_name;
                #1;
                @(posedge clk);
                dump_name = $sformatf("feature_map_mem_%0d.mem", memory_dump_counter);
                $writememb(dump_name, dut.bram_feature_map_instance.memory);
                memory_dump_counter = memory_dump_counter + 1;

                $display("[%0t] Memory dump saved to %s", $time, dump_name);
            
            end

        end
    end

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
        .output_fifo_full_next(output_fifo_full_next),

        .active(module_active)
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
        
        // ============================================================
        // MAIN TEST SEQUENCE
        // ============================================================
        
        // Initialize all signals to prevent X states
        current_test_state = TEST_RESET;
        rst_n = 1'b0;
        enable = 1'b0;
        input_write_enable = 1'b0;
        input_fifo_data = '0;
        output_fifo_full_next = 1'b0;  // External output FIFO has space
        reset_counter = 0;
        clock_counter = 0;
        
        $display("[%0t] Starting test sequence...", $time);
        
        // RESET PHASE: Hold reset low for 10 clock cycles
        current_test_state = TEST_RESET;
        $display("[%0t] RESET: Holding rst_n=0, enable=0 for 10 cycles", $time);
        repeat(10) @(posedge clk);
        
        // INIT PHASE: Release reset, keep enable low for 1 cycle
        current_test_state = TEST_INIT;
        rst_n = 1'b1;
        enable = 1'b0;
        $display("[%0t] INIT: Released reset, rst_n=1, enable=0", $time);
        @(posedge clk);
        
        // IDLE PHASE: Enable the DUT
        current_test_state = TEST_IDLE;
        enable = 1'b1;
        $display("[%0t] IDLE: DUT enabled, rst_n=1, enable=1", $time);
        @(posedge clk);
        
        // LOAD EVENTS PHASE: Read events from file
        current_test_state = TEST_LOAD_EVENTS;
        $display("[%0t] LOAD_EVENTS: Reading events from %s", $time, EVENT_FILE);
        load_events_from_file();
        @(posedge clk);
        
        // FEED EVENTS PHASE: Send events to DUT one by one
        current_test_state = TEST_FEED_EVENTS;
        $display("[%0t] FEED_EVENTS: Feeding %0d events to DUT", $time, total_events_loaded);
        feed_events_to_dut();
        
        // Wait a few cycles to observe DUT behavior
        current_test_state = TEST_DONE;
        $display("[%0t] TEST_DONE: Observing DUT for a few cycles...", $time);
        repeat(10) @(posedge clk);
        
        $display("[%0t] Test completed successfully", $time);

        current_test_state = TEST_WAIT_PROCESSING;
        $display("[%0t] TEST_WAIT_PROCESSING: Waiting for DUT to finish processing", $time);
        // Wait for DUT to finish processing
        repeat(2000) @(posedge clk);
        $finish;
    end
    
    // ============================================================
    // EVENT LOADING AND FEEDING TASKS
    // ============================================================
    
    // Task to load events from hex file
    task load_events_from_file();
        int display_count;
        begin
            // Initialize event memory to known values
            for (int i = 0; i < MAX_EVENTS; i++) begin
                event_memory[i] = {EVENT_WIDTH{1'bx}};
            end
            
            // Read the binary file
            $readmemb(EVENT_FILE, event_memory);
            
            // Count how many events were actually loaded
            total_events_loaded = 0;
            for (int i = 0; i < MAX_EVENTS; i++) begin
                if (event_memory[i] !== {EVENT_WIDTH{1'bx}}) begin
                    total_events_loaded = i + 1;
                end else begin
                    break; // Stop at first uninitialized entry
                end
            end
            
            $display("[%0t] Loaded %0d events from file", $time, total_events_loaded);
            
            // Display first few events for verification
            display_count = (total_events_loaded < 5) ? total_events_loaded : 5;
            for (int i = 0; i < display_count; i++) begin
                event_t decoded_event;
                decoded_event = decode_event(event_memory[i]);
                $display("[%0t]   Event %0d: ts=%b, x=%0d, y=%0d, spikes=0x%0h", 
                        $time, i, decoded_event.timestep, decoded_event.x, 
                        decoded_event.y, decoded_event.spikes);
            end
            
            if (total_events_loaded > 5) begin
                $display("[%0t]   ... and %0d more events", $time, total_events_loaded - 5);
            end
        end
    endtask
    
    // Task to feed events to DUT
    task feed_events_to_dut();
        event_t current_event;
        begin
            current_event_index = 0;
            
            while (current_event_index < total_events_loaded) begin
                // Wait for DUT to be ready (FIFO not full)
                while (input_fifo_full_next) begin
                    $display("[%0t] Waiting for input FIFO space...", $time);
                    @(posedge clk);
                end
                
                // Send the current event
                input_fifo_data = event_memory[current_event_index];
                input_write_enable = 1'b1;
                
                // Decode and display the event being sent
                current_event = decode_event(event_memory[current_event_index]);
                $display("[%0t] Feeding event %0d: ts=%b, x=%0d, y=%0d, spikes=0x%0h", 
                        $time, current_event_index, current_event.timestep, 
                        current_event.x, current_event.y, current_event.spikes);
                
                @(posedge clk);
                
                // Stop writing after one clock cycle
                input_write_enable = 1'b0;
                current_event_index = current_event_index + 1;
                
                // Optional: Add delay between events
                // repeat(2) @(posedge clk);
            end
            
            $display("[%0t] Finished feeding all %0d events", $time, total_events_loaded);
        end
    endtask
    
    // Function to decode packed event data back to struct
    function event_t decode_event(logic [EVENT_WIDTH-1:0] packed_event);
        event_t result;
        
        result.timestep = packed_event[EVENT_WIDTH-1];
        result.x = packed_event[BITS_PER_COORDINATE + IN_CHANNELS +: BITS_PER_COORDINATE];
        result.y = packed_event[IN_CHANNELS +: BITS_PER_COORDINATE];
        result.spikes = packed_event[IN_CHANNELS-1:0];
        
        return result;
    endfunction

endmodule