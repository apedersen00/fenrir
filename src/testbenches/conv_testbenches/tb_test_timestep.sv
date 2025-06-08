// =============================================================================
// File: testbenches/tb_timestep_transition_fixed.sv
// Description: Test timestep-triggered conv→pool state transitions (Vivado compatible)
// =============================================================================

import snn_interfaces_pkg::*;

module tb_timestep_transition;

    // Testbench parameters - minimal for state machine testing
    localparam int COORD_BITS = 8;
    localparam int IMG_WIDTH = 32;
    localparam int IMG_HEIGHT = 32;
    localparam int CHANNELS = 2;
    localparam int BITS_PER_CHANNEL = 9;
    localparam int FIFO_DATA_WIDTH = 2 * COORD_BITS;
    localparam int INPUT_FIFO_EVENT_CAPACITY = 16;
    localparam int INPUT_FIFO_ADDR_WIDTH = $clog2(INPUT_FIFO_EVENT_CAPACITY);
    localparam int BRAM_DATA_WIDTH = CHANNELS * BITS_PER_CHANNEL;
    localparam int BRAM_ADDR_WIDTH = $clog2(IMG_WIDTH * IMG_HEIGHT);

    // Clock and reset
    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;  // 100MHz clock

    // DUT signals
    logic sys_enable;
    logic sys_reset;
    logic timestep;  // ← This is what we're testing!
    logic system_active;
    logic fifo_empty;
    logic fifo_full;
    logic [FIFO_DATA_WIDTH-1:0] spike_event;
    logic write_enable;
    logic output_fifo_full;

    // Variables to track state changes (instead of $past)
    logic prev_pooling_request;
    logic prev_conv_ready;
    logic prev_pool_active;
    logic [1:0] prev_state;

    // DUT instantiation
    fast_conv_controller #(
        .COORD_BITS(COORD_BITS),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL),
        .FIFO_DATA_WIDTH(FIFO_DATA_WIDTH),
        .INPUT_FIFO_EVENT_CAPACITY(INPUT_FIFO_EVENT_CAPACITY),
        .INPUT_FIFO_ADDR_WIDTH(INPUT_FIFO_ADDR_WIDTH),
        .BRAM_DATA_WIDTH(BRAM_DATA_WIDTH),
        .BRAM_ADDR_WIDTH(BRAM_ADDR_WIDTH)
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

    // Function to convert state enum to string (instead of .name())
    function automatic string state_to_string(logic [1:0] state_val);
        case (state_val)
            2'b00: return "CONV_MODE";
            2'b01: return "CONV_FINISHING"; 
            2'b10: return "POOL_MODE";
            2'b11: return "PAUSE_POOLING";
            default: return "UNKNOWN";
        endcase
    endfunction

    // Track previous values each cycle
    always_ff @(posedge clk) begin
        prev_pooling_request <= dut.pooling_request_pending;
        prev_conv_ready <= dut.conv_module_ready;
        prev_pool_active <= dut.pooling_module_active;
        prev_state <= dut.state;
    end

    // Test procedure
    initial begin
        $display("=== Timestep Transition Testbench (Vivado Compatible) ===");
        $display("Testing conv→pool state machine transitions");
        
        // =====================================================================
        // Phase 1: Initialize system
        // =====================================================================
        $display("\n--- Phase 1: System Initialization ---");
        
        // Initialize all signals
        rst_n = 0;
        sys_enable = 0;
        sys_reset = 1;
        timestep = 0;  // Start with timestep LOW
        spike_event = 0;
        write_enable = 0;
        output_fifo_full = 0;
        
        // Reset sequence
        #30;
        rst_n = 1;
        sys_reset = 0;
        sys_enable = 1;
        #20;
        
        $display("T=%0t: System initialized", $time);
        $display("         State: %s", state_to_string(dut.state));
        $display("         Conv module ready: %b", dut.conv_module_ready);
        $display("         Pooling request pending: %b", dut.pooling_request_pending);
        
        // =====================================================================
        // Phase 2: Verify we start in CONV_MODE
        // =====================================================================
        $display("\n--- Phase 2: Verify Initial State ---");
        
        if (dut.state == 2'b00) begin  // CONV_MODE = 0
            $display("✅ T=%0t: Correctly started in CONV_MODE", $time);
        end else begin
            $display("❌ T=%0t: ERROR - Expected CONV_MODE, got %s", $time, state_to_string(dut.state));
        end
        
        // Wait a few cycles to ensure state is stable
        repeat(5) @(posedge clk);
        
        $display("T=%0t: State stable in %s for 5 cycles", $time, state_to_string(dut.state));
        
        // =====================================================================
        // Phase 3: Send timestep pulse and observe transition
        // =====================================================================
        $display("\n--- Phase 3: Timestep Pulse Test ---");
        
        $display("T=%0t: BEFORE timestep pulse:", $time);
        $display("         State: %s", state_to_string(dut.state));
        $display("         timestep: %b", timestep);
        
        $display("         pooling_request_pending: %b", dut.pooling_request_pending);
        
        // Send timestep pulse: LOW → HIGH for 1 cycle → LOW
        @(posedge clk);
        timestep = 1;  // Timestep goes HIGH
        $display("T=%0t: Timestep asserted HIGH", $time);
        
        @(posedge clk);
        timestep = 0;  // Timestep goes LOW (1 cycle pulse)
        $display("T=%0t: Timestep deasserted LOW (1 cycle pulse complete)", $time);
        
        $display("T=%0t: AFTER timestep pulse:", $time);
        $display("         State: %s", state_to_string(dut.state));
        
        $display("         pooling_request_pending: %b", dut.pooling_request_pending);
        
        // =====================================================================
        // Phase 4: Monitor state transitions over several cycles
        // =====================================================================
        $display("\n--- Phase 4: Monitor State Transitions ---");
        
        for (int cycle = 0; cycle < 20; cycle++) begin
            @(posedge clk);
            
            // Print state info every cycle (concise)
            $display("T=%0t: Cycle %2d | State: %s | Conv: ready=%b active=%b | Pool: pending=%b active=%b", 
                     $time, cycle,
                     state_to_string(dut.state),
                     dut.conv_module_ready, dut.conv_module_active,
                     dut.pooling_request_pending, dut.pooling_module_active);
            
            // Highlight important transitions
            if (dut.state != dut.next_state) begin
                $display("    🔄 STATE TRANSITION: %s → %s", 
                         state_to_string(dut.state), state_to_string(dut.next_state));
            end
        end
        
        // =====================================================================
        // Phase 5: Test multiple timestep pulses
        // =====================================================================
        $display("\n--- Phase 5: Multiple Timestep Pulses ---");
        
        // Wait for system to settle
        repeat(10) @(posedge clk);
        
        // Send second timestep pulse
        $display("T=%0t: Sending SECOND timestep pulse...", $time);
        
        @(posedge clk);
        timestep = 1;
        $display("T=%0t: Second timestep HIGH", $time);
        
        @(posedge clk);
        timestep = 0;
        $display("T=%0t: Second timestep LOW", $time);
        
        // Monitor response
        //for (int cycle = 0; cycle < 15; cycle++) begin
        //    @(posedge clk);
        //    $display("T=%0t: Post-2nd-timestep Cycle %2d | State: %s | Pool pending: %b", 
        //             $time, cycle, state_to_string(dut.state), dut.pooling_request_pending);
        //end
        #(1000);
        
        // =====================================================================
        // Phase 6: Summary
        // =====================================================================
        $display("\n--- Test Summary ---");
        $display("Final state: %s", state_to_string(dut.state));
        $display("Conv module status: ready=%b, active=%b", dut.conv_module_ready, dut.conv_module_active);
        $display("Pool status: pending=%b, active=%b", dut.pooling_request_pending, dut.pooling_module_active);
        
        #50;
        $display("\n=== Timestep Transition Test Complete ===");
        $finish;
    end

    // =========================================================================
    // Monitors for key signals (using tracked previous values)
    // =========================================================================
    
    
    
    // Monitor pooling request changes
    always @(posedge clk) begin
        if (dut.pooling_request_pending && !prev_pooling_request) begin
            $display("📌 POOLING REQUEST SET at T=%0t", $time);
        end
        if (!dut.pooling_request_pending && prev_pooling_request) begin
            $display("📌 POOLING REQUEST CLEARED at T=%0t", $time);
        end
    end
    
    // Monitor state machine transitions
    always @(posedge clk) begin
        if (dut.state != prev_state) begin
            $display("🎯 STATE CHANGE: %s → %s at T=%0t", 
                     state_to_string(prev_state), state_to_string(dut.state), $time);
        end
    end
    
    // Monitor key module status
    always @(posedge clk) begin
        // Convolution module status changes
        if (dut.conv_module_ready != prev_conv_ready) begin
            $display("🔧 CONV MODULE READY: %b → %b at T=%0t", 
                     prev_conv_ready, dut.conv_module_ready, $time);
        end
        
        // Pooling module status changes  
        if (dut.pooling_module_active != prev_pool_active) begin
            $display("🏊 POOL MODULE ACTIVE: %b → %b at T=%0t", 
                     prev_pool_active, dut.pooling_module_active, $time);
        end
    end

endmodule