import snn_interfaces_pkg::*;

module fast_conv_controller #(
    
    // =========================================================================
    // Core params
    // =========================================================================
    parameter int COORD_BITS        = DEFAULT_COORD_BITS,
    parameter int IMG_WIDTH         = DEFAULT_IMG_WIDTH,
    parameter int IMG_HEIGHT        = DEFAULT_IMG_HEIGHT,
    parameter int CHANNELS          = DEFAULT_CHANNELS,
    parameter int BITS_PER_CHANNEL  = DEFAULT_NEURON_BITS, 

    // =========================================================================
    // MEMORY PARAMS
    // =========================================================================
    parameter int BRAM_DATA_WIDTH   = CHANNELS * BITS_PER_CHANNEL,
    parameter int BRAM_ADDR_WIDTH   = $clog2(IMG_WIDTH * IMG_HEIGHT),

    // =========================================================================
    // Input FIFO params
    // =========================================================================
    parameter int FIFO_DATA_WIDTH           = 2 * COORD_BITS,
    parameter int INPUT_FIFO_EVENT_CAPACITY = DEFAULT_INPUT_FIFO_EVENT_CAPACITY, // 1024 samples
    parameter int INPUT_FIFO_ADDR_WIDTH     = $clog2(INPUT_FIFO_EVENT_CAPACITY)

)(
    // =========================================================================
    // Clock and reset
    // =========================================================================

    input logic clk,
    input logic rst_n,

    // =========================================================================
    // External control signals - can be refactored to an interface later
    // =========================================================================
    input logic sys_enable,
    input logic sys_reset, // reset everything, including the fifo
    input logic timestep, //for signaling the start of a new timestep -> start pooling the membrane potentials

    // =========================================================================
    // Status outputs
    // =========================================================================
    output logic system_active, // indicates if the system is currently processing
    output logic fifo_empty, // indicates if the input fifo is empty
    output logic fifo_full, // indicates if the input fifo is full

    // =========================================================================
    // Input Data interface for spike events -> fifo
    // =========================================================================
    input logic [FIFO_DATA_WIDTH - 1 : 0] spike_event,
    input logic write_enable, // write enable for the input fifo

    // =========================================================================
    // Output FIFO
    // =========================================================================
    input logic output_fifo_full

);
    // =========================================================================
    // Interfaces for internal modules
    // =========================================================================
    snn_control_if  capture_ctrl_bus();
    snn_control_if  conv_ctrl_bus();
    snn_event_if    capture_to_conv_bus();
    snn_control_if  arbiter_ctrl_bus();
    snn_control_if  pooling_ctrl_bus();

    fifo_if #(
        .DATA_WIDTH(FIFO_DATA_WIDTH),
        .ADDR_WIDTH(INPUT_FIFO_ADDR_WIDTH)
    ) controller_fifo_bus(
        .clk(clk),
        .rst_n(rst_n)
    );

    assign controller_fifo_bus.write_data = spike_event;
    assign controller_fifo_bus.write_en = write_enable && sys_enable && !controller_fifo_bus.full;
    assign fifo_empty = controller_fifo_bus.empty;
    assign fifo_full = controller_fifo_bus.full;

    // Arbiter interface for convolution and pooling modules
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    )  conv_read_bus();
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) conv_write_bus();
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) pool_read_bus();
    arbiter_if #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL)
    ) pool_write_bus();

    dp_bram_if #(
        .DATA_WIDTH(BRAM_DATA_WIDTH),
        .ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) bram_bus();

    // =========================================================================
    // State machine type and signals
    // =========================================================================
    typedef enum logic [1:0] {
        CONV_MODE,
        CONV_FINISHING,
        POOL_MODE,
        PAUSE_POOLING
    } processor_state_t;

    processor_state_t state, next_state;

    // =========================================================================
    // Timestep control and buffer
    // =========================================================================
    logic pooling_request_pending;


    // Buffer timestep until pooling can be processed
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            pooling_request_pending <= 1'b0;
        end else if (sys_reset) begin
            pooling_request_pending <= 1'b0;
        end else if (timestep) begin
            pooling_request_pending <= 1'b1;
        end else if (state == POOL_MODE) begin
            pooling_request_pending <= 1'b0;
        end
    end

    // =========================================================================
    // signals from internal modules
    // =========================================================================
    logic conv_module_active;
    logic pooling_module_active;
    logic conv_module_ready;
    logic pooling_module_done;

    // =========================================================================
    // State machine for processing modes
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= CONV_MODE;
        end else if (sys_reset) begin
            state <= CONV_MODE;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        next_state = state; // Default to current state
    
        case (state)

            CONV_MODE: begin
                if (pooling_request_pending && conv_module_ready) begin // Conv module is not doing anything if it is ready so can switch immediately
                    next_state = POOL_MODE;
                end else if (pooling_request_pending && !conv_module_active) begin
                    next_state = CONV_FINISHING;
                end else begin
                    next_state = CONV_MODE; // Stay in convolution mode
                end 
            end

            CONV_FINISHING: begin
                if (conv_module_ready) begin
                    next_state = POOL_MODE;
                end else begin
                    next_state = CONV_FINISHING; // Wait for convolution module to finish
                end
            end

            POOL_MODE: begin
                if (output_fifo_full) begin
                    next_state = PAUSE_POOLING;
                end else if (pooling_module_done && !pooling_request_pending) begin
                    next_state = CONV_MODE;
                end else begin
                    next_state = POOL_MODE;
                end 
            end

            PAUSE_POOLING: begin
                if (!output_fifo_full) begin
                    next_state = POOL_MODE; // Resume pooling when output FIFO is not full
                end else begin
                    next_state = PAUSE_POOLING; // Stay in pause state until output FIFO is ready
                end 
            end
        endcase
    end

    // =========================================================================
    // Control Signal Routing
    // =========================================================================
    assign conv_ctrl_bus.clk = clk;
    assign conv_ctrl_bus.enable = sys_enable;
    assign conv_ctrl_bus.reset = sys_reset || !rst_n;
    
    assign capture_ctrl_bus.clk = clk;
    assign capture_ctrl_bus.enable = sys_enable;
    assign capture_ctrl_bus.reset = sys_reset || !rst_n;

    assign pooling_ctrl_bus.clk = clk;
    assign pooling_ctrl_bus.enable = sys_enable && (state == POOL_MODE);
    assign pooling_ctrl_bus.reset = sys_reset || !rst_n;
    assign pooling_module_done = pooling_ctrl_bus.done;
    assign pooling_module_active = pooling_ctrl_bus.active;


    assign arbiter_ctrl_bus.clk = clk;
    assign arbiter_ctrl_bus.enable = sys_enable;
    assign arbiter_ctrl_bus.reset = sys_reset || !rst_n;
    assign arbiter_ctrl_bus.conv_or_pool = (state == CONV_MODE || state == CONV_FINISHING);
    

    // =========================================================================
    // Status Signal Aggregation
    // =========================================================================

    assign conv_module_active = capture_ctrl_bus.active || conv_ctrl_bus.active;
    assign conv_module_ready = conv_ctrl_bus.ready;

    assign system_active = conv_module_active || pooling_module_active;

    // =========================================================================
    // FIFO Instance
    // =========================================================================
    
    fifo #(
        .DATA_WIDTH(FIFO_DATA_WIDTH),
        .ADDR_WIDTH(INPUT_FIFO_ADDR_WIDTH)
    ) input_fifo (
        .fifo_port(controller_fifo_bus.fifo_module)
    );

    // =========================================================================
    // CApture event instance
    // =========================================================================

    capture_event #(
        .DATA_WIDTH(FIFO_DATA_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .IMG_WIDTH(IMG_WIDTH)
    ) capture_event_inst (
        .fifo_port(controller_fifo_bus.consumer),
        .conv_port(capture_to_conv_bus.capture),
        .ctrl_port(capture_ctrl_bus.capture)
    );

    // =========================================================================
    // Fast Convolution Instance
    // =========================================================================

    fast_conv #(
        .COORD_BITS(COORD_BITS),
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .KERNEL_SIZE(3)
    ) fast_conv_inst (
        .ctrl_port(conv_ctrl_bus.convolution),
        .event_port(capture_to_conv_bus.convolution),
        .mem_read(conv_read_bus.read_port),
        .mem_write(conv_write_bus.write_port)
    );

    // =========================================================================
    // Sum Pooling Instance
    // =========================================================================
    sum_pooling #(
        .CHANNELS(CHANNELS),
        .BITS_PER_CHANNEL(BITS_PER_CHANNEL),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) sum_pooling_inst (
        .ctrl_port(pooling_ctrl_bus.pooling),
        .mem_read(pool_read_bus.read_port),
        .mem_write(pool_write_bus.write_port)
    );

    // =========================================================================
    // Dual port bram instance
    // =========================================================================

    dp_bram #(
        .DATA_WIDTH(BRAM_DATA_WIDTH),
        .ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) feature_map_memory (
        .bram_port(bram_bus.bram_module)
    );

    // =========================================================================
    // Arbiter Instance
    // =========================================================================
    arbiter #(
    .COORD_BITS(COORD_BITS),
    .CHANNELS(CHANNELS),
    .BITS_PER_CHANNEL(BITS_PER_CHANNEL),
    .IMG_WIDTH(IMG_WIDTH),
    .IMG_HEIGHT(IMG_HEIGHT),
    .BRAM_DATA_WIDTH(BRAM_DATA_WIDTH),
    .BRAM_ADDR_WIDTH(BRAM_ADDR_WIDTH)
    ) arbiter_inst (
        .clk(clk),
        .rst_n(rst_n),

        .ctrl_port(arbiter_ctrl_bus.arbiter),
        .conv_read_port(conv_read_bus.arbiter),
        .conv_write_port(conv_write_bus.arbiter),

        .pool_read_port(pool_read_bus.arbiter),
        .pool_write_port(pool_write_bus.arbiter),

        // BRAM interface
        .bram_port(bram_bus.arbiter)
    );

    // =========================================================================
    // Debug and Monitoring
    // =========================================================================
    initial begin
        $display("=== Fast Convolution Controller Configuration ===");
        $display("Image Size: %0dx%0d pixels", IMG_WIDTH, IMG_HEIGHT);
        $display("Channels: %0d @ %0d bits each = %0d total bits", 
                 CHANNELS, BITS_PER_CHANNEL, BRAM_DATA_WIDTH);
        $display("Input FIFO: %0d entries @ %0d bits each", 
                 INPUT_FIFO_EVENT_CAPACITY, FIFO_DATA_WIDTH);
        $display("BRAM: %0d addresses @ %0d bits each", 
                 2**BRAM_ADDR_WIDTH, BRAM_DATA_WIDTH);
        $display("State Machine: CONV/POOL with output flow control");
        $display("================================================");
    end

    
        
    always @(posedge clk) begin
        // Check for X/Z values on critical signals
        if ($isunknown(capture_ctrl_bus.active)) begin
            $display("T=%0t: WARNING: capture_ctrl_bus.active is X/Z!", $time);
        end
        
        if ($isunknown(conv_ctrl_bus.active)) begin
            $display("T=%0t: WARNING: conv_ctrl_bus.active is X/Z!", $time);
        end
        
        if ($isunknown(conv_ctrl_bus.ready)) begin
            $display("T=%0t: WARNING: conv_ctrl_bus.ready is X/Z!", $time);
        end
        
        // Monitor FIFO and capture interaction
        if (!controller_fifo_bus.empty) begin
            $display("T=%0t: FIFO not empty - capture should activate", $time);
            $display("        capture_enable=%b, capture_reset=%b, capture_active=%b", 
                     capture_ctrl_bus.enable, capture_ctrl_bus.reset, capture_ctrl_bus.active);
        end
        
        // Monitor event flow
        if (capture_to_conv_bus.event_valid) begin
            $display("T=%0t: Event valid: coord=(%0d,%0d), ready=%b, ack=%b", 
                     $time, 
                     capture_to_conv_bus.event_coord.x,
                     capture_to_conv_bus.event_coord.y,
                     capture_to_conv_bus.event_ready,
                     capture_to_conv_bus.event_ack);
        end
    end

endmodule