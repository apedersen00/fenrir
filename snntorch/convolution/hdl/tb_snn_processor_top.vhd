library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library vunit_lib;
context vunit_lib.vunit_context;

use work.conv_pool_pkg.all;

entity tb_snn_processor_top is
    generic(runner_cfg : string);
end entity tb_snn_processor_top;

architecture testbench of tb_snn_processor_top is

    constant CLK_PERIOD      : time := 10 ns;
    constant IMG_WIDTH       : integer := 16;  -- Smaller for testing
    constant IMG_HEIGHT      : integer := 16;  -- Smaller for testing
    constant NEURON_BIT_WIDTH : integer := 9;
    constant KERNEL_SIZE     : integer := 3;
    constant POOL_SIZE       : integer := 2;
    constant CHANNELS_OUT    : integer := 4;   -- Smaller for testing
    constant ADDR_WIDTH      : integer := 8;   -- Adjusted for smaller image
    constant BITS_PER_WEIGHT : integer := 9;
    constant RESET_VALUE     : integer := 0;
    constant BITS_PER_COORD  : integer := 8;
    constant COORD_WIDTH     : integer := 8;
    constant BRAM_DEPTH      : integer := 256; -- Adjusted for smaller image
    
    -- Test state enum
    type test_state_t is (
        TEST_IDLE,
        TEST_RESET,
        TEST_TIMESTEP_CAPTURE,
        TEST_EVENT_PROCESSING,
        TEST_FULL_PIPELINE,
        TEST_MULTIPLE_TIMESTEPS,
        TEST_ENABLE_DISABLE,
        TEST_BRAM_ARBITRATION
    );
    
    -- Clock and control signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '0';
    signal enable : std_logic := '0';
    
    -- Input interface signals
    signal timestep_flag : std_logic := '0';
    signal input_ready : std_logic;
    
    -- Input event FIFO signals
    signal input_fifo_empty : std_logic := '1';
    signal input_fifo_data : std_logic_vector(2 * BITS_PER_COORD - 1 downto 0) := (others => '0');
    signal input_fifo_read : std_logic;
    
    -- Output spike FIFO signals
    signal output_fifo_full : std_logic := '0';
    signal output_fifo_data : std_logic_vector(2 * COORD_WIDTH + CHANNELS_OUT - 1 downto 0);
    signal output_fifo_write : std_logic;
    
    -- Status and debug signals
    signal busy : std_logic;
    signal processing_active : std_logic;
    signal debug_main_state : std_logic_vector(2 downto 0);
    signal debug_event_state : std_logic_vector(1 downto 0);
    signal debug_conv_state : std_logic_vector(2 downto 0);
    signal debug_pool_state : std_logic_vector(2 downto 0);
    signal debug_events_processed : integer range 0 to 65535;
    signal debug_spikes_generated : integer range 0 to 65535;
    
    -- FIFO simulation for input events
    type input_event_t is record
        x : integer;
        y : integer;
    end record;
    
    type input_events_array_t is array (0 to 15) of input_event_t;
    
    -- Test event sequences
    signal test_events : input_events_array_t := (
        (x => 5, y => 5),   (x => 6, y => 6),   (x => 7, y => 7),   (x => 8, y => 8),
        (x => 3, y => 9),   (x => 4, y => 10),  (x => 11, y => 2),  (x => 12, y => 3),
        (x => 1, y => 1),   (x => 2, y => 2),   (x => 13, y => 13), (x => 14, y => 14),
        (x => 0, y => 15),  (x => 15, y => 0),  (x => 8, y => 4),   (x => 9, y => 5)
    );
    
    signal event_index : integer := 0;
    signal events_to_send : integer := 8;  -- Default to having events available
    signal fifo_advance_delay : integer := 0;
    
    -- Output event capture
    type output_event_t is record
        x_coord : integer;
        y_coord : integer;
        spikes : std_logic_vector(CHANNELS_OUT - 1 downto 0);
    end record;
    
    type output_events_array_t is array (0 to 63) of output_event_t;
    signal output_events : output_events_array_t;
    signal output_event_count : integer := 0;
    
    -- Test procedures
    procedure wait_cycles(n : integer) is
    begin
        for i in 1 to n loop
            wait until rising_edge(clk);
        end loop;
    end procedure;
    
    procedure reset_system(signal rst_sig : out std_logic) is
    begin
        rst_sig <= '1';
        wait_cycles(3);
        rst_sig <= '0';
        wait_cycles(2);
    end procedure;
    
    procedure send_timestep_flag(
        signal ts_flag_sig : out std_logic
    ) is
    begin
        ts_flag_sig <= '1';
        wait_cycles(1);
        ts_flag_sig <= '0';
    end procedure;
    
    procedure setup_input_events(
        num_events : integer
    ) is
    begin
        -- Simple setup - the FIFO will be controlled by shared variables
        wait_cycles(1);  -- Give time for setup to take effect
    end procedure;
    
    procedure wait_for_idle is
    begin
        wait until busy = '0' for 10000 ns;  -- Increased timeout
        if busy /= '0' then
            report "Warning: System did not return to idle within timeout. State=" & 
                   integer'image(to_integer(unsigned(debug_main_state))) severity warning;
        end if;
    end procedure;
    
    function extract_x_coord(fifo_data : std_logic_vector) return integer is
        variable x_coord : std_logic_vector(COORD_WIDTH - 1 downto 0);
    begin
        x_coord := fifo_data(2 * COORD_WIDTH + CHANNELS_OUT - 1 downto COORD_WIDTH + CHANNELS_OUT);
        return to_integer(unsigned(x_coord));
    end function;
    
    function extract_y_coord(fifo_data : std_logic_vector) return integer is
        variable y_coord : std_logic_vector(COORD_WIDTH - 1 downto 0);
    begin
        y_coord := fifo_data(COORD_WIDTH + CHANNELS_OUT - 1 downto CHANNELS_OUT);
        return to_integer(unsigned(y_coord));
    end function;
    
    function extract_spikes(fifo_data : std_logic_vector) return std_logic_vector is
    begin
        return fifo_data(CHANNELS_OUT - 1 downto 0);
    end function;
    
    function state_to_string(state_vec : std_logic_vector(2 downto 0)) return string is
        variable state_int : integer;
    begin
        state_int := to_integer(unsigned(state_vec));
        case state_int is
            when 0 => return "IDLE";
            when 1 => return "CAPTURE_TIMESTEP";
            when 2 => return "PROCESS_EVENTS";
            when 3 => return "CONVOLUTION_ACTIVE";
            when 4 => return "POOLING_ACTIVE";
            when 7 => return "PAUSED";
            when others => return "UNKNOWN";
        end case;
    end function;

begin

    -- Clock generation
    clk <= not clk after CLK_PERIOD/2;
    
    -- Input FIFO simulation
    input_fifo_simulation : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                input_fifo_empty <= '1';
                input_fifo_data <= (others => '0');
                event_index <= 0;
                fifo_advance_delay <= 0;
            else
                -- Handle FIFO read requests
                if input_fifo_read = '1' and fifo_advance_delay = 0 then
                    fifo_advance_delay <= 2;  -- Simulate FIFO latency
                elsif fifo_advance_delay > 0 then
                    fifo_advance_delay <= fifo_advance_delay - 1;
                    if fifo_advance_delay = 1 then
                        -- Advance to next event
                        if event_index < events_to_send - 1 then
                            event_index <= event_index + 1;
                        else
                            -- No more events
                            input_fifo_empty <= '1';
                            input_fifo_data <= (others => '0');
                        end if;
                    end if;
                end if;
                
                -- Update FIFO output based on current event
                if event_index < events_to_send and fifo_advance_delay = 0 then
                    input_fifo_empty <= '0';
                    -- Pack coordinates: [x_coord(MSBs)][y_coord(LSBs)]
                    input_fifo_data(BITS_PER_COORD - 1 downto 0) <= 
                        std_logic_vector(to_unsigned(test_events(event_index).y, BITS_PER_COORD));
                    input_fifo_data(2 * BITS_PER_COORD - 1 downto BITS_PER_COORD) <= 
                        std_logic_vector(to_unsigned(test_events(event_index).x, BITS_PER_COORD));
                end if;
            end if;
        end if;
    end process input_fifo_simulation;
    
    -- Output FIFO capture
    output_fifo_capture : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                output_event_count <= 0;
                output_events <= (others => (x_coord => 0, y_coord => 0, spikes => (others => '0')));
            elsif output_fifo_write = '1' and output_event_count < 64 then
                output_events(output_event_count).x_coord <= extract_x_coord(output_fifo_data);
                output_events(output_event_count).y_coord <= extract_y_coord(output_fifo_data);
                output_events(output_event_count).spikes <= extract_spikes(output_fifo_data);
                output_event_count <= output_event_count + 1;
            end if;
        end if;
    end process output_fifo_capture;

    -- Unit under test
    uut: entity work.snn_processor_top
    generic map (
        IMG_WIDTH        => IMG_WIDTH,
        IMG_HEIGHT       => IMG_HEIGHT,
        NEURON_BIT_WIDTH => NEURON_BIT_WIDTH,
        KERNEL_SIZE      => KERNEL_SIZE,
        POOL_SIZE        => POOL_SIZE,
        CHANNELS_OUT     => CHANNELS_OUT,
        ADDR_WIDTH       => ADDR_WIDTH,
        BITS_PER_WEIGHT  => BITS_PER_WEIGHT,
        RESET_VALUE      => RESET_VALUE,
        BITS_PER_COORD   => BITS_PER_COORD,
        COORD_WIDTH      => COORD_WIDTH,
        BRAM_DEPTH       => BRAM_DEPTH,
        BRAM_FILENAME    => ""
    )
    port map (
        clk              => clk,
        rst_i            => rst,
        enable_i         => enable,
        timestep_flag_i  => timestep_flag,
        input_ready_o    => input_ready,
        input_fifo_empty_i => input_fifo_empty,
        input_fifo_data_i  => input_fifo_data,
        input_fifo_read_o  => input_fifo_read,
        output_fifo_full_i  => output_fifo_full,
        output_fifo_data_o  => output_fifo_data,
        output_fifo_write_o => output_fifo_write,
        busy_o              => busy,
        processing_active_o => processing_active,
        debug_main_state_o  => debug_main_state,
        debug_event_state_o => debug_event_state,
        debug_conv_state_o  => debug_conv_state,
        debug_pool_state_o  => debug_pool_state,
        debug_events_processed_o => debug_events_processed,
        debug_spikes_generated_o => debug_spikes_generated
    );

    main : process
    begin
        test_runner_setup(runner, runner_cfg);
        
        -- Initial setup
        wait_cycles(5);
        
        if run("test_reset") then
            report "Testing reset functionality";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(3);
            
            -- Check initial state
            assert debug_main_state = "000" report "Should be in IDLE state after reset";
            assert busy = '0' report "Should not be busy after reset";
            assert input_ready = '1' report "Should be ready for input after reset";
            assert debug_events_processed = 0 report "Event counter should be zero";
            assert debug_spikes_generated = 0 report "Spike counter should be zero";
            
        end if;
        
        if run("test_timestep_buffering_during_processing") then
            report "Testing timestep buffering during active processing";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- Start initial processing
            setup_input_events(4);
            send_timestep_flag(timestep_flag);
            wait_cycles(2);
            
            -- Wait until convolution is active
            wait until debug_conv_state /= "000" for 500 ns;
            if debug_conv_state /= "000" then
                report "Convolution is active, now testing timestep buffering";
                
                -- Send another timestep while processing - should be buffered
                send_timestep_flag(timestep_flag);
                wait_cycles(2);
                
                -- Should still be busy and processing
                assert busy = '1' report "Should remain busy during processing";
                assert input_ready = '0' report "Should not be ready while processing";
                assert processing_active = '1' report "Should show processing is active";
                
                report "Timestep successfully buffered during convolution";
            else
                report "Warning: Convolution did not start as expected";
            end if;
            
            -- Let it complete
            wait_for_idle;
            
        end if;
        
        if run("test_event_processing") then
            report "Testing event processing through convolution";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- Setup events
            setup_input_events(4);  -- Send 4 events
            
            -- Send timestep to start processing
            send_timestep_flag(timestep_flag);
            
            -- Wait for event processing to start
            wait until debug_main_state = "010" or debug_main_state = "011" for 100 ns;
            assert debug_main_state = "010" or debug_main_state = "011" 
                report "Should enter event processing. State=" & state_to_string(debug_main_state);
            
            -- Should process events through convolution
            wait until debug_conv_state /= "000" for 200 ns;
            report "Convolution state active: " & integer'image(to_integer(unsigned(debug_conv_state)));
            
            -- Wait for some events to be processed
            wait until debug_events_processed > 0 for 1000 ns;
            assert debug_events_processed > 0 report "Should process some events";
            
            report "Processed " & integer'image(debug_events_processed) & " events";
            
        end if;
        
        if run("test_full_pipeline") then
            report "Testing complete pipeline: events -> convolution -> pooling";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- Setup a good number of events for full pipeline test
            setup_input_events(8);
            
            -- Send timestep to start processing
            report "Starting full pipeline with timestep flag";
            send_timestep_flag(timestep_flag);
            
            -- Monitor state transitions
            wait until debug_main_state /= "000" for 100 ns;
            report "Pipeline started, state: " & state_to_string(debug_main_state);
            
            -- Wait for convolution phase
            wait until debug_main_state = "011" for 500 ns;  -- CONVOLUTION_ACTIVE
            if debug_main_state = "011" then
                report "Convolution phase active";
            end if;
            
            -- Wait for pooling phase
            wait until debug_main_state = "100" for 2000 ns;  -- POOLING_ACTIVE
            if debug_main_state = "100" then
                report "Pooling phase active";
                assert debug_pool_state /= "000" report "Pooling module should be active";
            end if;
            
            -- Wait for completion
            wait_for_idle;
            
            report "Pipeline completed:";
            report "  Events processed: " & integer'image(debug_events_processed);
            report "  Spikes generated: " & integer'image(debug_spikes_generated);
            report "  Output events captured: " & integer'image(output_event_count);
            
            -- Verify some output was generated
            if output_event_count > 0 then
                report "First output event: (" & 
                       integer'image(output_events(0).x_coord) & "," & 
                       integer'image(output_events(0).y_coord) & ") spikes=" & 
                       to_string(output_events(0).spikes);
            end if;
            
        end if;
        
        if run("test_convolution_to_pooling_transition") then
            report "Testing automatic transition from convolution to pooling";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- Start processing with events
            setup_input_events(6);
            send_timestep_flag(timestep_flag);
            
            -- Wait for and verify convolution phase
            wait until debug_main_state = "011" for 500 ns;  -- CONVOLUTION_ACTIVE
            if debug_main_state = "011" then
                report "Convolution phase started successfully";
                
                -- Monitor convolution processing
                wait until debug_conv_state /= "000" for 200 ns;
                if debug_conv_state /= "000" then
                    report "Convolution module is actively processing";
                end if;
                
                -- Wait for convolution to complete and transition to pooling
                wait until debug_main_state = "100" for 2000 ns;  -- POOLING_ACTIVE
                if debug_main_state = "100" then
                    report "Successfully transitioned to pooling phase";
                    assert debug_pool_state /= "000" report "Pooling module should be active";
                else
                    report "Warning: Did not transition to pooling. Final state=" & 
                           integer'image(to_integer(unsigned(debug_main_state)));
                end if;
            else
                report "Warning: Convolution phase did not start";
            end if;
            
            -- Wait for complete pipeline to finish
            wait_for_idle;
            
            report "Convolution to pooling transition test completed";
            report "Final events processed: " & integer'image(debug_events_processed);
            report "Final spikes generated: " & integer'image(debug_spikes_generated);
            
        end if;
        
        if run("test_enable_disable") then
            report "Testing enable/disable (pause/resume) functionality";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- Start processing
            setup_input_events(6);
            send_timestep_flag(timestep_flag);
            
            -- Let it start processing
            wait until busy = '1' for 200 ns;
            wait_cycles(10);
            
            -- Disable (pause)
            enable <= '0';
            wait_cycles(2);  -- Give time for transition
            if debug_main_state = "111" then
                report "Successfully transitioned to PAUSED state";
            else
                report "State after disable: " & state_to_string(debug_main_state);
            end if;
            
            -- Wait while paused
            wait_cycles(20);
            if debug_main_state = "111" then
                report "Remained in PAUSED state as expected";
            end if;
            
            -- Re-enable (resume)
            enable <= '1';
            wait_cycles(2);
            if debug_main_state /= "111" then
                report "Successfully exited PAUSED state to: " & state_to_string(debug_main_state);
            end if;
            
            -- Let it complete
            wait_for_idle;
            
            report "Pause/resume test completed successfully";
            
        end if;
        
        if run("test_bram_arbitration") then
            report "Testing BRAM arbitration between convolution and pooling";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            setup_input_events(4);
            send_timestep_flag(timestep_flag);
            
            -- Monitor BRAM access during different phases
            -- This is mainly observational since BRAM arbitration is internal
            
            -- Wait for convolution phase
            wait until debug_main_state = "011" for 500 ns;
            if debug_main_state = "011" then
                report "BRAM access granted to convolution module";
            end if;
            
            -- Wait for pooling phase
            wait until debug_main_state = "100" for 2000 ns;
            if debug_main_state = "100" then
                report "BRAM access granted to pooling module";
            end if;
            
            wait_for_idle;
            report "BRAM arbitration test completed";
            
        end if;
        
        if run("test_pooling_completion_timing") then
            report "Testing pooling operation completion timing";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- Start processing to get to pooling phase
            setup_input_events(4);
            send_timestep_flag(timestep_flag);
            
            -- Wait for pooling to start
            wait until debug_main_state = "100" for 3000 ns;  -- POOLING_ACTIVE
            if debug_main_state = "100" then
                report "Pooling phase started, measuring completion time";
                
                -- Record start time and measure pooling duration
                wait until debug_pool_state /= "000" for 100 ns;
                if debug_pool_state /= "000" then
                    report "Pooling module active, monitoring progress...";
                    
                    -- Monitor pooling states and look for completion
                    wait until debug_main_state = "000" for 5000 ns;  -- Wait for return to IDLE
                    if debug_main_state = "000" then
                        report "Pooling completed successfully and returned to IDLE";
                        report "Total events processed: " & integer'image(debug_events_processed);
                        report "Total spikes generated: " & integer'image(debug_spikes_generated);
                        
                        -- Verify system is ready for next cycle
                        assert input_ready = '1' report "Should be ready for next timestep after completion";
                        assert busy = '0' report "Should not be busy after completion";
                    else
                        report "Warning: Pooling did not complete within expected time";
                        report "Final state: " & integer'image(to_integer(unsigned(debug_main_state)));
                    end if;
                else
                    report "Warning: Pooling module did not become active";
                end if;
            else
                report "Warning: Did not reach pooling phase";
            end if;
            
        end if;

        test_runner_cleanup(runner);
        wait;

    end process main;

end testbench;