library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.conv_pool_pkg.all;

entity tb_conv_pool_fast is
end entity tb_conv_pool_fast;

architecture testbench of tb_conv_pool_fast is

    constant CLK_PERIOD : time := 10 ns;
    constant BITS_PER_COORD : integer := 8;
    constant CHANNELS_OUT : integer := 12;
    constant BITS_PER_NEURON : integer := 6;
    -- control signals
    signal clk : std_logic := '1';
    signal rst_o, enable_o, timestep_o : std_logic := '0';
    
    --event signals
    signal event_fifo_empty_o : std_logic := '1';
    -- shape of event bus: [x_coord(8), y_coord(8), channel(1), ...]
    signal event_fifo_bus_o : std_logic_vector(2 * BITS_PER_COORD downto 0) := (others => '0');
    signal event_fifo_read_i : std_logic;

    -- extra debug signals
    signal uut_main_state, uut_next_state, uut_last_state : main_state_et;
    signal uut_main_state_vec, uut_next_state_vec, uut_last_state_vec : std_logic_vector(2 downto 0);

    signal uut_timestep_pending : std_logic;
    signal uut_current_event : event_tensor_t;
    signal uut_event_valid : std_logic;
    signal uut_read_cycle : integer;

    -- uut ram signals
    signal uut_mem_neuron_wea, uut_mem_neuron_web, uut_mem_neuron_ena, uut_mem_neuron_enb : std_logic;
    signal uut_mem_neuron_addra, uut_mem_neuron_addrb : std_logic_vector(9 downto 0);  -- FIXED: 10-bit
    signal uut_mem_neuron_dia, uut_mem_neuron_dib : std_logic_vector(CHANNELS_OUT * BITS_PER_NEURON - 1 downto 0);  -- FIXED: CHANNELS_OUT
    signal uut_mem_neuron_doa, uut_mem_neuron_dob : std_logic_vector(CHANNELS_OUT * BITS_PER_NEURON - 1 downto 0);  -- FIXED: CHANNELS_OUT
    signal uut_total_coords_to_update : integer;  -- ADD: Missing signal

    signal uut_conv_thread_1, uut_conv_thread_2 : conv_thread_t;  -- ADD: Missing conv threads
    signal uut_conv_in_progress : std_logic;  -- ADD: Missing conv in progress signal

    -- ========================================= TIMING PROCEDURES =========================================
    procedure waitf(n : in integer) is
    begin
        for i in 1 to n loop
            wait until rising_edge(clk);
        end loop;
    end procedure waitf;

    -- Drive a signal and wait for it to settle
    procedure drive_and_settle(signal sig : out std_logic; value : std_logic; settle_cycles : integer := 1) is
    begin
        sig <= value;
        waitf(settle_cycles);
    end procedure drive_and_settle;

    -- IMPROVED: Check signals after rising edge with small delay for stability
    procedure check_signal_stable(signal sig : in std_logic; expected : std_logic; error_msg : string) is
    begin
        waitf(1); -- Wait for next rising edge
        wait for 2 ns; -- Small delay for signal stability after clock edge
    assert sig = expected
        report error_msg & " - Expected: " & std_logic'image(expected) & 
                   ", Got: " & std_logic'image(sig)
            severity failure;
    end procedure check_signal_stable;

    -- IMPROVED: Check state signals with rising edge timing
    procedure check_state_stable(signal state_sig : in main_state_et; expected : main_state_et; error_msg : string) is
    begin
        waitf(1); -- Wait for next rising edge  
        wait for 2 ns; -- Small delay for stability
        assert state_sig = expected
        report error_msg & " - Expected: " & main_state_et'image(expected) & 
               ", Got: " & main_state_et'image(state_sig)
            severity failure;
    end procedure check_state_stable;

    -- IMPROVED: Check event tensor with rising edge timing
    procedure check_event_tensor_stable(signal tensor_sig : in event_tensor_t; expected : event_tensor_t; error_msg : string) is
begin
    waitf(1); -- Wait for next rising edge
    wait for 2 ns; -- Small delay for stability
    assert (tensor_sig.x_coord = expected.x_coord and 
            tensor_sig.y_coord = expected.y_coord and 
            tensor_sig.channel = expected.channel)
        report error_msg & 
               " - Expected: (" & integer'image(expected.x_coord) & 
               ", " & integer'image(expected.y_coord) & 
               ", " & integer'image(expected.channel) & ")" &
               ", Got: (" & integer'image(tensor_sig.x_coord) & 
               ", " & integer'image(tensor_sig.y_coord) & 
               ", " & integer'image(tensor_sig.channel) & ")"
        severity failure;
end procedure check_event_tensor_stable;

    procedure drive_event_tensor(signal ebus : out std_logic_vector; tensor : event_tensor_t; settle_cycles : integer := 1) is
    begin
        ebus <= tensor_to_bus(tensor, BITS_PER_COORD, 1);
        waitf(settle_cycles);
    end procedure drive_event_tensor;

    -- ALTERNATIVE: Even simpler approach - check immediately after driving
    procedure drive_and_check(signal sig : out std_logic; value : std_logic; error_msg : string) is
    begin
        sig <= value;
        waitf(2); -- Give 2 cycles for response
        wait for 2 ns; -- Small stability delay
        -- Now check whatever response you expect
    end procedure drive_and_check;

    procedure check_signal_now(signal sig : in std_logic; expected : std_logic; error_msg : string) is
    begin
        wait for 2 ns; -- Just stability delay, no clock waiting
        assert sig = expected
            report error_msg & " - Expected: " & std_logic'image(expected) & 
               ", Got: " & std_logic'image(sig)
            severity failure;
    end procedure check_signal_now;

    procedure check_state_now(signal state_sig : in main_state_et; expected : main_state_et; error_msg : string) is
    begin
        wait for 2 ns; -- Just stability delay, no clock waiting
        assert state_sig = expected
            report error_msg & " - Expected: " & main_state_et'image(expected) & 
                   ", Got: " & main_state_et'image(state_sig)
            severity failure;
    end procedure check_state_now;

    procedure check_event_tensor_now(signal tensor_sig : in event_tensor_t; expected : event_tensor_t; error_msg : string) is
    begin
        wait for 2 ns; -- Just stability delay, no clock waiting
        assert (tensor_sig.x_coord = expected.x_coord and 
                tensor_sig.y_coord = expected.y_coord and 
                tensor_sig.channel = expected.channel)
            report error_msg & 
                   " - Expected: (" & integer'image(expected.x_coord) & 
                   ", " & integer'image(expected.y_coord) & 
                   ", " & integer'image(expected.channel) & ")" &
                   ", Got: (" & integer'image(tensor_sig.x_coord) & 
                   ", " & integer'image(tensor_sig.y_coord) & 
                   ", " & integer'image(tensor_sig.channel) & ")"
            severity failure;
    end procedure check_event_tensor_now;

begin

    clk_process: process
    begin
        clk <= '1';
        wait for CLK_PERIOD/2;
        clk <= '0';
        wait for CLK_PERIOD/2;
    end process;

    uut: entity work.conv_pool_fast
    generic map (
        BITS_PER_COORD => BITS_PER_COORD,
        CHANNELS_OUT => CHANNELS_OUT,
        BITS_PER_NEURON => BITS_PER_NEURON
    )
    port map (
        clk => clk,
        rst_i => rst_o,
        enable_i => enable_o,
        timestep_i => timestep_o,
        event_fifo_empty_i => event_fifo_empty_o,
        event_fifo_bus_i => event_fifo_bus_o,
        event_fifo_read_o => event_fifo_read_i,

        debug_main_state => uut_main_state,
        debug_next_state => uut_next_state,
        debug_last_state => uut_last_state,
        debug_timestep_pending => uut_timestep_pending,
        debug_current_event => uut_current_event,
        debug_event_valid => uut_event_valid,
        debug_read_cycle => uut_read_cycle,
        debug_mem_neuron_wea => uut_mem_neuron_wea,
        debug_mem_neuron_web => uut_mem_neuron_web,
        debug_mem_neuron_ena => uut_mem_neuron_ena,
        debug_mem_neuron_enb => uut_mem_neuron_enb,
        debug_mem_neuron_addra => uut_mem_neuron_addra,
        debug_mem_neuron_addrb => uut_mem_neuron_addrb,
        debug_mem_neuron_dia => uut_mem_neuron_dia,
        debug_mem_neuron_dib => uut_mem_neuron_dib,
        debug_mem_neuron_doa => uut_mem_neuron_doa,
        debug_mem_neuron_dob => uut_mem_neuron_dob,
        debug_total_coords_to_update => uut_total_coords_to_update,
        debug_conv_thread_1 => uut_conv_thread_1,
        debug_conv_thread_2 => uut_conv_thread_2,
        debug_convolution_in_progress => uut_conv_in_progress
    );

    main : process
        variable test_tensor : event_tensor_t;
    begin

        -- Initial stabilization
        waitf(10);

        -- Test: test_reset_no_enable
        report "Running test: test_reset_no_enable";
        -- Lets reset the module
            -- Reset and enable 0
            drive_and_settle(rst_o, '1', 1);
            drive_and_settle(enable_o, '0', 1); -- Disable the module during reset
            drive_and_settle(timestep_o, '0', 1);
            drive_and_settle(event_fifo_empty_o, '1', 1); -- Ensure FIFO is empty
            check_state_stable(
                uut_main_state, RESET, "uut_main_state should be RESET after reset signal"
            );
            check_signal_stable(
                event_fifo_read_i, '0', "uut should not request fifo read when reset"
            );

            waitf(1); -- Wait for next clock edge
            -- Now disable reset and check state
            drive_and_settle(rst_o, '0', 1);
            -- should go to pause when enable is 0
            check_state_stable(
                uut_main_state, PAUSE, "uut_main_state should be PAUSE after reset and enable 0"
            );
        report "Test test_reset_no_enable completed";

        -- Test: test_reset_enable
        report "Running test: test_reset_enable";
        -- Reset and enable 1
            drive_and_settle(rst_o, '1', 2);
            drive_and_settle(enable_o, '1', 2); -- Enable the module during reset
            drive_and_settle(timestep_o, '0', 2);
            drive_and_settle(event_fifo_empty_o, '1', 2); -- Ensure FIFO is empty
            check_state_stable(
                uut_main_state, RESET, "uut_main_state should be RESET after reset signal"
            );
            check_signal_stable(
                event_fifo_read_i, '0', "uut should not request fifo read when reset"
            );

            -- Now disable reset and check state
            drive_and_settle(rst_o, '0', 2);
            check_state_stable(
                uut_main_state, IDLE, "uut_main_state should be IDLE after reset and enable"
            );
        report "Test test_reset_enable completed";

        -- Test: test_fifo_read_request
        report "Running test: test_fifo_read_request";
        -- Setup: Get to IDLE state with FIFO empty
            drive_and_settle(rst_o, '0', 2);
            drive_and_settle(enable_o, '1', 2);
            drive_and_settle(timestep_o, '0', 2);
            drive_and_settle(event_fifo_empty_o, '1', 2);
            
            check_state_stable(uut_main_state, IDLE, "should be in IDLE");
            check_signal_stable(event_fifo_read_i, '0', "no read request in IDLE");
            
            -- Signal that FIFO has data available
            event_fifo_empty_o <= '0';
            waitf(1); -- Wait for state transition
            
            -- Cycle 1 of READ_REQUEST: Assert read request
            check_state_now(uut_main_state, READ_REQUEST, "should enter READ_REQUEST when FIFO not empty");
            check_signal_now(event_fifo_read_i, '1', "should assert read request on cycle 1");
            waitf(1); -- Move to next cycle
            -- FIFO responds with data (simulating 1-cycle FIFO read latency)
            test_tensor := create_tensor(x_coord => 10, y_coord => 10, channel => 0);
            event_fifo_bus_o <= tensor_to_bus(test_tensor, BITS_PER_COORD, 1);
            
            --waitf(1); -- Move to cycle 2 of READ_REQUEST
            
            -- Cycle 2 of READ_REQUEST: Capture data, read request goes low
            check_state_now(uut_main_state, READ_REQUEST, "should stay in READ_REQUEST on cycle 2");
            check_signal_now(event_fifo_read_i, '0', "read request should be low on cycle 2");
            -- Event capture happens at end of this cycle
            
            waitf(1); -- Move to EVENT_CONV
            
            -- Should now be in EVENT_CONV with captured data
            check_state_now(uut_main_state, EVENT_CONV, "should transition to EVENT_CONV after data capture");
            check_signal_now(uut_event_valid, '1', "event should be valid in EVENT_CONV");
            check_event_tensor_now(uut_current_event, test_tensor, "should have captured correct event");
        report "Test test_fifo_read_request completed";

        report "All tests completed successfully";
        wait;

    end process main;

end testbench;