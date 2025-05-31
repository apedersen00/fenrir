library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.conv_pool_pkg.all;

entity tb_conv_pool is
end entity tb_conv_pool;

architecture testbench of tb_conv_pool is
    
    constant CLK_PERIOD : time := 10 ns;
    constant BITS_PER_COORD : integer := 8;
    constant CHANNELS_IN : integer := 1;

    -- control signals
    signal clk : std_logic := '1';
    signal rst_o, enable_o, timestep_o : std_logic := '0';
    
    --event signals
    signal event_fifo_empty_o : std_logic := '1';
    -- shape of event bus: [x_coord(8), y_coord(8), channel(1), ...]
    signal event_fifo_bus_o : std_logic_vector(2 * BITS_PER_COORD + CHANNELS_IN - 1 downto 0) := (others => '0');
    signal event_fifo_read_i : std_logic;

    -- extra debug signals
    signal uut_main_state, uut_next_state, uut_last_state : main_state_et;
    signal uut_main_state_vec, uut_next_state_vec, uut_last_state_vec : std_logic_vector(2 downto 0);

    signal uut_timestep_pending : std_logic;
    signal uut_current_event : event_tensor_t;
    signal uut_event_valid : std_logic;

    -- ========================================= TIMING PROCEDURES =========================================
    -- Original waitf for compatibility  
    procedure waitf(n : in integer) is
    begin
        for i in 1 to n loop
            wait until rising_edge(clk);
        end loop;
    end procedure waitf;
    
    -- Wait for falling edges (better for checking registered outputs)
    procedure wait_falling_edges(n : in integer) is
    begin
        for i in 1 to n loop
            wait until falling_edge(clk);
        end loop;
    end procedure wait_falling_edges;
    
    -- Drive a signal and wait for it to settle
    procedure drive_and_settle(signal sig : out std_logic; value : std_logic; settle_cycles : integer := 1) is
    begin
        sig <= value;
        waitf(settle_cycles); -- Use rising edges for driving
    end procedure drive_and_settle;
    
    -- Check a signal at a stable time (after falling edge)
    procedure check_signal_stable(signal sig : in std_logic; expected : std_logic; error_msg : string) is
    begin
        wait until falling_edge(clk); -- Wait for stable time
        wait for 1 ns; -- Small delay for stability
        assert sig = expected
            report error_msg & " - Expected: " & std_logic'image(expected) & 
                   ", Got: " & std_logic'image(sig)
            severity failure;
    end procedure check_signal_stable;
    
    -- Check a state signal at stable time
    procedure check_state_stable(signal state_sig : in main_state_et; expected : main_state_et; error_msg : string) is
    begin
        wait until falling_edge(clk);
        wait for 1 ns;
        assert state_sig = expected
            report error_msg & " - Expected: " & main_state_et'image(expected) & 
                   ", Got: " & main_state_et'image(state_sig)
            severity failure;
    end procedure check_state_stable;

    -- FIXED: Check event tensor equality
    procedure check_event_tensor_stable(signal tensor_sig : in event_tensor_t; expected : event_tensor_t; error_msg : string) is
    begin
        wait until falling_edge(clk);
        wait for 1 ns;
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
        ebus <= tensor_to_bus(tensor, BITS_PER_COORD, CHANNELS_IN);
        waitf(settle_cycles);
    end procedure drive_event_tensor;

begin

    clk_process: process
    begin
        clk <= '1';
        wait for CLK_PERIOD/2;
        clk <= '0';
        wait for CLK_PERIOD/2;
    end process;

    uut: entity work.conv_pool
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
        debug_event_valid => uut_event_valid
    );

    main : process

        variable test_tensor : event_tensor_t;

    beginwaitf(10); -- Initial wait to ensure everything is stable

        -- test reset
        if true then
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

            -- Now disable reset and check state
            drive_and_settle(rst_o, '0', 1);
            -- should go to pause when enable is 0
            check_state_stable(
                uut_main_state, PAUSE, "uut_main_state should be PAUSE after reset and enable 0"
            );
                        
        end if;

        if true then
            -- Reset and enable 1
            drive_and_settle(rst_o, '1', 1);
            drive_and_settle(enable_o, '1', 1); -- Enable the module during reset
            drive_and_settle(timestep_o, '0', 1);
            drive_and_settle(event_fifo_empty_o, '1', 1); -- Ensure FIFO is empty
            check_state_stable(
                uut_main_state, RESET, "uut_main_state should be RESET after reset signal"
            );
            check_signal_stable(
                event_fifo_read_i, '0', "uut should not request fifo read when reset"
            );

            -- Now disable reset and check state
            drive_and_settle(rst_o, '0', 1);
            check_state_stable(
                uut_main_state, IDLE, "uut_main_state should be IDLE after reset and enable"
            );
        end if;

        -- Test fifo read request behavior
        if true then
            -- Let module go to IDLE
            drive_and_settle(rst_o, '0', 1);
            drive_and_settle(enable_o, '1', 1);
            drive_and_settle(timestep_o, '0', 1);
            drive_and_settle(event_fifo_empty_o, '1', 1);
            check_state_stable(
                uut_main_state, IDLE, "uut_main_state should be IDLE after reset and enable"
            );
            -- FIXED: Typo corrected
            check_signal_stable(
                uut_event_valid, '0', "uut should not have valid event in IDLE state"
            );
            -- FIXED: Use proper tensor checking procedure
            check_event_tensor_stable(
                uut_current_event, create_tensor(0, 0, 0),
                "uut should have current_event as zero tensor in IDLE state"
            );

            -- FIXED: Drive event tensor BEFORE making FIFO not empty
            test_tensor := create_tensor(x_coord => 10, y_coord => 10, channel => 0);
            drive_event_tensor(event_fifo_bus_o, test_tensor, 1);
            
            -- Now make FIFO not empty
            drive_and_settle(event_fifo_empty_o, '0', 1);
            check_signal_stable(
                event_fifo_read_i, '1', "uut should request fifo read when not empty"
            );
            
            -- FIXED: Check that state transitions to EVENT_CONV
            check_state_stable(
                uut_main_state, EVENT_CONV, "uut should transition to EVENT_CONV when processing event"
            );
            
            -- FIXED: Check that the event was captured correctly
            waitf(2); -- Give time for event to be captured
            check_signal_stable(
                uut_event_valid, '1', "uut should have valid event after reading from FIFO"
            );
            check_event_tensor_stable(
                uut_current_event, test_tensor,
                "uut should have captured the correct event from FIFO"
            );
        end if;wait for 100 ns;
        wait;

    end process main;

end testbench;