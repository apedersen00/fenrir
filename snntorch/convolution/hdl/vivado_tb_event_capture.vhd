library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.conv_pool_pkg.all;

entity tb_event_capture is
end entity tb_event_capture;

architecture testbench of tb_event_capture is

    constant CLK_PERIOD : time := 10 ns;
    constant BITS_PER_COORD : integer := 8;
    constant IMG_WIDTH : integer := 32;
    constant IMG_HEIGHT : integer := 32;
    
    -- Clock and control signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '0';
    signal enable : std_logic := '0';
    
    -- FIFO interface signals
    signal fifo_empty : std_logic := '1';
    signal fifo_bus : std_logic_vector(2 * BITS_PER_COORD - 1 downto 0) := (others => '0');
    signal fifo_read : std_logic;
    
    -- Data output signals
    signal data_valid : std_logic;
    signal data_out : vector2_t;
    signal data_consumed : std_logic := '0';
    
    -- Debug signals
    signal debug_state : std_logic_vector(1 downto 0);

    -- Simple helper procedures
    procedure wait_cycles(n : integer) is
    begin
        for i in 1 to n loop
            wait until rising_edge(clk);
        end loop;
    end procedure;
    
    procedure reset_system(signal rst_sig : out std_logic) is
    begin
        rst_sig <= '1';
        wait_cycles(2);
        rst_sig <= '0';
        wait_cycles(1);
    end procedure;
    
    procedure send_coordinates(
        x : integer; 
        y : integer;
        signal fifo_bus_sig : out std_logic_vector(2 * BITS_PER_COORD - 1 downto 0);
        signal fifo_empty_sig : out std_logic
    ) is
        variable coord_bus : std_logic_vector(2 * BITS_PER_COORD - 1 downto 0);
    begin
        -- Pack coordinates: [x_coord(MSBs)][y_coord(LSBs)]
        coord_bus(BITS_PER_COORD - 1 downto 0) := std_logic_vector(to_unsigned(y, BITS_PER_COORD));
        coord_bus(2 * BITS_PER_COORD - 1 downto BITS_PER_COORD) := std_logic_vector(to_unsigned(x, BITS_PER_COORD));
        
        fifo_bus_sig <= coord_bus;
        fifo_empty_sig <= '0';
        wait_cycles(1);
    end procedure;
    
    procedure clear_fifo(
        signal fifo_empty_sig : out std_logic;
        signal fifo_bus_sig : out std_logic_vector(2 * BITS_PER_COORD - 1 downto 0)
    ) is
    begin
        fifo_empty_sig <= '1';
        fifo_bus_sig <= (others => '0');
    end procedure;
    
    procedure consume_data(signal data_consumed_sig : out std_logic) is
    begin
        data_consumed_sig <= '1';
        wait_cycles(1);
        data_consumed_sig <= '0';
    end procedure;

begin

    -- Clock generation
    clk <= not clk after CLK_PERIOD/2;

    -- Unit under test
    uut: entity work.event_capture
    generic map (
        BITS_PER_COORD => BITS_PER_COORD,
        IMG_WIDTH => IMG_WIDTH,
        IMG_HEIGHT => IMG_HEIGHT
    )
    port map (
        clk => clk,
        rst_i => rst,
        enable_i => enable,
        fifo_empty_i => fifo_empty,
        fifo_bus_i => fifo_bus,
        fifo_read_o => fifo_read,
        data_valid_o => data_valid,
        data_out_o => data_out,
        data_consumed_i => data_consumed,
        debug_state_o => debug_state
    );

    main : process
    begin

        -- Initial stabilization
        wait_cycles(10);

        -- Test: test_reset
        report "Running test: test_reset";
        report "Testing reset functionality";
            
            -- Reset the system
            reset_system(rst);
            enable <= '1';  -- Need enable high to exit RESET state
            wait_cycles(1);
            
            -- Check initial state
            assert debug_state = "00" report "Should be in IDLE state after reset";
            assert data_valid = '0' report "Data valid should be low after reset";
            assert fifo_read = '0' report "FIFO read should be low after reset";
        report "Test test_reset completed";

        -- Test: test_basic_capture
        report "Running test: test_basic_capture";
        report "Testing basic coordinate capture";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- Send coordinates (10, 15)
            send_coordinates(10, 15, fifo_bus, fifo_empty);
            
            -- Should enter READ_REQUEST state
            wait_cycles(1);
            assert debug_state = "01" report "Should be in READ_REQUEST state";
            assert fifo_read = '1' report "FIFO read should be high";
            
            -- Should move to DATA_READY state
            wait_cycles(1);
            assert debug_state = "10" report "Should be in DATA_READY state";
            assert fifo_read = '0' report "FIFO read should be low";
            assert data_valid = '1' report "Data should be valid";
            assert data_out.x = 10 report "X coordinate should be 10";
            assert data_out.y = 15 report "Y coordinate should be 15";
            
            -- Consume the data
            consume_data(data_consumed);
            wait_cycles(1);
            
            -- Should return to IDLE
            assert debug_state = "00" report "Should return to IDLE state";
            assert data_valid = '0' report "Data valid should be low";
            
            clear_fifo(fifo_empty, fifo_bus);
        report "Test test_basic_capture completed";

        -- Test: test_enable_disable
        report "Running test: test_enable_disable";
        report "Testing enable/disable functionality";
            
            reset_system(rst);
            enable <= '0';  -- Start disabled
            wait_cycles(2);
            
            -- Send data while disabled
            send_coordinates(5, 7, fifo_bus, fifo_empty);
            wait_cycles(3);
            
            -- Should stay in IDLE
            assert debug_state = "00" report "Should stay in IDLE when disabled";
            assert fifo_read = '0' report "Should not read FIFO when disabled";
            
            -- Clear FIFO before enabling to test proper enable behavior
            clear_fifo(fifo_empty, fifo_bus);
            wait_cycles(2);
            
            -- Enable and check if it stays in IDLE (no data available)
            enable <= '1';
            wait_cycles(2);
            
            assert debug_state = "00" report "Should be in IDLE when enabled with empty FIFO";
            
            -- Now send data and it should capture
            send_coordinates(5, 7, fifo_bus, fifo_empty);
            wait_cycles(1);
            assert debug_state = "01" report "Should enter READ_REQUEST when data available";
            
            clear_fifo(fifo_empty, fifo_bus);
        report "Test test_enable_disable completed";

        -- Test: test_multiple_captures
        report "Running test: test_multiple_captures";
        report "Testing multiple coordinate captures";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- First capture
            send_coordinates(1, 2, fifo_bus, fifo_empty);
            wait_cycles(2);
            report "First capture: x=" & integer'image(data_out.x) & " y=" & integer'image(data_out.y) & " state=" & integer'image(to_integer(unsigned(debug_state)));
            assert data_out.x = 1 and data_out.y = 2 report "First capture failed";
            consume_data(data_consumed);
            clear_fifo(fifo_empty, fifo_bus);
            wait_cycles(3);  -- Extra wait
            
            -- Verify we're back in IDLE
            report "After first capture: state=" & integer'image(to_integer(unsigned(debug_state)));
            assert debug_state = "00" report "Should be back in IDLE after first capture";
            
            -- Second capture  
            send_coordinates(20, 25, fifo_bus, fifo_empty);
            wait_cycles(2);
            report "Second capture: x=" & integer'image(data_out.x) & " y=" & integer'image(data_out.y) & " state=" & integer'image(to_integer(unsigned(debug_state)));
            assert data_out.x = 20 and data_out.y = 25 report "Second capture failed";
            consume_data(data_consumed);
            clear_fifo(fifo_empty, fifo_bus);
        report "Test test_multiple_captures completed";

        -- Test: test_boundary_coordinates
        report "Running test: test_boundary_coordinates";
        report "Testing boundary coordinate values";
            
            reset_system(rst);
            enable <= '1';
            wait_cycles(2);
            
            -- Test maximum coordinates
            send_coordinates(255, 255, fifo_bus, fifo_empty);  -- Max for 8-bit
            wait_cycles(2);
            assert data_out.x = 255 and data_out.y = 255 report "Max coordinates failed";
            consume_data(data_consumed);
            clear_fifo(fifo_empty, fifo_bus);
            wait_cycles(2);
            
            -- Test zero coordinates
            send_coordinates(0, 0, fifo_bus, fifo_empty);
            wait_cycles(2);
            assert data_out.x = 0 and data_out.y = 0 report "Zero coordinates failed";
            consume_data(data_consumed);
            clear_fifo(fifo_empty, fifo_bus);
        report "Test test_boundary_coordinates completed";

        report "All tests completed successfully";
        wait;

    end process main;

end testbench;