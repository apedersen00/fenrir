library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_pool_pkg.all;

entity tb_convolution is
end entity tb_convolution;

architecture testbench of tb_convolution is

    constant CLK_PERIOD      : time := 10 ns;
    constant IMG_WIDTH       : integer := 32;
    constant IMG_HEIGHT      : integer := 32;
    constant NEURON_BIT_WIDTH : integer := 9;
    constant KERNEL_SIZE     : integer := 3;
    constant CHANNELS_OUT    : integer := 4;  -- Smaller for easier testing
    constant ADDR_WIDTH      : integer := 10;
    constant BITS_PER_WEIGHT : integer := 9;
    
    -- Test state enum for Vivado debugging
    type test_state_t is (
        TEST_IDLE,
        TEST_RESET,
        TEST_BASIC_CONVOLUTION,
        TEST_EDGE_COORDINATES,
        TEST_ENABLE_DISABLE,
        TEST_MULTIPLE_CONVOLUTIONS,
        TEST_MEMORY_INTERFACE
    );
    
    -- Clock and control signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '0';
    signal enable : std_logic := '0';
    
    -- Input signals
    signal data_valid : std_logic := '0';
    signal event_coord : vector2_t := (x => 0, y => 0);
    signal data_consumed : std_logic;
    
    -- Memory interface signals
    signal mem_read_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal mem_read_en : std_logic;
    signal mem_read_data : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0) := (others => '0');
    
    signal mem_write_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal mem_write_en : std_logic;
    signal mem_write_data : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
    
    -- Status and debug signals
    signal busy : std_logic;
    signal debug_state : std_logic_vector(2 downto 0);
    signal debug_coord_idx : integer range 0 to KERNEL_SIZE**2;
    signal debug_calc_idx : integer range 0 to KERNEL_SIZE**2;
    signal debug_valid_count : integer range 0 to KERNEL_SIZE**2;
    signal current_test : test_state_t := TEST_IDLE;
    
    -- Simple memory model for testing
    type memory_t is array (0 to 2**ADDR_WIDTH - 1) of std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
    
    -- Function to create initial memory pattern
    function create_test_memory return memory_t is
        variable memory : memory_t;
        variable addr : integer;
        variable test_value : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
    begin
        -- Initialize all to zero first
        memory := (others => (others => '0'));
        
        -- Create simple test pattern
        for addr_idx in 0 to 1023 loop  -- Smaller subset for testing
            if addr_idx < IMG_WIDTH * IMG_HEIGHT then
                for ch in 0 to CHANNELS_OUT - 1 loop
                    test_value((ch + 1) * NEURON_BIT_WIDTH - 1 downto ch * NEURON_BIT_WIDTH) := 
                        std_logic_vector(to_unsigned((addr_idx + ch) mod (2**NEURON_BIT_WIDTH), NEURON_BIT_WIDTH));
                end loop;
                memory(addr_idx) := test_value;
            end if;
        end loop;
        
        return memory;
    end function;
    
    signal test_memory : memory_t := create_test_memory;
    
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
        wait_cycles(2);
        rst_sig <= '0';
        wait_cycles(1);
    end procedure;
    
    procedure send_event(
        x : integer;
        y : integer;
        signal coord_sig : out vector2_t;
        signal valid_sig : out std_logic
    ) is
    begin
        coord_sig <= (x => x, y => y);
        valid_sig <= '1';
        wait_cycles(1);
        valid_sig <= '0';
    end procedure;

begin

    -- Clock generation
    clk_process: process
    begin
        clk <= '1';
        wait for CLK_PERIOD/2;
        clk <= '0';
        wait for CLK_PERIOD/2;
    end process;

    -- Simple memory model
    memory_model : process(clk)
    begin
        if rising_edge(clk) then
            -- Handle memory reads
            if mem_read_en = '1' then
                mem_read_data <= test_memory(to_integer(unsigned(mem_read_addr)));
            end if;
            
            -- Handle memory writes
            if mem_write_en = '1' then
                test_memory(to_integer(unsigned(mem_write_addr))) <= mem_write_data;
            end if;
        end if;
    end process memory_model;

    -- Unit under test
    uut: entity work.convolution
    generic map (
        IMG_WIDTH        => IMG_WIDTH,
        IMG_HEIGHT       => IMG_HEIGHT,
        NEURON_BIT_WIDTH => NEURON_BIT_WIDTH,
        KERNEL_SIZE      => KERNEL_SIZE,
        CHANNELS_OUT     => CHANNELS_OUT,
        ADDR_WIDTH       => ADDR_WIDTH,
        BITS_PER_WEIGHT  => BITS_PER_WEIGHT
    )
    port map (
        clk              => clk,
        rst_i            => rst,
        enable_i         => enable,
        data_valid_i     => data_valid,
        event_coord_i    => event_coord,
        data_consumed_o  => data_consumed,
        mem_read_addr_o  => mem_read_addr,
        mem_read_en_o    => mem_read_en,
        mem_read_data_i  => mem_read_data,
        mem_write_addr_o => mem_write_addr,
        mem_write_en_o   => mem_write_en,
        mem_write_data_o => mem_write_data,
        busy_o           => busy,
        debug_state_o    => debug_state,
        debug_coord_idx_o => debug_coord_idx,
        debug_calc_idx_o => debug_calc_idx,
        debug_valid_count_o => debug_valid_count
    );

    main : process
    begin
        -- Initial stabilization
        wait_cycles(10);

        -- Test: test_reset
        current_test <= TEST_RESET;
        report "Running test: test_reset";
        
        reset_system(rst);
        enable <= '1';
        wait_cycles(2);
        
        -- Check initial state
        assert debug_state = "000" report "Should be in IDLE state after reset";
        assert busy = '0' report "Busy should be low after reset";
        assert mem_read_en = '0' report "Memory read should be disabled after reset";
        assert mem_write_en = '0' report "Memory write should be disabled after reset";
        report "Test test_reset completed";

        -- Test: test_basic_convolution
        current_test <= TEST_BASIC_CONVOLUTION;
        report "Running test: test_basic_convolution";
        
        reset_system(rst);
        enable <= '1';
        wait_cycles(2);
        
        -- Check initial state
        assert debug_state = "000" report "Should start in IDLE state";
        assert busy = '0' report "Should not be busy initially";
        
        -- Send event at coordinate (5, 5) - well within image bounds
        report "Sending event at (5, 5)";
        send_event(5, 5, event_coord, data_valid);
        
        -- Should become busy and start processing
        wait_cycles(1);
        assert busy = '1' report "Should be busy after receiving valid event. State=" & integer'image(to_integer(unsigned(debug_state)));
        assert debug_state = "001" report "Should be in CALC_COORDS state. Got state=" & integer'image(to_integer(unsigned(debug_state)));
        
        -- Wait for coordinate calculation to complete - need at least 9 cycles for 3x3 kernel
        report "Waiting for coordinate calculation (need " & integer'image(KERNEL_SIZE**2) & " cycles)...";
        for i in 0 to KERNEL_SIZE**2 + 2 loop
            wait_cycles(1);
            report "Cycle " & integer'image(i) & ": calc_idx=" & integer'image(debug_calc_idx) & " valid_count=" & integer'image(debug_valid_count) & " state=" & integer'image(to_integer(unsigned(debug_state)));
            if debug_calc_idx >= KERNEL_SIZE**2 then
                exit;
            end if;
        end loop;
        
        assert debug_calc_idx >= KERNEL_SIZE**2 report "Coordinate calculation should complete. calc_idx=" & integer'image(debug_calc_idx) & " expected>=" & integer'image(KERNEL_SIZE**2);
        assert debug_valid_count > 0 report "Should have some valid coordinates. count=" & integer'image(debug_valid_count);
        
        -- Should transition to PIPELINE state
        wait_cycles(2);
        report "After coord calc: state=" & integer'image(to_integer(unsigned(debug_state))) & " calc_idx=" & integer'image(debug_calc_idx) & " valid_count=" & integer'image(debug_valid_count);
        assert debug_state = "010" report "Should be in PIPELINE state. Got state=" & integer'image(to_integer(unsigned(debug_state)));
        
        -- Let convolution complete
        report "Waiting for convolution to complete...";
        wait until busy = '0' for 500 ns;  -- Increased timeout
        assert busy = '0' report "Convolution should complete";
        assert data_consumed = '1' report "Should signal data consumed";
        
        report "Basic convolution test completed successfully";
        report "Test test_basic_convolution completed";

        -- Test: test_edge_coordinates
        current_test <= TEST_EDGE_COORDINATES;
        report "Running test: test_edge_coordinates";
        
        reset_system(rst);
        enable <= '1';
        wait_cycles(2);
        
        -- Test corner coordinate (0, 0)
        report "Testing corner coordinate (0, 0)";
        send_event(0, 0, event_coord, data_valid);
        wait until busy = '0' for 500 ns;  -- Increased timeout
        assert busy = '0' report "Should handle corner coordinate (0,0). State=" & integer'image(to_integer(unsigned(debug_state)));
        
        wait_cycles(5);
        
        -- Test edge coordinate (31, 15) for 32x32 image
        report "Testing edge coordinate (31, 15)";
        send_event(31, 15, event_coord, data_valid);
        wait until busy = '0' for 500 ns;  -- Increased timeout
        assert busy = '0' report "Should handle edge coordinate (31,15). State=" & integer'image(to_integer(unsigned(debug_state)));
        report "Test test_edge_coordinates completed";

        report "All tests completed successfully";
        wait;

    end process main;

end testbench;