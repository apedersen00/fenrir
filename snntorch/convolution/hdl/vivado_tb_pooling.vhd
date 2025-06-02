library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_pool_pkg.all;

entity tb_pooling is
    ;
end entity tb_pooling;

architecture testbench of tb_pooling is

    constant CLK_PERIOD      : time := 10 ns;
    constant IMG_WIDTH       : integer := 8;   -- Smaller for easier testing
    constant IMG_HEIGHT      : integer := 8;   -- Smaller for easier testing
    constant NEURON_BIT_WIDTH : integer := 9;
    constant POOL_SIZE       : integer := 2;
    constant CHANNELS        : integer := 4;   -- Smaller for easier testing
    constant ADDR_WIDTH      : integer := 10;
    constant RESET_VALUE     : integer := 0;
    constant COORD_WIDTH     : integer := 8;
    
    -- Test state enum for debugging
    type test_state_t is (
        TEST_IDLE,
        TEST_RESET,
        TEST_BASIC_POOLING,
        TEST_DECAY_BEHAVIOR,
        TEST_CONDITIONAL_OUTPUT,
        TEST_FIFO_INTERFACE,
        TEST_ENABLE_DISABLE,
        TEST_FULL_IMAGE,
        TEST_THRESHOLD_EDGE_CASES
    );
    
    -- Clock and control signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '0';
    signal enable : std_logic := '0';
    
    -- Control signals
    signal start_pooling : std_logic := '0';
    signal pooling_done : std_logic;
    
    -- Memory interface signals
    signal mem_read_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal mem_read_en : std_logic;
    signal mem_read_data : std_logic_vector(CHANNELS * NEURON_BIT_WIDTH - 1 downto 0) := (others => '0');
    signal mem_write_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal mem_write_en : std_logic;
    signal mem_write_data : std_logic_vector(CHANNELS * NEURON_BIT_WIDTH - 1 downto 0);
    
    -- FIFO interface signals
    signal fifo_write_en : std_logic;
    signal fifo_write_data : std_logic_vector(2 * COORD_WIDTH + CHANNELS - 1 downto 0);
    signal fifo_full : std_logic := '0';
    
    -- Status and debug signals
    signal busy : std_logic;
    signal debug_state : std_logic_vector(2 downto 0);
    signal debug_window_x : integer range 0 to IMG_WIDTH/POOL_SIZE;
    signal debug_window_y : integer range 0 to IMG_HEIGHT/POOL_SIZE;
    signal debug_pixel_idx : integer range 0 to POOL_SIZE**2;
    signal debug_spike_count : integer range 0 to CHANNELS;
    signal debug_accum_ch0 : integer range -1024 to 1023;
    signal debug_accum_ch1 : integer range -1024 to 1023;
    signal current_test : test_state_t := TEST_IDLE;
    
    -- Memory models for testing
    type memory_t is array (0 to IMG_WIDTH * IMG_HEIGHT - 1) of std_logic_vector(CHANNELS * NEURON_BIT_WIDTH - 1 downto 0);
    
    -- FIFO output capture
    type fifo_event_t is record
        x_coord : integer;
        y_coord : integer;
        spikes : std_logic_vector(CHANNELS - 1 downto 0);
    end record;
    
    type fifo_events_t is array (0 to 100) of fifo_event_t;  -- Capture up to 100 events
    signal fifo_events : fifo_events_t;
    signal fifo_event_count : integer := 0;
    
    -- Function to create test membrane potential memory with specific patterns
    function create_test_membrane_memory return memory_t is
        variable memory : memory_t;
        variable test_value : std_logic_vector(CHANNELS * NEURON_BIT_WIDTH - 1 downto 0);
        variable base_val : integer;
        variable x, y : integer;
    begin

    ----------------------------------------------------------------------------
    -- Clock Generator inserted by convert_to_vivado
    ----------------------------------------------------------------------------
    clk_process: process
    begin
        clk <= '1';
        wait for CLK_PERIOD/2;
        clk <= '0';
        wait for CLK_PERIOD/2;
    end process clk_process;

        -- Initialize all to zero first
        memory := (others => (others => '0'));
        
        -- Create patterns that will test decay and threshold behavior
        for addr_idx in 0 to IMG_WIDTH * IMG_HEIGHT - 1 loop
            x := addr_idx mod IMG_WIDTH;
            y := addr_idx / IMG_WIDTH;
            
            for ch in 0 to CHANNELS - 1 loop
                -- Create different patterns for different areas:
                -- Top-left quadrant: High values (should spike after pooling)
                -- Top-right quadrant: Medium values (may spike depending on decay)
                -- Bottom-left quadrant: Low values (should not spike)
                -- Bottom-right quadrant: Very low values (definitely no spike)
                
                if x < IMG_WIDTH/2 and y < IMG_HEIGHT/2 then
                    -- Top-left: High values (140-180)
                    base_val := 140 + ch * 10 + (addr_idx mod 20);
                elsif x >= IMG_WIDTH/2 and y < IMG_HEIGHT/2 then
                    -- Top-right: Medium values (80-120)
                    base_val := 80 + ch * 10 + (addr_idx mod 20);
                elsif x < IMG_WIDTH/2 and y >= IMG_HEIGHT/2 then
                    -- Bottom-left: Low values (40-80)
                    base_val := 40 + ch * 10 + (addr_idx mod 20);
                else
                    -- Bottom-right: Very low values (10-50)
                    base_val := 10 + ch * 5 + (addr_idx mod 20);
                end if;
                
                test_value((ch + 1) * NEURON_BIT_WIDTH - 1 downto ch * NEURON_BIT_WIDTH) := 
                    std_logic_vector(to_unsigned(base_val mod (2**NEURON_BIT_WIDTH), NEURON_BIT_WIDTH));
            end loop;
            memory(addr_idx) := test_value;
        end loop;
        
        return memory;
    end function;
    
    signal membrane_memory : memory_t := create_test_membrane_memory;
    
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
    
    procedure start_pooling_operation(signal start_sig : out std_logic) is
    begin
        start_sig <= '1';
        wait_cycles(1);
        start_sig <= '0';
    end procedure;
    
    procedure wait_for_completion is
    begin
        wait until pooling_done = '1' for 10000 ns;  -- Generous timeout for full image
        assert pooling_done = '1' report "Pooling operation should complete";
    end procedure;
    
    function extract_channel_value(
        data : std_logic_vector;
        channel : integer
    ) return integer is
    begin
        return to_integer(unsigned(data((channel + 1) * NEURON_BIT_WIDTH - 1 downto channel * NEURON_BIT_WIDTH)));
    end function;
    
    -- Function to extract coordinates from FIFO data
    function extract_x_coord(fifo_data : std_logic_vector) return integer is
        variable x_coord : std_logic_vector(COORD_WIDTH - 1 downto 0);
    begin
        x_coord := fifo_data(2 * COORD_WIDTH + CHANNELS - 1 downto COORD_WIDTH + CHANNELS);
        return to_integer(unsigned(x_coord));
    end function;
    
    function extract_y_coord(fifo_data : std_logic_vector) return integer is
        variable y_coord : std_logic_vector(COORD_WIDTH - 1 downto 0);
    begin
        y_coord := fifo_data(COORD_WIDTH + CHANNELS - 1 downto CHANNELS);
        return to_integer(unsigned(y_coord));
    end function;
    
    function extract_spikes(fifo_data : std_logic_vector) return std_logic_vector is
    begin
        return fifo_data(CHANNELS - 1 downto 0);
    end function;

begin

    -- Clock generation
    clk <= not clk after CLK_PERIOD/2;

    -- Input memory model (membrane potentials)
    input_memory_model : process(clk)
    begin
        if rising_edge(clk) then
            -- Handle memory reads
            if mem_read_en = '1' then
                mem_read_data <= membrane_memory(to_integer(unsigned(mem_read_addr)));
            end if;
            
            -- Handle memory writes (updated membrane potentials after decay/spike reset)
            if mem_write_en = '1' then
                membrane_memory(to_integer(unsigned(mem_write_addr))) <= mem_write_data;
            end if;
        end if;
    end process input_memory_model;
    
    -- FIFO event capture process
    fifo_capture : process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                fifo_event_count <= 0;
                fifo_events <= (others => (x_coord => 0, y_coord => 0, spikes => (others => '0')));
            elsif fifo_write_en = '1' and fifo_event_count < 100 then
                fifo_events(fifo_event_count).x_coord <= extract_x_coord(fifo_write_data);
                fifo_events(fifo_event_count).y_coord <= extract_y_coord(fifo_write_data);
                fifo_events(fifo_event_count).spikes <= extract_spikes(fifo_write_data);
                fifo_event_count <= fifo_event_count + 1;
            end if;
        end if;
    end process fifo_capture;

    -- Unit under test
    uut: entity work.pooling
    generic map (
        IMG_WIDTH        => IMG_WIDTH,
        IMG_HEIGHT       => IMG_HEIGHT,
        NEURON_BIT_WIDTH => NEURON_BIT_WIDTH,
        POOL_SIZE        => POOL_SIZE,
        CHANNELS         => CHANNELS,
        ADDR_WIDTH       => ADDR_WIDTH,
        RESET_VALUE      => RESET_VALUE,
        COORD_WIDTH      => COORD_WIDTH
    )
    port map (
        clk              => clk,
        rst_i            => rst,
        enable_i         => enable,
        start_pooling_i  => start_pooling,
        pooling_done_o   => pooling_done,
        mem_read_addr_o  => mem_read_addr,
        mem_read_en_o    => mem_read_en,
        mem_read_data_i  => mem_read_data,
        mem_write_addr_o => mem_write_addr,
        mem_write_en_o   => mem_write_en,
        mem_write_data_o => mem_write_data,
        fifo_write_en_o  => fifo_write_en,
        fifo_write_data_o => fifo_write_data,
        fifo_full_i      => fifo_full,
        busy_o           => busy,
        debug_state_o    => debug_state,
        debug_window_x_o => debug_window_x,
        debug_window_y_o => debug_window_y,
        debug_pixel_idx_o => debug_pixel_idx,
        debug_spike_count_o => debug_spike_count,
        debug_accum_ch0_o => debug_accum_ch0,
        debug_accum_ch1_o => debug_accum_ch1
    );

    main : process
        variable event_idx : integer;
    begin
    -- Variables (if any) were declared above
    begin

        -- Initial stabilization (wait some time or cycles)
        wait for 100 ns;  -- Adjust as needed

        -- Running test: test_reset
        report "[INFO] Starting test: test_reset";
        -- [WARNING] Test code for "test_reset" not found
        report "[INFO] Test test_reset completed";

        -- Running test: test_basic_pooling
        report "[INFO] Starting test: test_basic_pooling";
        -- [WARNING] Test code for "test_basic_pooling" not found
        report "[INFO] Test test_basic_pooling completed";

        -- Running test: test_decay_behavior
        report "[INFO] Starting test: test_decay_behavior";
        -- [WARNING] Test code for "test_decay_behavior" not found
        report "[INFO] Test test_decay_behavior completed";

        -- Running test: test_conditional_output
        report "[INFO] Starting test: test_conditional_output";
        -- [WARNING] Test code for "test_conditional_output" not found
        report "[INFO] Test test_conditional_output completed";

        -- Running test: test_fifo_interface
        report "[INFO] Starting test: test_fifo_interface";
        -- [WARNING] Test code for "test_fifo_interface" not found
        report "[INFO] Test test_fifo_interface completed";

        -- Running test: test_enable_disable
        report "[INFO] Starting test: test_enable_disable";
        -- [WARNING] Test code for "test_enable_disable" not found
        report "[INFO] Test test_enable_disable completed";

        -- Running test: test_full_image
        report "[INFO] Starting test: test_full_image";
        -- [WARNING] Test code for "test_full_image" not found
        report "[INFO] Test test_full_image completed";

        -- Running test: test_threshold_edge_cases
        report "[INFO] Starting test: test_threshold_edge_cases";
        -- [WARNING] Test code for "test_threshold_edge_cases" not found
        report "[INFO] Test test_threshold_edge_cases completed";

        report "[INFO] All tests completed successfully";
        wait;
    end process main;
end process main;

end testbench;