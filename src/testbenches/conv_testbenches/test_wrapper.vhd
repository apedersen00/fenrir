library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity test_wrapper is
end entity;

architecture tb of test_wrapper is

    constant BITS_PER_COORDINATE : integer := 7;
    constant IN_CHANNELS : integer := 6;
    constant OUT_CHANNELS : integer := 12;
    constant INPUT_FIFO_EVENT_CAPACITY : integer := 4096;


    -- Clock and reset signals
    signal clk    : std_logic := '0';
    signal rst_n  : std_logic := '0';
    signal enable : std_logic := '1';  -- Assuming enable is always high for this test
    signal input_write_enable : std_logic := '1';
    signal input_fifo_full_next : std_logic;
    signal input_fifo_data  : std_logic_vector(2 * BITS_PER_COORDINATE + IN_CHANNELS downto 0) := (others => '0');
    signal output_fifo_data : std_logic_vector((BITS_PER_COORDINATE - 1) * 2 + OUT_CHANNELS downto 0);
    signal output_fifo_full_next : std_logic := '0';
    signal output_fifo_write_enable : std_logic;

    constant clk_period : time := 10 ns;
    
    function pack_event_data(
        ts     : std_logic;
        x      : std_logic_vector(BITS_PER_COORDINATE-1 downto 0);
        y      : std_logic_vector(BITS_PER_COORDINATE-1 downto 0);
        spikes : std_logic_vector(IN_CHANNELS-1 downto 0)
    ) return std_logic_vector is
        variable result : std_logic_vector(2 * BITS_PER_COORDINATE + IN_CHANNELS downto 0);
    begin
        -- Pack in order: [ts][x][y][spikes] (MSB to LSB)
        result := ts & x & y & spikes;
        return result;
    end function;

begin

    -- Direct instantiation without component declaration
    uut: entity work.conv2d_wrapper
        generic map (
            KERNEL_SIZE => 3,
            IN_CHANNELS => IN_CHANNELS,
            OUT_CHANNELS => OUT_CHANNELS,
            IMG_HEIGHT => 30,
            IMG_WIDTH => 40,

            BITS_PER_KERNEL_WEIGHT => 6,
            BITS_PER_NEURON => 9,
            INPUT_FIFO_EVENT_CAPACITY => INPUT_FIFO_EVENT_CAPACITY,
            BITS_PER_COORDINATE => BITS_PER_COORDINATE
        )
        port map (
            clk   => clk,
            rst_n => rst_n,
            enable => enable,

            input_write_enable => input_write_enable,
            input_fifo_full_next => input_fifo_full_next,
            input_fifo_data => input_fifo_data,
            
            output_fifo_data => output_fifo_data,
            output_fifo_full_next => output_fifo_full_next,
            output_fifo_write_enable => output_fifo_write_enable
        );

    -- Clock and reset processes same as before...
    -- Clock generation
    clk_process : process
    begin
        while true loop
            clk <= '0';
            wait for clk_period / 2;
            clk <= '1';
            wait for clk_period / 2;
        end loop;
        wait;
    end process;

    -- Reset process
    stim_proc: process
    begin
        rst_n <= '0';
        wait for 20 ns;
        rst_n <= '1';
        input_fifo_data <= pack_event_data(
            '0',
            std_logic_vector(to_unsigned(5, BITS_PER_COORDINATE)),
            std_logic_vector(to_unsigned(6, BITS_PER_COORDINATE)),
            "111111"
        );
        input_write_enable <= '1';
        -- Add further stimulus here if needed
        wait until rising_edge(clk);
        input_write_enable <= '0';
        
        wait for 200 ns;

        -- End simulation
        assert false report "Simulation Ended" severity note;
        wait;
    end process;

end architecture;