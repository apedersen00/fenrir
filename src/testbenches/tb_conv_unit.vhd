library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_conv_unit is
end entity tb_conv_unit;

architecture testbench of tb_conv_unit is

    CONSTANT CLK_PERIOD : time := 10 ns;

    signal clk : std_logic := '1';
    signal enable : std_logic := '0';
    signal input_data : std_logic_vector(111 downto 0) := (others => '0');
    signal output_data : std_logic_vector(111 downto 0) := (others => '0');
    signal kernels : std_logic_vector(39 downto 0) := (others => '0');
    signal neuron_reset_value : std_logic_vector(3 downto 0) := (others => '0');
    signal neuron_threshold_value : std_logic_vector(3 downto 0) := (others => '1');
    signal leakage_param : std_logic_vector(3 downto 0) := (others => '0');
    signal timestamp_event : std_logic_vector(11 downto 0) := (others => '0');
    signal spike_events : std_logic_vector(9 downto 0) := (others => '0');
    signal event_happened_flag : std_logic := '0';

begin 

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.conv_unit
        port map(
            enable => enable,
            input_data => input_data,
            output_data => output_data,
            kernels => kernels,
            neuron_reset_value => neuron_reset_value,
            neuron_threshold_value => neuron_threshold_value,
            leakage_param => leakage_param,
            timestamp_event => timestamp_event,
            spike_events => spike_events,
            event_happened_flag => event_happened_flag
        );

    stimulus : process

    type array_kernels is array (0 to 8) of std_logic_vector(39 downto 0);
    variable kernel_values : array_kernels := (
        0 => x"123456789A",  -- 40 bits
        1 => x"123456789A",
        2 => x"123456789A",
        3 => x"123456789A",
        4 => x"123456789A",
        5 => x"123456789A",
        6 => x"123456789A",
        7 => x"123456789A",
        8 => x"123456789A"
);

    type array_pixel is array (0 to 8) of std_logic_vector(111 downto 0);
    variable pixel_values : array_pixel := (
        0 => x"0000000000000000000000000001",
        1 => x"0010000000000000000000000002",
        2 => x"0020000000000000000000000003",
        3 => x"0030000000000000000000000004",
        4 => x"0040000000000000000000000005",
        5 => x"0050000000000000000000000006",
        6 => x"0060000000000000000000000007",
        7 => x"0070000000000000000000000008",
        8 => x"0080000000000000000000000009"
    );

    
    begin

        wait for CLK_PERIOD * 10;
        enable <= '1';
        timestamp_event <= x"008";
        for i in 0 to 8 loop

            input_data <= pixel_values(i);
            kernels <= kernel_values(i);
            wait for CLK_PERIOD;

        end loop;
        enable <= '0';
        wait;

    end process;

end architecture testbench;

