library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_convolution_layer is
end entity tb_convolution_layer;

architecture testbench of tb_convolution_layer is

    CONSTANT CLK_PERIOD : time := 10 ns;

    CONSTANT XCoordinateWidth : integer := 8;
    CONSTANT YCoordinateWidth : integer := 8;
    CONSTANT TimeStampWidth : integer := 32;

    signal clk : std_logic := '1';
    signal reset_o : std_logic := '0';
    signal config_command_o : std_logic_vector(1 downto 0) := (others => '0');
    signal config_data_io : std_logic_vector(31 downto 0) := (others => '0');
    signal event_data_o : std_logic_vector(XCoordinateWidth + YCoordinateWidth + TimeStampWidth - 1 downto 0) := (others => '0');
    signal event_fifo_empty_no : std_logic := '1';
    signal event_fifo_read_i : std_logic;

begin
end architecture testbench;