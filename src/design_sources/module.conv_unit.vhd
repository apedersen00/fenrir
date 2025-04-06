library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_system.all;

entity conv_unit is
    generic(
        NEURON_ADDRESS_WIDTH : integer := 128;
        x_coordinate_width : integer := 8;
        y_coordinate_width : integer := 8;
        timestamp_width : integer := 32
    );
    port(
        clk : in std_logic;
        neurons: in std_logic_vector(NEURON_ADDRESS_WIDTH - 1 downto 0);
        x : in std_logic_vector(x_coordinate_width - 1 downto 0);
        y : in std_logic_vector(y_coordinate_width - 1 downto 0);
        timestamp : in std_logic_vector(timestamp_width - 1 downto 0);
        polarity : in signed(1 downto 0)
    )
    