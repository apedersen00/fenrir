library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity conv_control is
    generic(
        BITS_PER_NEURON : integer := 10;
        FEATURE_MAPS : integer := 10;
        TIMESTAMP_WIDTH: integer := 12;
        NEURON_RESET_WIDTH : integer := 4;
        NEURON_THRESHOLD_WIDTH : integer := 4;
        KERNEL_BIT_WIDTH : integer := 4;
        LEAKAGE_PARAM_WIDTH : integer := 4;
        TIME_SCALING_FACTOR : integer := 200;
        IMAGE_WIDTH : integer := 32;
        IMAGE_HEIGHT : integer := 32;
        BITS_PER_PIXEL_ADDRESS : integer := 8;
        KERNEL_SIZE : integer := 3
    );
    port(
        clk : in std_logic;
        reset : in std_logic;
        event_out : out std_logic_vector();
    )