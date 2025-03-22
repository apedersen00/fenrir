library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.conv_types.all;

entity conv_controller is
    generic(
        IMG_WIDTH : integer := 32;
        IMG_HEIGHT : integer := 32
    );
    port(
        clk : in std_logic;
        rst : in std_logic;

        -- for the 8xConv block
        windows : out window_array_8_t;
        kernel : out kernel_t;
    );
end entity conv_controller;

architecture Behavioral of conv_controller is
begin
end architecture Behavioral;