library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_types.all;

entity convolution_single is 
generic(
    conv_out_bit_width : integer := 8
);
port(
    clk: in std_logic;
    window: in window_t;
    kernel: in kernel_t;
    conv_out: out signed(conv_out_bit_width-1 downto 0)
);
end entity convolution_single;

architecture Behavioral of convolution_single is

begin

    process(clk)
    begin
        if rising_edge(clk) then
            conv_out <= dot_product(window, kernel, conv_out_bit_width);
        end if;
    end process;
    
end architecture Behavioral;