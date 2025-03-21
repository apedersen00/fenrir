library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_types.all;

entity parallel_conv is
    port (
        clk      : in std_logic;
        window   : in window_array_8_t;
        kernel   : in kernel_t;
        conv_out : out conv_out_array_t
    );
end entity parallel_conv;

architecture rtl of parallel_conv is
begin

    gen_convs: for i in 0 to 7 generate
        conv_inst: entity work.convolution_single
            generic map (
                conv_out_bit_width => CONV_OUT_BIT_WIDTH
            )
            port map (
                clk      => clk,
                window   => window(i),
                kernel   => kernel,
                conv_out => conv_out(i)
            );
    end generate;

end architecture rtl;
