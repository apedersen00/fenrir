library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_types.all;

package test_utils is

    constant KERNEL_MIN_VAL : integer := -(2**(DEFAULT_KERNEL_BIT_WIDTH-1));
    constant KERNEL_MAX_VAL : integer :=  (2**(DEFAULT_KERNEL_BIT_WIDTH-1)) - 1;

    type window_int_array is array (natural range <>) of integer range -1 to 1;
    type kernel_int_array is array (natural range <>) of integer range KERNEL_MIN_VAL to KERNEL_MAX_VAL;

    function make_window_from_array(values : window_int_array) return window_t;
    function make_kernel_from_array(values : kernel_int_array) return kernel_t;

end package test_utils;
