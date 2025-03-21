library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_types.all;
use work.test_utils.all;

package body test_utils is

    function make_window_from_array(values : window_int_array) return window_t is
        variable w : window_t;
    begin
        w.v00 := to_signed(values(0), 2); w.v01 := to_signed(values(1), 2); w.v02 := to_signed(values(2), 2);
        w.v10 := to_signed(values(3), 2); w.v11 := to_signed(values(4), 2); w.v12 := to_signed(values(5), 2);
        w.v20 := to_signed(values(6), 2); w.v21 := to_signed(values(7), 2); w.v22 := to_signed(values(8), 2);
        return w;
    end function;

    function make_kernel_from_array(values : kernel_int_array) return kernel_t is
        variable k : kernel_t;
    begin
        k.k00 := to_signed(values(0), DEFAULT_KERNEL_BIT_WIDTH);
        k.k01 := to_signed(values(1), DEFAULT_KERNEL_BIT_WIDTH);
        k.k02 := to_signed(values(2), DEFAULT_KERNEL_BIT_WIDTH);
        k.k10 := to_signed(values(3), DEFAULT_KERNEL_BIT_WIDTH);
        k.k11 := to_signed(values(4), DEFAULT_KERNEL_BIT_WIDTH);
        k.k12 := to_signed(values(5), DEFAULT_KERNEL_BIT_WIDTH);
        k.k20 := to_signed(values(6), DEFAULT_KERNEL_BIT_WIDTH);
        k.k21 := to_signed(values(7), DEFAULT_KERNEL_BIT_WIDTH);
        k.k22 := to_signed(values(8), DEFAULT_KERNEL_BIT_WIDTH);
        return k;
    end function;

end package body test_utils;
