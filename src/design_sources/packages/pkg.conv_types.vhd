library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_float_types.all;

package conv_types is

    constant DEFAULT_KERNEL_BIT_WIDTH : integer := 3;
    constant CONV_OUT_BIT_WIDTH : integer := 8;
    
    subtype int2_t is signed(1 downto 0);
    subtype kernel_val_t is signed(DEFAULT_KERNEL_BIT_WIDTH-1 downto 0);

    type window_t is record
        v00, v01, v02 : int2_t;
        v10, v11, v12 : int2_t;
        v20, v21, v22 : int2_t;
    end record;

    type kernel_t is record
        k00, k01, k02 : kernel_val_t;
        k10, k11, k12 : kernel_val_t;
        k20, k21, k22 : kernel_val_t;
    end record;

    attribute use_dsp : string;
    attribute use_dsp of dot_product : function is "yes";

    function dot_product(
        w: window_t;
        k: kernel_t;
        result_width: integer
    ) return signed;


    type window_array_8_t is array(0 to 7) of window_t;
    type conv_out_array_t is array(0 to 7) of signed(CONV_OUT_BIT_WIDTH-1 downto 0);

end package conv_types;

package body conv_types is

    function dot_product(
        w: window_t;
        k: kernel_t;
        result_width: integer
    ) return signed is
        
        variable acc : signed(result_width-1 downto 0) := (others => '0');

    begin

            acc := 
                resize(w.v00 * k.k00, result_width) +
                resize(w.v01 * k.k01, result_width) +
                resize(w.v02 * k.k02, result_width) +
                resize(w.v10 * k.k10, result_width) +
                resize(w.v11 * k.k11, result_width) +
                resize(w.v12 * k.k12, result_width) +
                resize(w.v20 * k.k20, result_width) +
                resize(w.v21 * k.k21, result_width) +
                resize(w.v22 * k.k22, result_width);

        return acc;

    end function;

end package body;