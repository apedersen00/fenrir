library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.fixed_float_types.all;

package conv_types is

    constant DEFAULT_KERNEL_BIT_WIDTH : integer := 7;
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

    function kernel_ram_to_kernel_t(
        kernel_data : std_logic_vector(63 downto 0)
    ) return kernel_t;

    function kernel_t_to_kernel_ram(
        k: kernel_t
    ) return std_logic_vector;

    function dot_product(
        w: window_t;
        k: kernel_t;
        result_width: integer
    ) return signed;


    type window_array_8_t is array(0 to 7) of window_t;
    type conv_out_array_t is array(0 to 7) of signed(CONV_OUT_BIT_WIDTH-1 downto 0);

end package conv_types;

package body conv_types is

    function kernel_ram_to_kernel_t(
        kernel_data : std_logic_vector(63 downto 0)
    ) return kernel_t is

        variable k : kernel_t;

    begin

        k.k00 := signed(kernel_data(DEFAULT_KERNEL_BIT_WIDTH - 1 downto 0));
        k.k01 := signed(kernel_data(2*DEFAULT_KERNEL_BIT_WIDTH - 1 downto DEFAULT_KERNEL_BIT_WIDTH));
        k.k02 := signed(kernel_data(3*DEFAULT_KERNEL_BIT_WIDTH - 1 downto 2*DEFAULT_KERNEL_BIT_WIDTH));
        k.k10 := signed(kernel_data(4*DEFAULT_KERNEL_BIT_WIDTH - 1 downto 3*DEFAULT_KERNEL_BIT_WIDTH));
        k.k11 := signed(kernel_data(5*DEFAULT_KERNEL_BIT_WIDTH - 1 downto 4*DEFAULT_KERNEL_BIT_WIDTH));
        k.k12 := signed(kernel_data(6*DEFAULT_KERNEL_BIT_WIDTH - 1 downto 5*DEFAULT_KERNEL_BIT_WIDTH));
        k.k20 := signed(kernel_data(7*DEFAULT_KERNEL_BIT_WIDTH - 1 downto 6*DEFAULT_KERNEL_BIT_WIDTH));
        k.k21 := signed(kernel_data(8*DEFAULT_KERNEL_BIT_WIDTH - 1 downto 7*DEFAULT_KERNEL_BIT_WIDTH));
        k.k22 := signed(kernel_data(9*DEFAULT_KERNEL_BIT_WIDTH - 1 downto 8*DEFAULT_KERNEL_BIT_WIDTH));

        return k;
    end function;

    function kernel_t_to_kernel_ram(
        k: kernel_t
    ) return std_logic_vector is

        variable kernel_data : std_logic_vector(63 downto 0);
    begin

        kernel_data :=  std_logic_vector(
                            resize(k.k00, DEFAULT_KERNEL_BIT_WIDTH) &
                            resize(k.k01, DEFAULT_KERNEL_BIT_WIDTH) &
                            resize(k.k02, DEFAULT_KERNEL_BIT_WIDTH) &
                            resize(k.k10, DEFAULT_KERNEL_BIT_WIDTH) &
                            resize(k.k11, DEFAULT_KERNEL_BIT_WIDTH) &
                            resize(k.k12, DEFAULT_KERNEL_BIT_WIDTH) &
                            resize(k.k20, DEFAULT_KERNEL_BIT_WIDTH) &
                            resize(k.k21, DEFAULT_KERNEL_BIT_WIDTH) &
                            resize(k.k22, DEFAULT_KERNEL_BIT_WIDTH)
                        );

        return kernel_data;

    end function;

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