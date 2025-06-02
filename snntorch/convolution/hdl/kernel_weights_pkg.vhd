----------------------------------------------------------------------------------------------------
--  Simple Working VHDL Package: kernel_weights_pkg  
--  Quick-start kernel weights for immediate testing
--  Configuration: 3x3 kernel, 4 channels, 9-bit weights
----------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package kernel_weights_pkg is

    -- Kernel configuration constants
    constant KERNEL_SIZE_CFG     : integer := 3;
    constant CHANNELS_OUT_CFG    : integer := 4;
    constant BITS_PER_WEIGHT_CFG : integer := 9;
    
    -- Kernel weight type definition
    type kernel_weights_t is array (0 to KERNEL_SIZE_CFG**2 - 1, 0 to CHANNELS_OUT_CFG - 1) 
         of signed(BITS_PER_WEIGHT_CFG - 1 downto 0);
    
    -- Simple edge detection weights for immediate testing
    constant KERNEL_WEIGHTS : kernel_weights_t := (
        0 => (to_signed(-1, BITS_PER_WEIGHT_CFG), to_signed(-2, BITS_PER_WEIGHT_CFG), to_signed(-1, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG)), -- Pos(0,0)
        1 => (to_signed(-2, BITS_PER_WEIGHT_CFG), to_signed(-4, BITS_PER_WEIGHT_CFG), to_signed(-2, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG)), -- Pos(0,1)
        2 => (to_signed(-1, BITS_PER_WEIGHT_CFG), to_signed(-2, BITS_PER_WEIGHT_CFG), to_signed(-1, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG)), -- Pos(0,2)
        3 => (to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG)), -- Pos(1,0)
        4 => (to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(16, BITS_PER_WEIGHT_CFG)), -- Pos(1,1)
        5 => (to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG)), -- Pos(1,2)
        6 => (to_signed(1, BITS_PER_WEIGHT_CFG), to_signed(2, BITS_PER_WEIGHT_CFG), to_signed(1, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG)), -- Pos(2,0)
        7 => (to_signed(2, BITS_PER_WEIGHT_CFG), to_signed(4, BITS_PER_WEIGHT_CFG), to_signed(2, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG)), -- Pos(2,1)
        8 => (to_signed(1, BITS_PER_WEIGHT_CFG), to_signed(2, BITS_PER_WEIGHT_CFG), to_signed(1, BITS_PER_WEIGHT_CFG), to_signed(0, BITS_PER_WEIGHT_CFG))  -- Pos(2,2)
    );
    
    -- Helper function to get weight value
    function get_kernel_weight(
        kernel_pos : integer;
        channel : integer
    ) return signed;
    
end package kernel_weights_pkg;

package body kernel_weights_pkg is

    function get_kernel_weight(
        kernel_pos : integer;
        channel : integer
    ) return signed is
    begin
        if kernel_pos >= 0 and kernel_pos < KERNEL_SIZE_CFG**2 and
           channel >= 0 and channel < CHANNELS_OUT_CFG then
            return KERNEL_WEIGHTS(kernel_pos, channel);
        else
            return to_signed(0, BITS_PER_WEIGHT_CFG);
        end if;
    end function;

end package body kernel_weights_pkg;