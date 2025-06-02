----------------------------------------------------------------------------------------------------
--  Template VHDL Package: kernel_weights_pkg
--  Template for kernel weight configuration packages
--  
--  This file shows the structure expected by the configurable convolution module.
--  Generate actual weight packages using generate_kernel_weights.py
--
--  Default configuration: 3x3 kernel, 4 channels, 9-bit weights
----------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package kernel_weights_pkg is

    -- Kernel configuration constants
    -- These MUST match the convolution module generics
    constant KERNEL_SIZE_CFG     : integer := 3;
    constant CHANNELS_OUT_CFG    : integer := 4;
    constant BITS_PER_WEIGHT_CFG : integer := 9;
    
    -- Kernel weight type definition
    type kernel_weights_t is array (0 to KERNEL_SIZE_CFG**2 - 1, 0 to CHANNELS_OUT_CFG - 1) 
         of signed(BITS_PER_WEIGHT_CFG - 1 downto 0);
    
    -- Default kernel weights (simple edge detection pattern)
    -- Position mapping for 3x3 kernel:
    --   0 1 2
    --   3 4 5  
    --   6 7 8
    constant KERNEL_WEIGHTS : kernel_weights_t := (
        0 => (  -1,   -2,   -1,    0), -- Pos(0,0) - Top-left
        1 => (  -2,   -4,   -2,    0), -- Pos(0,1) - Top-center  
        2 => (  -1,   -2,   -1,    0), -- Pos(0,2) - Top-right
        3 => (   0,    0,    0,    0), -- Pos(1,0) - Mid-left
        4 => (   0,    0,    0,   16), -- Pos(1,1) - Center (main response)
        5 => (   0,    0,    0,    0), -- Pos(1,2) - Mid-right
        6 => (   1,    2,    1,    0), -- Pos(2,0) - Bottom-left
        7 => (   2,    4,    2,    0), -- Pos(2,1) - Bottom-center
        8 => (   1,    2,    1,    0)  -- Pos(2,2) - Bottom-right
    );
    
    -- Alternative weight sets for different purposes
    -- Uncomment and modify KERNEL_WEIGHTS assignment to use these
    
    -- Sobel X edge detection
    constant SOBEL_X_WEIGHTS : kernel_weights_t := (
        0 => (  -1,   -1,   -1,   -1), -- Pos(0,0)
        1 => (   0,    0,    0,    0), -- Pos(0,1)
        2 => (   1,    1,    1,    1), -- Pos(0,2)
        3 => (  -2,   -2,   -2,   -2), -- Pos(1,0)
        4 => (   0,    0,    0,    0), -- Pos(1,1)
        5 => (   2,    2,    2,    2), -- Pos(1,2)
        6 => (  -1,   -1,   -1,   -1), -- Pos(2,0)
        7 => (   0,    0,    0,    0), -- Pos(2,1)
        8 => (   1,    1,    1,    1)  -- Pos(2,2)
    );
    
    -- Sobel Y edge detection
    constant SOBEL_Y_WEIGHTS : kernel_weights_t := (
        0 => (  -1,   -2,   -1,    0), -- Pos(0,0)
        1 => (  -2,   -4,   -2,    0), -- Pos(0,1)
        2 => (  -1,   -2,   -1,    0), -- Pos(0,2)
        3 => (   0,    0,    0,    0), -- Pos(1,0)
        4 => (   0,    0,    0,    0), -- Pos(1,1)
        5 => (   0,    0,    0,    0), -- Pos(1,2)
        6 => (   1,    2,    1,    0), -- Pos(2,0)
        7 => (   2,    4,    2,    0), -- Pos(2,1)
        8 => (   1,    2,    1,    0)  -- Pos(2,2)
    );
    
    -- Gaussian blur approximation
    constant GAUSSIAN_WEIGHTS : kernel_weights_t := (
        0 => (   1,    1,    1,    1), -- Pos(0,0)
        1 => (   2,    2,    2,    2), -- Pos(0,1)
        2 => (   1,    1,    1,    1), -- Pos(0,2)
        3 => (   2,    2,    2,    2), -- Pos(1,0)
        4 => (   4,    4,    4,    4), -- Pos(1,1)
        5 => (   2,    2,    2,    2), -- Pos(1,2)
        6 => (   1,    1,    1,    1), -- Pos(2,0)
        7 => (   2,    2,    2,    2), -- Pos(2,1)
        8 => (   1,    1,    1,    1)  -- Pos(2,2)
    );
    
    -- Identity/passthrough
    constant IDENTITY_WEIGHTS : kernel_weights_t := (
        0 => (   0,    0,    0,    0), -- Pos(0,0)
        1 => (   0,    0,    0,    0), -- Pos(0,1)
        2 => (   0,    0,    0,    0), -- Pos(0,2)
        3 => (   0,    0,    0,    0), -- Pos(1,0)
        4 => (   8,    8,    8,    8), -- Pos(1,1) - Center only
        5 => (   0,    0,    0,    0), -- Pos(1,2)
        6 => (   0,    0,    0,    0), -- Pos(2,0)
        7 => (   0,    0,    0,    0), -- Pos(2,1)
        8 => (   0,    0,    0,    0)  -- Pos(2,2)
    );
    
    -- Helper function to get weight value with bounds checking
    function get_kernel_weight(
        kernel_pos : integer;
        channel : integer
    ) return signed;
    
    -- Utility functions for weight analysis
    function get_max_weight return integer;
    function get_min_weight return integer;
    function get_weight_sum(channel : integer) return integer;
    
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
            -- Return zero for out-of-bounds access
            return to_signed(0, BITS_PER_WEIGHT_CFG);
        end if;
    end function;
    
    function get_max_weight return integer is
        variable max_val : integer := -2**(BITS_PER_WEIGHT_CFG-1);
    begin
        for pos in 0 to KERNEL_SIZE_CFG**2 - 1 loop
            for ch in 0 to CHANNELS_OUT_CFG - 1 loop
                if to_integer(KERNEL_WEIGHTS(pos, ch)) > max_val then
                    max_val := to_integer(KERNEL_WEIGHTS(pos, ch));
                end if;
            end loop;
        end loop;
        return max_val;
    end function;
    
    function get_min_weight return integer is
        variable min_val : integer := 2**(BITS_PER_WEIGHT_CFG-1) - 1;
    begin
        for pos in 0 to KERNEL_SIZE_CFG**2 - 1 loop
            for ch in 0 to CHANNELS_OUT_CFG - 1 loop
                if to_integer(KERNEL_WEIGHTS(pos, ch)) < min_val then
                    min_val := to_integer(KERNEL_WEIGHTS(pos, ch));
                end if;
            end loop;
        end loop;
        return min_val;
    end function;
    
    function get_weight_sum(channel : integer) return integer is
        variable sum_val : integer := 0;
    begin
        if channel >= 0 and channel < CHANNELS_OUT_CFG then
            for pos in 0 to KERNEL_SIZE_CFG**2 - 1 loop
                sum_val := sum_val + to_integer(KERNEL_WEIGHTS(pos, channel));
            end loop;
        end if;
        return sum_val;
    end function;

end package body kernel_weights_pkg;