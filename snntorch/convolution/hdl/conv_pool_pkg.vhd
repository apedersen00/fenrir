library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package conv_pool_pkg is

    -- Define a type for 2D coordinates
    type vector2_t is record
        x : integer;
        y : integer;
    end record vector2_t;

end package conv_pool_pkg;