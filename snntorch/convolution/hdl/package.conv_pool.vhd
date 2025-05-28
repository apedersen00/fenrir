-- File: package.conv_pool.vhd
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Package header
package conv_pool_pkg is

    type main_state_et is (
        IDLE,
        EVENT_CONV,
        PAUSE,
        POOL,
        CONFIG,
        RESET
    );

    function state_to_slv(state : main_state_et) return std_logic_vector;

end package conv_pool_pkg;

-- Package body
package body conv_pool_pkg is

    function state_to_slv(state : main_state_et) return std_logic_vector is
    begin
        return std_logic_vector(to_unsigned(main_state_et'pos(state), 3));
    end function;

end package body conv_pool_pkg;