library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_system.all;

entity address_calculator is
    port(
        clk : in std_logic;
        AER_In : EVENT_R;
        
    )