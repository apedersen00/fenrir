library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_control_t.all;

entity conv_event_gen is 
    port(
        clk : in std_logic;
        event_happened : in std_logic;
        events : in std_logic_vector(FEATURE_MAPS - 1 downto 0);
        x : in std_logic_vector(RAW_EVENT_X_WIDTH - 1 downto 0);
        y : in std_logic_vector(RAW_EVENT_Y_WIDTH - 1 downto 0);
        timestamp : in std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0);
    )