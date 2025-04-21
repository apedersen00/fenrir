library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

--use work.conv_control_t.all;

entity conv_event_gen is 
    port(
        clk : in std_logic;
        event_happened : in std_logic;
        events : in std_logic_vector(2 - 1 downto 0);
        x : in integer range 0 to 10 - 1;
        y : in integer range 0 to 10 - 1;
        timestamp : in std_logic_vector(4 - 1 downto 0)
    );
end entity conv_event_gen;

architecture behavior of conv_event_gen is 
begin
end architecture behavior;