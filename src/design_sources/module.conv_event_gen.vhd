library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package conv_event_gen_pkg is
    type event_record is record
        x : integer range 0 to 10 - 1;
        y : integer range 0 to 10 - 1;
        timestamp : std_logic_vector(4 - 1 downto 0);
        events : std_logic_vector(2 - 1 downto 0);
    end record;
end package conv_event_gen_pkg;
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.conv_event_gen_pkg.all;
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

    type event_buffer_arr is array (0 to 10) of event_record;
    signal event_buffer : event_buffer_arr;
    signal write_ptr : integer range 0 to 10 := 0;
    signal read_ptr : integer range 0 to 10 := 0;
    signal event_count : integer range 0 to 10 := 0;
    signal event_full : std_logic := '0';

begin

    process(clk, event_happened)
    begin
        if rising_edge(clk) then
            if event_happened = '1' then

                event_buffer(write_ptr).x <= x;
                event_buffer(write_ptr).y <= y;
                event_buffer(write_ptr).timestamp <= timestamp;
                event_buffer(write_ptr).events <= events;
                write_ptr <= (write_ptr + 1) mod 10;
                event_count <= event_count + 1;


            end if;
        end if;
    end process;

    process(clk, event_count)
    begin
        if rising_edge(clk) then
            if event_count > 10 then
                event_full <= '1';
            else
                event_full <= '0';
            end if;
        end if;
    end process;

end architecture behavior;