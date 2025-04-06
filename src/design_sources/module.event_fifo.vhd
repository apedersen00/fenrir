library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_system.all;

entity aer_fifo is 
    generic(
        FIFO_DEPTH : integer := 3
    );
    port (
        clk : in std_logic;
        reset : in std_logic;
        write_enable : in std_logic;
        read_enable : in std_logic;
        data_in : in EVENT_R;
        data_out : out EVENT_R;
        flag_full : out std_logic;
        flag_empty : out std_logic
    );
end entity aer_fifo;

architecture rtl of aer_fifo is

    type memory_array is array (0 to FIFO_DEPTH - 1) of EVENT_R;
    signal memory : memory_array;
    signal write_pointer : integer range 0 to FIFO_DEPTH - 1 := 0;
    signal read_pointer  : integer range 0 to FIFO_DEPTH - 1 := 0;
    signal count         : integer range 0 to FIFO_DEPTH := 0;

begin

    process (clk, reset)
    begin
        if reset = '1' then 

            write_pointer <= 0;
            read_pointer  <= 0;
            count         <= 0;
            data_out.x <= (others => '0');
            data_out.y <= (others => '0');
            data_out.polarity <= (others => '0');
            data_out.timestamp <= (others => '0');

        elsif rising_edge(clk) then
            -- READ FIRST
            if read_enable = '1' and count > 0 then
                data_out <= memory(read_pointer);
                read_pointer <= (read_pointer + 1) mod FIFO_DEPTH;
                count <= count - 1;
            end if;

            -- WRITE NEXT
            if write_enable = '1' and count < FIFO_DEPTH then
                memory(write_pointer) <= data_in;
                write_pointer <= (write_pointer + 1) mod FIFO_DEPTH;
                count <= count + 1;
            end if;

            flag_full  <= '1' when count = FIFO_DEPTH else '0';
            flag_empty <= '1' when count = 0 else '0';
        end if;
    end process;

end architecture rtl;
