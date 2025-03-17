library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity mem_synapse is 
    generic (
        G_DEBUG                 : boolean   := false;
        G_DEBUG_COUNTER_INIT    : integer   := 0;
        DEPTH                   : natural   := 16384;
        WIDTH                   : natural   := 32;
        WIDTH_ADDR              : natural   := 14;
        FILENAME                : string := ""  -- Default to empty string
    );
    port (
        clk : in std_logic;
        we : in std_logic;
        addr : in std_logic_vector(WIDTH_ADDR - 1 downto 0);
        din : in std_logic_vector(WIDTH - 1 downto 0);
        dout : out std_logic_vector(WIDTH - 1 downto 0)
    );
end entity mem_synapse;

architecture behavioral of mem_synapse is

    type RamType is array(0 to DEPTH - 1) of std_logic_vector(WIDTH - 1 downto 0);
    signal debug_counter : integer := G_DEBUG_COUNTER_INIT;

    impure function InitRam(fName : in string) return RamType is
        FILE RamFile : text;
        variable file_status : file_open_status;
        variable RamFileLine : line;
        variable RAM : RamType := (others => (others => '0'));
        variable v_data : std_logic_vector(WIDTH - 1 downto 0);
    begin
        if fName'length > 0 then
            file_open(file_status, RamFile, fName, read_mode);
            
            if file_status = open_ok then
                for I in RamType'range loop
                    if not endfile(RamFile) then
                        readline(RamFile, RamFileLine);
                        read(RamFileLine, v_data);
                        RAM(I) := v_data;
                    end if;
                end loop;
                file_close(RamFile);
            else
                report "Failed to open file: " & fName severity warning;
            end if;
        else
            report "No filename provided, initializing RAM with zeros" severity note;
        end if;
        
        return RAM;
    end function;

    signal RAM : RamType := InitRam(FILENAME);

begin

    process (clk)
    begin
        if rising_edge(clk) then
            if we = '1' then
                RAM(to_integer(unsigned(addr))) <= din;
            end if;
            dout <= RAM(to_integer(unsigned(addr)));
        end if;
    end process;

    -- Debug process with local counter
    debug_proc: process (clk)
    begin
        if G_DEBUG and rising_edge(clk) then
            -- report DIN, DOUT, ADDR, WE, and the current counter value
            report "DIN: " & to_hstring(din) & " DOUT: " & to_hstring(dout) & 
                   " ADDR: " & to_hstring(addr) & " WE: " & std_logic'image(we) & 
                   " COUNTER: " & integer'image(debug_counter);
            debug_counter <= debug_counter + 1;
        end if;
    end process debug_proc;

end architecture behavioral;