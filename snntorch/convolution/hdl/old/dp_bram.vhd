library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity TRUE_DUAL_PORT_READ_FIRST is
generic(
    RAM_DEPTH : integer := 1024; -- Depth of the RAM
    DATA_WIDTH : integer := 16; -- Width of the data bus
    ADDR_WIDTH : integer := 10; -- Width of the address bus
    FILE_NAME : string := "" -- FIXED: Default empty string
);
port(
    clka : in std_logic;
    clkb : in std_logic;
    ena : in std_logic;
    enb : in std_logic;
    wea : in std_logic;
    web : in std_logic;
    addra : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
    addrb : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
    dia : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    dib : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    doa : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    dob : out std_logic_vector(DATA_WIDTH - 1 downto 0)
);
end entity TRUE_DUAL_PORT_READ_FIRST;

architecture rtl of TRUE_DUAL_PORT_READ_FIRST is

    type ram_type is array (0 to RAM_DEPTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);

    -- FIXED: Corrected file initialization function
    impure function InitRamFromFile(RamFileName: in string) return ram_type is
        FILE RamFile : text;
        variable file_status: file_open_status;
        variable line_buffer: line;
        variable RAM : ram_type := (others => (others => '0')); -- Initialize to zeros
        variable v_data_temp : integer; -- Temporary for reading
        variable good_read : boolean;
    begin
        -- FIXED: Check if filename is not empty
        if RamFileName'length > 0 then
            file_open(file_status, RamFile, RamFileName, read_mode);
            if file_status = open_ok then
                for I in RAM'range loop
                    if not endfile(RamFile) then
                        readline(RamFile, line_buffer);
                        read(line_buffer, v_data_temp, good_read);
                        if good_read then
                            RAM(I) := std_logic_vector(to_unsigned(v_data_temp, DATA_WIDTH));
                        end if;
                    else
                        exit; -- Exit loop if end of file reached
                    end if;
                end loop;
                file_close(RamFile);
                report "Successfully loaded RAM from file: " & RamFileName;
            else
                report "Warning: Could not open RAM file: " & RamFileName & " - initializing to zeros" severity warning;
            end if;
        else
            report "No RAM file specified - initializing to zeros" severity note;
        end if;
        return RAM;
    end function InitRamFromFile;

    -- FIXED: Use signal instead of variable for shared memory
    signal ram : ram_type := InitRamFromFile(FILE_NAME);

begin

    -- FIXED: Separate processes for each port (cleaner for dual-port)
    port_a_process : process (clka)
    begin
        if rising_edge(clka) then
            if ena = '1' then
                doa <= ram(to_integer(unsigned(addra)));
                if wea = '1' then
                    ram(to_integer(unsigned(addra))) <= dia;
                end if;
            end if;
        end if;
    end process port_a_process;

    port_b_process : process (clkb)
    begin
        if rising_edge(clkb) then
            if enb = '1' then
                dob <= ram(to_integer(unsigned(addrb)));
                if web = '1' then
                    ram(to_integer(unsigned(addrb))) <= dib;
                end if;
            end if;
        end if;
    end process port_b_process;

end rtl;