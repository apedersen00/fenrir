---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: dual_port_bram.vhd
--  Description: Single-port BRAM with one clock.
--
--  Author(s):
--      - A. Pedersen, Aarhus University
--      - A. Cherencq, Aarhus University
--
---------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use std.textio.all;

--  Instantiation Template:
--  INST_NAME : entity work.SINGLE_PORT_BRAM
--  generic map (
--      DEPTH       =>
--      WIDTH       =>
--      FILENAME    =>
--  )
--  port map (
--      i_we        =>
--      i_addr      =>
--      i_data      =>
--      o_data      =>
--      i_clk       =>
--  );

entity SINGLE_PORT_BRAM is
    generic(
        DEPTH       : integer;    -- BRAM depth
        WIDTH       : integer;    -- BRAM width
        FILENAME    : string := ""
    );
    port(
        -- interface
        i_we    : in std_logic;                                                     -- write enable
        i_addr  : in std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0); -- address
        i_data  : in std_logic_vector(WIDTH - 1 downto 0);                          -- data in
        o_data  : out std_logic_vector(WIDTH - 1 downto 0);                         -- data out

        -- auxiliary
        i_clk   : in std_logic;
    );
end SINGLE_PORT_BRAM;

architecture syn of SINGLE_PORT_BRAM is

    type ram_type is array (DEPTH - 1 downto 0) of std_logic_vector(WIDTH - 1 downto 0);

    impure function InitRamFromFile(RamFileName : in string) return ram_type is
        FILE RamFile         : text;
        variable file_status : file_open_status;
        variable RamFileLine : line;
        variable RAM         : ram_type := (others => (others => '0'));
        variable v_data      : std_logic_vector(WIDTH - 1 downto 0);
    begin
        if RamFileName'length > 0 then
            file_open(file_status, RamFile, RamFileName, read_mode);
            
            if file_status = open_ok then
                for I in ram_type'range loop
                    if not endfile(RamFile) then
                        readline(RamFile, RamFileLine);
                        read(RamFileLine, v_data);
                        RAM(I) := v_data;
                    end if;
                end loop;
                file_close(RamFile);
            else
                report "Failed to open file: " & RamFileName severity error;
            end if;
        else
            report "No filename provided, initializing RAM with zeros" severity error;
        end if;
        
        return RAM;
    end function;

    signal RAM : ram_type := InitRamFromFile(FILENAME);

begin

    process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_we = '1' then
                RAM(to_integer(unsigned(i_addr))) <= i_data;
            end if;
            o_data <= RAM(to_integer(unsigned(i_addr)));
        end if;
    end process;

end syn;