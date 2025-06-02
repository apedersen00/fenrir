----------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
----------------------------------------------------------------------------------------------------
--
--  File: dual_port_bram.vhd
--  Description: Dual-port BRAM with one clock (IEEE Standard Compliant)
--  VHDL Version: VHDL-2008
--
--  Author(s):
--      - A. Pedersen, Aarhus University
--      - A. Cherencq, Aarhus University
--      - Modified for IEEE compliance
--
---------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use std.textio.all;

--  Instantiation Template:
--  INST_NAME : entity work.DUAL_PORT_BRAM
--  generic map (
--      DEPTH       =>
--      WIDTH       =>
--      FILENAME    =>
--  )
--  port map (
--      i_we        =>
--      i_waddr     =>
--      i_wdata     =>
--      i_re        =>
--      i_raddr     =>
--      o_rdata     =>
--      i_clk       =>
--  );

entity DUAL_PORT_BRAM is
    generic(
        DEPTH       : integer;      -- BRAM depth
        WIDTH       : integer;      -- BRAM width
        FILENAME    : string := ""
    );
    port(
        -- write interface
        i_we    : in std_logic;                                                     -- write enable
        i_waddr : in std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0); -- write address
        i_wdata : in std_logic_vector(WIDTH - 1 downto 0);                          -- write data
        
        -- read interface
        i_re    : in std_logic;                                                     -- read enable
        i_raddr : in std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0); -- read address
        o_rdata : out std_logic_vector(WIDTH - 1 downto 0);                         -- read data

        -- auxiliary
        i_clk   : in std_logic
    );
end DUAL_PORT_BRAM;

architecture syn of DUAL_PORT_BRAM is

    type ram_type is array (0 to DEPTH - 1) of std_logic_vector(WIDTH - 1 downto 0);

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
                report "Failed to open file: " & RamFileName severity warning;
            end if;
        else
            -- No filename provided, initialize with zeros (this is normal)
            RAM := (others => (others => '0'));
        end if;
        
        return RAM;
    end function;

    signal RAM : ram_type := InitRamFromFile(FILENAME);

    begin

    -- Dual port BRAM process
    bram_process : process(i_clk)
    begin
        if rising_edge(i_clk) then
            -- Write port
            if i_we = '1' then
                RAM(to_integer(unsigned(i_waddr))) <= i_wdata;
            end if;
            
            -- Read port
            if i_re = '1' then
                o_rdata <= RAM(to_integer(unsigned(i_raddr)));
            end if;
        end if;
    end process bram_process;

end syn;