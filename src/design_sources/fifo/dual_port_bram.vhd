/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: dual_port_bram.vhd
    Description: Dual-port BRAM with one clock.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.math_real.all;

entity DUAL_PORT_BRAM is
    generic(
        DEPTH : integer;    -- BRAM depth
        WIDTH : integer     -- BRAM width
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
        i_clk   : in std_logic;
    );
end DUAL_PORT_BRAM;

architecture syn of DUAL_PORT_BRAM is

    type ram_type is array (DEPTH - 1 downto 0) of std_logic_vector(WIDTH - 1 downto 0);
    shared variable RAM : ram_type;

    begin

    process(i_clk)
    begin
        if i_clk'event and i_clk = '1' then
            if i_we = '1' then
                RAM(conv_integer(i_waddr)) := i_wdata;
            end if;
        end if;
    end process;

    process(i_clk)
    begin
        if i_clk'event and i_clk = '1' then
            if i_re = '1' then
                o_rdata <= RAM(conv_integer(i_raddr));
            end if;
        end if;
    end process;

end syn;