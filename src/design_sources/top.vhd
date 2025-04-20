/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: fenrir.vhd
    Description: Top module of FENRIR.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity fenrir is
    port (
        -- control
        clk             : in std_logic;
    );
end fenrir;

architecture Structural of fenrir is

    -- write interface
    signal wen   : std_logic;
    signal waddr : std_logic_vector(9 downto 0);
    signal wdata : std_logic_vector(31 downto 0);
    
    -- read interface
    signal ren   : std_logic;
    signal raddr : std_logic_vector(9 downto 0);
    signal rdata : std_logic_vector(31 downto 0);

begin

    dual_port_mem: entity work.DUAL_PORT_BRAM
        generic map (
            DEPTH                   => 1024,
            WIDTH                   => 32
        )
        port map (
            i_wen   => wen,
            i_waddr => waddr,
            i_wdata => wdata,
            i_ren   => ren,
            i_raddr => raddr,
            o_rdata => rdata,
            i_clk   => clk
        );

    process(clk)
    begin
        if rising_edge(clk) then
            waddr <= std_logic_vector(to_unsigned(0, waddr'length));
            wdata <= X"DEADBEEF";
            ren   <= '1';
            raddr <= std_logic_vector(to_unsigned(0, raddr'length));
        end if;
    end process;

    wen <= '1';

end Structural;
