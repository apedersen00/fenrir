---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: ps_fifo_adapter.vhd
--  Description: Module for writing to a FIFO with the PS.
--
--  Author(s):
--      - A. Pedersen, Aarhus University
--      - A. Cherencq, Aarhus University
--
---------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

--  INST_NAME : entity work.FIFO_ADAPTER
--  generic map (
--      WIDTH   =>
--  )
--  port map (
--      i_clk           =>
--      i_rst           =>
--      i_ps_write      =>
--      o_fifo_we       =>
--      o_fifo_wdata    =>
--  );

entity FIFO_ADAPTER is
    generic (
        WIDTH   : integer
    );
    port (
        i_clk               : in std_logic;
        i_rst               : in std_logic;
        i_ps_write          : in std_logic_vector(31 downto 0);
        o_fifo_we           : out std_logic;
        o_fifo_wdata        : out std_logic_vector(WIDTH - 1 downto 0)
    );
end FIFO_ADAPTER;

architecture behavior of FIFO_ADAPTER is

    signal pendulum         : std_logic;
    signal last_pendulum    : std_logic;

begin

    pendulum <= i_ps_write(31);

    write_fifo : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                last_pendulum <= pendulum;
            elsif (pendulum /= last_pendulum) then
                o_fifo_we       <= '1';
                o_fifo_wdata    <= i_ps_write(WIDTH - 1 downto 0);
                last_pendulum   <= pendulum;
            else
                o_fifo_we <= '0';
            end if;
        end if;
    end process;

end behavior;
