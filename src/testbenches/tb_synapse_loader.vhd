library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use std.env.finish;

entity fifo_tb is
end fifo_tb;

architecture behavior of fifo_tb is

    constant clk_period : time := 10 ns;
    constant DEPTH      : integer := 128;
    constant WIDTH      : integer := 32;

    signal we           : std_logic := '0';
    signal wdata        : std_logic_vector(WIDTH - 1 downto 0) := (others => '0');
    signal re           : std_logic := '0';
    signal rvalid       : std_logic;
    signal rdata        : std_logic_vector(WIDTH - 1 downto 0);
    signal empty        : std_logic;
    signal empty_next   : std_logic;
    signal full         : std_logic;
    signal full_next    : std_logic;
    signal fill_count   : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal clk          : std_logic := '0';
    signal rst          : std_logic := '1';
    signal fault        : std_logic;

begin

    INPUT_FIFO : entity work.BRAM_FIFO
        generic map (
            DEPTH => DEPTH,
            WIDTH => WIDTH
        )
        port map (
            i_we            => we,
            i_wdata         => wdata,
            i_re            => re,
            o_rvalid        => rvalid,
            o_rdata         => rdata,
            o_empty         => empty,
            o_empty_next    => empty_next,
            o_full          => full,
            o_full_next     => full_next,
            o_fill_count    => fill_count,
            i_clk           => clk,
            i_rst           => rst,
            o_fault         => fault
        );

    SYN_LOADER : entity work.SYNAPSE_LOADER
        generic map (
            SHOTGUN_NUM_REG => 8,
            SYN_MEM_DEPTH   => 128,
            SYN_MEM_WIDTH   => 32
        )
        port map (
            i_we            => we,
            i_wdata         => wdata,
            i_re            => re,
            o_rvalid        => rvalid,
            o_rdata         => rdata,
            o_empty         => empty,
            o_empty_next    => empty_next,
            o_full          => full,
            o_full_next     => full_next,
            o_fill_count    => fill_count,
            i_clk           => clk,
            i_rst           => rst,
            o_fault         => fault
        );

    clk <= not clk after clk_period / 2;

    PROC_SEQUENCER : process
    begin

        -- Test 1: fill and empty FIFO
        rst     <= '1';
        wdata   <= (others => '0');
        we      <= '0';
        re      <= '0';
        wait for 10 * clk_period;
        rst <= '0';
        wait until rising_edge(clk);

        -- start writing
        we <= '1';

        -- fill the FIFO
        while full_next = '0' loop
            wdata <= std_logic_vector(unsigned(wdata) + 1);
            wait until rising_edge(clk);
        end loop;

        -- stop writing
        we <= '0';

        -- empty the FIFO
        re <= '1';
        wait until empty_next = '1';
        wait for 10 * clk_period;

        finish;
    end process;

end behavior;
