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

    signal clk              : std_logic := '0';

    signal fifo_we          : std_logic := '0';
    signal fifo_wdata       : std_logic_vector(WIDTH - 1 downto 0) := (others => '0');
    signal fifo_re          : std_logic := '0';
    signal fifo_rvalid      : std_logic;
    signal fifo_rdata       : std_logic_vector(WIDTH - 1 downto 0);
    signal fifo_empty       : std_logic;
    signal fifo_empty_next  : std_logic;
    signal fifo_full        : std_logic;
    signal fifo_full_next   : std_logic;
    signal fifo_fill_count  : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal fifo_rst         : std_logic := '1';
    signal fifo_fault       : std_logic;

    signal synldr_pack_out  : std_logic_vector(WIDTH - 1 downto 0);
    signal synldr_halt      : std_logic := '0';
    signal synldr_start     : std_logic := '0';
    signal synldr_busy      : std_logic;
    signal synldr_rst       : std_logic := '1';
    signal synldr_fault     : std_logic;

    signal syn_addr         : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal syn_data         : std_logic_vector(WIDTH - 1 downto 0) := (others => '0');

begin

    INPUT_FIFO : entity work.BRAM_FIFO
        generic map (
            DEPTH => DEPTH,
            WIDTH => WIDTH
        )
        port map (
            i_we                => fifo_we,
            i_wdata             => fifo_wdata,
            i_re                => fifo_re,
            o_rvalid            => fifo_rvalid,
            o_rdata             => fifo_rdata,
            o_empty             => fifo_empty,
            o_empty_next        => fifo_empty_next,
            o_full              => fifo_full,
            o_full_next         => fifo_full_next,
            o_fill_count        => fifo_fill_count,
            i_clk               => clk,
            i_rst               => fifo_rst,
            o_fault             => fifo_fault
        );

    SYN_LOADER : entity work.SYNAPSE_LOADER
        generic map (
            SHOTGUN_NUM_REG => 8,
            SYN_MEM_DEPTH   => 128,
            SYN_MEM_WIDTH   => 32
        )
        port map (
            i_cfg_en            => '0',
            i_cfg_addr          => (others => '0'),
            i_cfg_val           => (others => '0'),
            o_fifo_re           => fifo_re,
            i_fifo_rvalid       => fifo_rvalid,
            i_fifo_rdata        => fifo_rdata,
            i_fifo_empty        => fifo_empty,
            i_fifo_empty_next   => fifo_empty_next,
            o_pack_out          => synldr_pack_out,
            i_pack_halt         => synldr_halt,
            o_syn_addr          => syn_addr,
            i_syn_data          => syn_data,
            i_start             => synldr_start,
            o_busy              => synldr_busy,
            i_clk               => clk,
            i_rst               => synldr_rst,
            o_fault             => synldr_fault
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
