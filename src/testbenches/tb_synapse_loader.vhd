library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use std.env.finish;

entity synapse_loader_tb is
end synapse_loader_tb;

architecture behavior of synapse_loader_tb is

    constant clk_period : time := 10 ns;
    constant DEPTH      : integer := 128;
    constant WIDTH      : integer := 32;

    signal clk              : std_logic := '0';

    signal fifo_we          : std_logic;
    signal fifo_wdata       : std_logic_vector(WIDTH - 1 downto 0);
    signal fifo_re          : std_logic;
    signal fifo_rvalid      : std_logic;
    signal fifo_rdata       : std_logic_vector(WIDTH - 1 downto 0);
    signal fifo_empty       : std_logic;
    signal fifo_empty_next  : std_logic;
    signal fifo_full        : std_logic;
    signal fifo_full_next   : std_logic;
    signal fifo_fill_count  : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal fifo_rst         : std_logic;
    signal fifo_fault       : std_logic;

    signal synldr_cfg_en    : std_logic;
    signal synldr_cfg_addr  : std_logic_vector(3 downto 0);
    signal synldr_cfg_val   : std_logic_vector(31 downto 0);
    signal synldr_weight    : std_logic_vector(7 downto 0);
    signal synldr_start     : std_logic;
    signal synldr_busy      : std_logic;
    signal synldr_rst       : std_logic;
    signal synldr_fault     : std_logic;

    signal syn_addr         : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal syn_data         : std_logic_vector(WIDTH - 1 downto 0);

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
            SHOTGUN_DEPTH   => 8,
            SYN_MEM_DEPTH   => 128,
            SYN_MEM_WIDTH   => 32
        )
        port map (
            i_cfg_en        => synldr_cfg_en,
            i_cfg_addr      => synldr_cfg_addr,
            i_cfg_val       => synldr_cfg_val,

            o_fifo_re       => fifo_re,
            i_fifo_rvalid   => fifo_rvalid,
            i_fifo_rdata    => fifo_rdata,

            o_syn_weight    => synldr_weight,

            o_syn_addr      => open,
            i_syn_data      => (others => '0'),

            i_start         => synldr_start,
            o_busy          => synldr_busy,
            i_clk           => clk,
            i_rst           => synldr_rst
        );

    clk <= not clk after clk_period / 2;

    PROC_SEQUENCER : process
    begin

        -- Reset FIFO
        fifo_rst    <= '1';
        fifo_we     <= '0';
        fifo_re     <= '0';
        fifo_wdata  <= (others => '0');
        wait for 10 * clk_period;
        finish;
    end process;

end behavior;
