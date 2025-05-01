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

    signal synmem_addr      : std_logic_vector(10 downto 0);
    signal synmem_dout      : std_logic_vector(31 downto 0);

    signal syn_addr         : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal syn_data         : std_logic_vector(WIDTH - 1 downto 0);

begin

    -- MNIST TEST
    -- 32x32 Input -> 10 output
    -- 32x32 = 1024 input neurons
    -- 1024x10 = 10240 synapses
    -- 10240/8 = 1280 addresses

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

    SYN_MEMORY : entity work.bram_mem
    generic map (
        DEPTH       => 1280,
        WIDTH       => 32,
        WIDTH_ADDR  => 11,
        FILENAME    => "data/syn_init.data"
    )
    port map (
        we      => '0',
        addr    => synmem_addr,
        din     => (others => '0'),
        dout    => synmem_dout,
        clk     => clk
    );

    SYN_LOADER : entity work.SYNAPSE_LOADER
        generic map (
            SYN_MEM_DEPTH   => 1280,
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
            o_syn_valid     => open,

            o_syn_addr      => synmem_addr,
            i_syn_data      => synmem_dout,

            i_start         => synldr_start,
            o_busy          => synldr_busy,
            i_clk           => clk,
            i_rst           => synldr_rst
        );

    clk <= not clk after clk_period / 2;

    PROC_SEQUENCER : process
    begin

        -- Reset Synapse Loader
        synldr_rst  <= '1';

        -- Reset FIFO
        fifo_rst    <= '1';
        fifo_we     <= '0';
        fifo_wdata  <= (others => '0');
        wait for 10 * clk_period;
        fifo_rst    <= '0';
        wait until rising_edge(clk);
        synldr_rst  <= '0';

        -- start writing
        fifo_we     <= '1';

        -- fill the FIFO
        while fifo_full_next = '0' loop
            fifo_wdata <= std_logic_vector(unsigned(fifo_wdata) + 1);
            wait until rising_edge(clk);
        end loop;

        -- configure synapse loader
        fifo_we         <= '0';
        synldr_cfg_en   <= '1';
        synldr_cfg_addr <= "0000";
        synldr_cfg_val  <=
            "00000000"                              &   -- zero padding
            std_logic_vector(to_unsigned(1, 2))     &   -- bits per weight
            std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
            std_logic_vector(to_unsigned(10, 11));     -- neurons per layer
        wait until rising_edge(clk);

        synldr_cfg_en   <= '0';
        synldr_start    <= '1';

        for i in 0 to 100 loop
            wait until rising_edge(clk);
        end loop;

        finish;
    end process;

end behavior;
