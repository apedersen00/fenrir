---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: tb_fully_connected.vhd
--  Description: Testbench for the fully-connected part of FENRIR.
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
use std.env.finish;

entity TB_FULLY_CONNECTED is
end TB_FULLY_CONNECTED;

architecture behavior of TB_FULLY_CONNECTED is

    constant clk_period : time := 10 ns;
    constant DEPTH      : integer := 128;
    constant WIDTH      : integer := 32;

    signal clk              : std_logic := '0';

    -- input event fifo
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

    -- synapse loader
    signal synldr_cfg_en        : std_logic;
    signal synldr_cfg_addr      : std_logic_vector(3 downto 0);
    signal synldr_cfg_val       : std_logic_vector(31 downto 0);
    signal synldr_weight        : std_logic_vector(7 downto 0);
    signal synldr_valid         : std_logic;
    signal synldr_valid_next    : std_logic;
    signal synldr_valid_last    : std_logic;
    signal synldr_start         : std_logic;
    signal synldr_busy          : std_logic;
    signal synldr_rst           : std_logic;
    signal synldr_fault         : std_logic;

    -- neuron loader
    signal nrnldr_cfg_en        : std_logic;
    signal nrnldr_cfg_addr      : std_logic_vector(3 downto 0);
    signal nrnldr_cfg_val       : std_logic_vector(31 downto 0);
    signal nrnldr_re            : std_logic;
    signal nrnldr_data          : std_logic_vector(35 downto 0);
    signal nrnldr_state         : std_logic_vector(11 downto 0);
    signal nrnldr_nrn_index     : std_logic_vector(11 downto 0);
    signal nrnldr_valid         : std_logic;
    signal nrnldr_valid_next    : std_logic;
    signal nnrldr_valid_last    : std_logic;
    signal nrnldr_start         : std_logic;
    signal nrnldr_busy          : std_logic;
    signal nrnldr_rst           : std_logic;

    -- lif
    signal lif_cfg_en            : std_logic;
    signal lif_cfg_addr          : std_logic_vector(3 downto 0);
    signal lif_cfg_val           : std_logic_vector(31 downto 0);
    signal lif_nrn_state         : std_logic_vector(11 downto 0);
    signal lif_syn_weight        : std_logic_vector(3 downto 0);
    signal lif_nrn_index         : std_logic_vector(15 downto 0);
    signal lif_timestep          : std_logic;
    signal lif_nrn_state_next    : std_logic_vector(11 downto 0);
    signal lif_event_fifo_out    : std_logic_vector(15 downto 0);
    signal lif_event_fifo_we     : std_logic;
    signal lif_continue          : std_logic;
    signal lif_rst               : std_logic;
    signal goto_idle             : std_logic;

    -- synapse memory
    signal synmem_addr      : std_logic_vector(10 downto 0);
    signal synmem_dout      : std_logic_vector(31 downto 0);

    -- neuron memory
    signal nrnmem_addr      : std_logic_vector(1 downto 0);
    signal nrnmem_dout      : std_logic_vector(35 downto 0);

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

    SYN_MEMORY : entity work.SINGLE_PORT_BRAM
    generic map (
        DEPTH       => 1280,
        WIDTH       => 32,
        FILENAME    => "data/syn_init.data"
    )
    port map (
        i_we        => '0',
        i_addr      => synmem_addr,
        i_data      => (others => '0'),
        o_data      => synmem_dout,
        i_clk       => clk
    );

    NRN_MEMORY : entity work.DUAL_PORT_BRAM
    generic map (
        DEPTH       => 4,
        WIDTH       => 36,
        FILENAME    => "data/nrn_init.data"
    )
    port map (
        i_we        => '0',
        i_waddr     => (others => '0'),
        i_wdata     => (others => '0'),
        i_re        => '1',
        i_raddr     => nrnmem_addr,
        o_rdata     => nrnldr_data,
        i_clk       => clk
    );

    SYN_LOADER : entity work.SYNAPSE_LOADER
    generic map (
        SYN_MEM_DEPTH   => 1280,
        SYN_MEM_WIDTH   => 32
    )
    port map (
        i_cfg_en            => synldr_cfg_en,
        i_cfg_addr          => synldr_cfg_addr,
        i_cfg_val           => synldr_cfg_val,

        o_fifo_re           => fifo_re,
        i_fifo_rvalid       => fifo_rvalid,
        i_fifo_rdata        => fifo_rdata,

        o_syn_weight        => synldr_weight,
        o_syn_valid         => synldr_valid,
        o_syn_valid_next    => synldr_valid_next,
        o_syn_valid_last    => synldr_valid_last,

        o_syn_addr          => synmem_addr,
        i_syn_data          => synmem_dout,

        i_start             => synldr_start,
        i_continue          => lif_continue,
        o_busy              => synldr_busy,
        i_goto_idle         => goto_idle,
        i_clk               => clk,
        i_rst               => synldr_rst
    );

    NRN_LOADER : entity work.NEURON_LOADER
    generic map (
        NRN_MEM_DEPTH   => 4
    )
    port map (
        i_cfg_en            => nrnldr_cfg_en,
        i_cfg_addr          => nrnldr_cfg_addr,
        i_cfg_val           => nrnldr_cfg_val,
        o_nrn_re            => nrnldr_re,
        o_nrn_addr          => nrnmem_addr,
        i_nrn_data          => nrnldr_data,
        o_nrn_state         => nrnldr_state,
        o_nrn_index         => nrnldr_nrn_index,
        o_nrn_valid         => nrnldr_valid,
        o_nrn_valid_next    => nrnldr_valid_next,
        o_nrn_valid_last    => nnrldr_valid_last,
        i_start             => nrnldr_start,
        i_continue          => lif_continue,
        o_busy              => nrnldr_busy,
        i_goto_idle         => goto_idle,
        i_clk               => clk,
        i_rst               => nrnldr_rst
    );

    LIF : entity work.LIF_NEURON
    port map (
        i_cfg_en            => lif_cfg_en,
        i_cfg_addr          => lif_cfg_addr,
        i_cfg_val           => lif_cfg_val,
        i_nrn_valid         => nrnldr_valid,
        i_nrn_valid_next    => nrnldr_valid_next,
        i_nrn_valid_last    => nnrldr_valid_last,
        i_nrn_state         => nrnldr_state,
        i_syn_valid         => synldr_valid,
        i_syn_valid_next    => synldr_valid_next,
        i_syn_valid_last    => synldr_valid_last,
        i_syn_weight        => synldr_weight,
        i_nrn_index         => nrnldr_nrn_index,
        i_timestep          => lif_timestep,
        o_nrn_state_next    => lif_nrn_state_next,
        o_event_fifo_out    => lif_event_fifo_out,
        o_event_fifo_we     => lif_event_fifo_we,
        o_continue          => lif_continue,
        o_goto_idle         => goto_idle,
        i_clk               => clk,
        i_rst               => lif_rst
    );

    clk <= not clk after clk_period / 2;

    PROC_SEQUENCER : process
    begin

        -- Reset Synapse and Neuron Loader
        synldr_rst  <= '1';
        nrnldr_rst  <= '1';

        -- Reset FIFO
        fifo_rst    <= '1';
        fifo_we     <= '0';
        fifo_wdata  <= (others => '0');
        wait for 10 * clk_period;
        fifo_rst    <= '0';
        wait until rising_edge(clk);
        synldr_rst  <= '0';
        nrnldr_rst  <= '0';

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
            std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
        wait until rising_edge(clk);
        synldr_cfg_en   <= '0';
        wait until rising_edge(clk);

        -- configure neuron loader
        nrnldr_cfg_en   <= '1';
        nrnldr_cfg_addr <= "0000";
        nrnldr_cfg_val  <=
            "0000000000"                            &   -- zero padding
            std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
            std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
        wait until rising_edge(clk);
        nrnldr_cfg_en   <= '0';
        wait until rising_edge(clk);

        -- start processing events
        synldr_start    <= '1';
        nrnldr_start    <= '1';

        for i in 0 to 100 loop
            wait until rising_edge(clk);
            synldr_start    <= '0';
            nrnldr_start    <= '0';
        end loop;

        finish;
    end process;

end behavior;
