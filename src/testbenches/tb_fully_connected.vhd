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

use ieee.std_logic_textio.all;
use std.textio.all;

entity TB_FULLY_CONNECTED is
end TB_FULLY_CONNECTED;

architecture behavior of TB_FULLY_CONNECTED is

    constant clk_period : time := 10 ns;
    constant DEPTH      : integer := 512;
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

    -- output event fifo
    signal out_fifo_we          : std_logic;
    signal out_fifo_wdata       : std_logic_vector(16 - 1 downto 0);
    signal out_fifo_re          : std_logic;
    signal out_fifo_rvalid      : std_logic;
    signal out_fifo_rdata       : std_logic_vector(16 - 1 downto 0);
    signal out_fifo_empty       : std_logic;
    signal out_fifo_empty_next  : std_logic;
    signal out_fifo_full        : std_logic;
    signal out_fifo_full_next   : std_logic;
    signal out_fifo_fill_count  : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal out_fifo_rst         : std_logic;
    signal out_fifo_fault       : std_logic;

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
    signal lif_out_valid         : std_logic;

    -- neuron writer
    signal nrnwrt_cfg_en        : std_logic;
    signal nrnwrt_cfg_addr      : std_logic_vector(3 downto 0);
    signal nrnwrt_cfg_val       : std_logic_vector(31 downto 0);
    signal nrnwrt_mem_we        : std_logic;
    signal nrnwrt_mem_addr      : std_logic_vector(1 downto 0);
    signal nrnwrt_mem_data      : std_logic_vector(35 downto 0);
    signal nrnwrt_rst           : std_logic;
    signal nrnwrt_fault         : std_logic;

    -- scheduler
    signal scheduler_en         : std_logic;
    signal scheduler_busy       : std_logic;
    signal scheduler_rst        : std_logic;
    signal scheduler_timestep   : std_logic;

    -- synapse memory
    signal synmem_addr      : std_logic_vector(9 downto 0);
    signal synmem_dout      : std_logic_vector(39 downto 0);

    -- neuron memory
    signal nrnmem_addr      : std_logic_vector(1 downto 0);
    signal nrnmem_dout      : std_logic_vector(35 downto 0);

    signal syn_addr         : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal syn_data         : std_logic_vector(WIDTH - 1 downto 0);

    -- testbench
    signal event_number     : integer range 0 to 1024;
    signal tb_timestep      : std_logic := '0';

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

    OUTPUT_FIFO : entity work.BRAM_FIFO
        generic map (
            DEPTH => DEPTH,
            WIDTH => 16
        )
        port map (
            i_we                => out_fifo_we,
            i_wdata             => out_fifo_wdata,
            i_re                => '0',
            o_rvalid            => open,
            o_rdata             => open,
            o_empty             => open,
            o_empty_next        => open,
            o_full              => out_fifo_full,
            o_full_next         => open,
            o_fill_count        => open,
            i_clk               => clk,
            i_rst               => out_fifo_rst,
            o_fault             => out_fifo_fault
        );

    SYN_MEMORY : entity work.SINGLE_PORT_BRAM
    generic map (
        DEPTH       => 1024,
        WIDTH       => 40,
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
        i_we        => nrnwrt_mem_we,
        i_waddr     => nrnwrt_mem_addr,
        i_wdata     => nrnwrt_mem_data,
        i_re        => nrnldr_re,
        i_raddr     => nrnmem_addr,
        o_rdata     => nrnldr_data,
        i_clk       => clk
    );

    SYN_LOADER : entity work.SYNAPSE_LOADER
    generic map (
        SYN_MEM_DEPTH   => 1024,
        SYN_MEM_WIDTH   => 40
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
        i_timestep          => scheduler_timestep,
        o_nrn_state_next    => lif_nrn_state_next,
        o_event_fifo_out    => out_fifo_wdata,
        o_event_fifo_we     => out_fifo_we,
        o_output_valid      => lif_out_valid,
        o_continue          => lif_continue,
        o_goto_idle         => goto_idle,
        i_clk               => clk,
        i_rst               => lif_rst
    );

    NRN_WRITER : entity work.NEURON_WRITER
    generic map (
        NRN_MEM_DEPTH   => 4
    )
    port map (
        i_cfg_en    => nrnwrt_cfg_en,
        i_cfg_addr  => nrnwrt_cfg_addr,
        i_cfg_val   => nrnwrt_cfg_val,
        o_nrn_we    => nrnwrt_mem_we,
        o_nrn_addr  => nrnwrt_mem_addr,
        o_nrn_data  => nrnwrt_mem_data,
        i_nrn_state => lif_nrn_state_next,
        i_valid     => lif_out_valid,
        i_nrn_data  => (others => '0'),
        i_clk       => clk,
        i_rst       => nrnwrt_rst,
        o_fault     => nrnwrt_fault
    );

    SCHEDULER : entity work.SCHEDULER
    port map (
        i_enable        => scheduler_en,
        i_timestep      => tb_timestep,
        i_synldr_busy   => synldr_busy,
        i_nrnldr_busy   => nrnldr_busy,
        o_synldr_start  => synldr_start,
        o_nrnldr_start  => nrnldr_start,
        o_timestep      => scheduler_timestep,
        i_fifo_in_empty => fifo_empty,
        i_fifo_out_full => out_fifo_full,
        o_busy          => scheduler_busy,
        i_clk           => clk,
        i_rst           => scheduler_rst
    );

    clk <= not clk after clk_period / 2;

    MEMREC_WRITE_PROCESS : process(clk)
        file result : text open write_mode is ("mem_rec.csv");
        variable lo : line;
    begin
        if rising_edge(clk) then
            if nrnwrt_mem_we = '1' then
                write(lo, nrnwrt_mem_addr);
                write(lo, ',');
                write(lo, nrnwrt_mem_data);
                writeline(result, lo);
            end if;
        end if;
    end process;

    SPKREC_WRITE_PROCESS : process(clk)
        file result : text open write_mode is ("spk_rec.csv");
        variable lo : line;
    begin
        if rising_edge(clk) then
            if out_fifo_we = '1' then
                write(lo, event_number);
                write(lo, ',');
                write(lo, out_fifo_wdata);
                writeline(result, lo);
            end if;
        end if;
    end process;

    PROC_SEQUENCER : process

        file bin_file           : text open read_mode is "C:/home/university/8-semester/fenrir/src/design_sources/data/spike_data.txt";
        variable line_buffer    : line;
        variable bv_data        : bit_vector(31 downto 0);
        variable slv_data       : std_logic_vector(31 downto 0);

        variable tstep          : integer := 0;

    begin

        -- Reset Synapse and Neuron Loader
        synldr_rst      <= '1';
        nrnldr_rst      <= '1';
        nrnwrt_rst      <= '1';
        scheduler_en    <= '0';
        scheduler_rst   <= '1';
        event_number    <= 0;

        -- Reset FIFOs
        fifo_rst        <= '1';
        fifo_we         <= '0';
        fifo_wdata      <= (others => '0');
        out_fifo_rst    <= '1';
        wait for 10 * clk_period;
        fifo_rst        <= '0';
        out_fifo_rst    <= '0';
        wait until rising_edge(clk);
        synldr_rst      <= '0';
        nrnldr_rst      <= '0';
        nrnwrt_rst      <= '0';
        scheduler_rst   <= '0';

        -- start writing
        fifo_we     <= '1';

        -- dummy fill since we always skip the first ???
        fifo_wdata  <= (others => '0');
        wait until rising_edge(clk);

        -- fill the FIFO
        while fifo_full_next = '0' loop
            if not endfile(bin_file) then
                readline(bin_file, line_buffer);
                read(line_buffer, bv_data);

                slv_data := to_stdlogicvector(bv_data);

                fifo_wdata <= slv_data;
            else
                exit;
            end if;

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

        -- configure lif
        lif_cfg_en      <= '1';
        lif_cfg_addr    <= "0000";
        lif_cfg_val     <=
            "00000000"                              &   -- zero padding
            std_logic_vector(to_unsigned(1, 12))    &   -- beta
            std_logic_vector(to_unsigned(38, 12));      -- threshold
        wait until rising_edge(clk);
        lif_cfg_en   <= '0';
        wait until rising_edge(clk);

        -- configure neuron writer
        nrnwrt_cfg_en      <= '1';
        nrnwrt_cfg_addr    <= "0000";
        nrnwrt_cfg_val     <=
            "0000000000"                            &   -- zero padding
            std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
            std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
        wait until rising_edge(clk);
        nrnwrt_cfg_en   <= '0';
        wait until rising_edge(clk);

        while fifo_empty = '0' loop

            -- start processing events
            if (event_number = 6) then
                tb_timestep <= '1';
            else
                tb_timestep <= '0';
            end if;

            scheduler_en    <= '1';
            wait until rising_edge(clk) and scheduler_busy = '1';
            scheduler_en    <= '0';

            while scheduler_busy = '1' loop
                wait until rising_edge(clk);
            end loop;

            for i in 0 to 10 loop
                wait until rising_edge(clk);
            end loop;

            event_number <= event_number + 1;

        end loop;

        for i in 0 to 10 loop
            wait until rising_edge(clk);
        end loop;

        finish;
    end process;

end behavior;
