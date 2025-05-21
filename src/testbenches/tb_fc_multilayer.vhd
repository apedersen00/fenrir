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

entity TB_FC_MULTILAYER is
end TB_FC_MULTILAYER;

architecture behavior of TB_FC_MULTILAYER is

    constant OUT_SIZE       : integer := 10;
    constant clk_period     : time      := 10 ns;
    signal clk              : std_logic := '0';

    signal fc1_cfg_en       : std_logic;
    signal fc1_cfg_addr     : std_logic_vector(7 downto 0);
    signal fc1_cfg_val      : std_logic_vector(31 downto 0);

    signal fc1_en           : std_logic;
    signal fc1_in_we        : std_logic;
    signal fc1_in_wdata     : std_logic_vector(11 downto 0);
    signal fc1_full         : std_logic;
    signal fc2_full         : std_logic;
    signal fc1_out_we       : std_logic;
    signal fc1_out_wdata    : std_logic_vector(11 downto 0);
    signal fc1_timestep     : std_logic;
    signal rst              : std_logic;
    signal fc1_busy         : std_logic;

    signal tb_tstep         : integer := 0;      -- for testbench only
    signal tb_nrnmem_we     : std_logic;
    signal tb_nrnmem_waddr  : std_logic_vector(integer(ceil(log2(real(integer(ceil(real(OUT_SIZE) / 3.0))))))-1 downto 0);
    signal tb_nrnmem_wdata  : std_logic_vector(35 downto 0);

begin

    FC1 : entity work.FC_LAYER
    generic map (
        IN_SIZE         => 1024,
        OUT_SIZE        => 10,
        SYN_MEM_WIDTH   => 40,
        BITS_PER_SYN    => 4,
        SYN_INIT_FILE   => "data/syn_init.data",
        NRN_INIT_FILE   => "data/nrn_init.data"
    )
    port map (
        i_cfg_en            => fc1_cfg_en,
        i_cfg_addr          => fc1_cfg_addr,
        i_cfg_val           => fc1_cfg_val,
        i_enable            => fc1_en,
        i_in_fifo_we        => fc1_in_we,
        i_in_fifo_wdata     => fc1_in_wdata,
        o_in_fifo_full      => fc1_full,
        i_out_fifo_full     => fc2_full,
        o_out_fifo_we       => fc1_out_we,
        o_out_fifo_wdata    => fc1_out_wdata,
        i_timestep          => fc1_timestep,
        i_rst               => rst,
        i_clk               => clk,
        o_busy              => fc1_busy,
        o_nrnmem_we         => tb_nrnmem_we,
        o_nrnmem_waddr      => tb_nrnmem_waddr,
        o_nrnmem_wdata      => tb_nrnmem_wdata
    );

    OUTPUT_FIFO : entity work.BRAM_FIFO
        generic map (
            DEPTH => 256,
            WIDTH => 12
        )
        port map (
            i_we                => fc1_out_we,
            i_wdata             => fc1_out_wdata,
            i_re                => '0',
            o_rvalid            => open,
            o_rdata             => open,
            o_empty             => open,
            o_empty_next        => open,
            o_full              => fc2_full,
            o_full_next         => open,
            o_fill_count        => open,
            i_clk               => clk,
            i_rst               => rst
        );

    clk <= not clk after clk_period / 2;

    SPKREC_WRITE_PROCESS : process(clk)
        file result : text open write_mode is ("spk_rec.csv");
        variable lo : line;
    begin
        if rising_edge(clk) then
            if fc1_out_we = '1' then
                write(lo, tb_tstep - 1);
                write(lo, ',');
                write(lo, fc1_out_wdata);
                writeline(result, lo);
            end if;
        end if;
    end process;

    MEMREC_WRITE_PROCESS : process(clk)
        file result : text open write_mode is ("mem_rec.csv");
        variable lo : line;
    begin
        if rising_edge(clk) then
            if (tb_nrnmem_we = '1') then
                write(lo, tb_tstep - 1);
                write(lo, ',');
                write(lo, tb_nrnmem_waddr);
                write(lo, ',');
                write(lo, tb_nrnmem_wdata);
                writeline(result, lo);
            end if;
        end if;
    end process;
    
    PROC_SEQUENCER : process

        file bin_file           : text open read_mode is "C:/home/university/8-semester/fenrir/src/design_sources/data/test_spike_data.txt";
        variable line_buffer    : line;
        variable bv_data        : bit_vector(12 downto 0);
        variable slv_data       : std_logic_vector(12 downto 0);

    begin

        -- Reset Synapse and Neuron Loader
        fc1_in_we   <= '0';
        fc1_en      <= '0';
        rst         <= '1';
        wait for 10 * clk_period;
        rst     <= '0';
        wait until rising_edge(clk);

        -- configure synapse loader
        fc1_cfg_en      <= '1';
        fc1_cfg_addr    <= "00000000";
        fc1_cfg_val     <=
            "00000000"                              &   -- zero padding
            std_logic_vector(to_unsigned(1, 2))     &   -- bits per weight
            std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
            std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
        wait until rising_edge(clk);
        fc1_cfg_en      <= '0';
        wait until rising_edge(clk);

        -- configure neuron loader
        fc1_cfg_en      <= '1';
        fc1_cfg_addr    <= "00010000";
        fc1_cfg_val     <=
            "0000000000"                            &   -- zero padding
            std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
            std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
        wait until rising_edge(clk);
        fc1_cfg_en      <= '0';
        wait until rising_edge(clk);

        -- configure lif
        fc1_cfg_en      <= '1';
        fc1_cfg_addr    <= "00100000";
        fc1_cfg_val     <=
            std_logic_vector(to_unsigned(100, 8))   &   -- weight scalar
            std_logic_vector(to_unsigned(130, 12))  &   -- beta
            std_logic_vector(to_unsigned(275, 12));     -- threshold
        wait until rising_edge(clk);
        fc1_cfg_en      <= '0';
        wait until rising_edge(clk);

        -- configure neuron writer
        fc1_cfg_en      <= '1';
        fc1_cfg_addr    <= "00110000";
        fc1_cfg_val     <=
            "0000000000"                            &   -- zero padding
            std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
            std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
        wait until rising_edge(clk);
        fc1_cfg_en      <= '0';
        wait until rising_edge(clk);

        while not endfile(bin_file) loop

            readline(bin_file, line_buffer);
            read(line_buffer, bv_data);

            slv_data := to_stdlogicvector(bv_data);

            if (slv_data(12) = '1') then
                fc1_timestep    <= '1';
                tb_tstep        <= tb_tstep + 1;
            else
                fc1_in_we       <= '1';
                fc1_in_wdata    <= slv_data(11 downto 0);
            end if;

            wait until rising_edge(clk);

            fc1_in_we       <= '0';
            fc1_in_wdata    <= (others => '0');

            wait until rising_edge(clk);

            fc1_en  <= '1';

            wait for 10 * clk_period;

            fc1_en  <= '0';

            while (fc1_busy = '1') loop
                wait until rising_edge(clk);
            end loop;

            wait for 10 * clk_period;

            fc1_timestep    <= '0';

        end loop;

        finish;
    end process;

end behavior;
