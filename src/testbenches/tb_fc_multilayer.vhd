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
        signal rst              : std_logic;

        signal fc1_cfg_en       : std_logic;
        signal fc1_cfg_addr     : std_logic_vector(7 downto 0);
        signal fc1_cfg_val      : std_logic_vector(31 downto 0);
        signal fc1_en           : std_logic;
        signal fc1_in_we        : std_logic;
        signal fc1_in_wdata     : std_logic_vector(12 downto 0);
        signal fc1_empty        : std_logic;
        signal fc1_full         : std_logic;
        signal fc1_fill_count   : std_logic_vector(7 downto 0);
        signal fc1_out_we       : std_logic;
        signal fc1_out_wdata    : std_logic_vector(12 downto 0);
        signal fc1_timestep     : std_logic;
        signal fc1_busy         : std_logic;

        signal fc2_cfg_en       : std_logic;
        signal fc2_cfg_addr     : std_logic_vector(7 downto 0);
        signal fc2_cfg_val      : std_logic_vector(31 downto 0);
        signal fc2_en           : std_logic;
        signal fc2_in_we        : std_logic;
        signal fc2_in_wdata     : std_logic_vector(12 downto 0);
        signal fc2_empty        : std_logic;
        signal fc2_full         : std_logic;
        signal fc2_fill_count   : std_logic_vector(7 downto 0);
        signal fc2_out_we       : std_logic;
        signal fc2_out_wdata    : std_logic_vector(12 downto 0);
        signal fc2_timestep     : std_logic;
        signal fc2_busy         : std_logic;

        signal fc3_cfg_en       : std_logic;
        signal fc3_cfg_addr     : std_logic_vector(7 downto 0);
        signal fc3_cfg_val      : std_logic_vector(31 downto 0);
        signal fc3_en           : std_logic;
        signal fc3_in_we        : std_logic;
        signal fc3_in_wdata     : std_logic_vector(12 downto 0);
        signal fc3_empty        : std_logic;
        signal fc3_full         : std_logic;
        signal fc3_fill_count   : std_logic_vector(7 downto 0);
        signal out_fifo_full    : std_logic;
        signal fc3_out_we       : std_logic;
        signal fc3_out_wdata    : std_logic_vector(12 downto 0);
        signal fc3_timestep     : std_logic;
        signal fc3_busy         : std_logic;

        signal out_fifo_empty   : std_logic;

        signal tb_fc1_tstep     : integer := 0;
        signal tb_fc2_tstep     : integer := 0;
        signal tb_fc3_tstep     : integer := 0;
        signal tb_tstep         : integer := 0;      -- for testbench only
        signal tb_nrnmem1_we    : std_logic;
        signal tb_nrnmem1_waddr : std_logic_vector(integer(ceil(log2(real(integer(ceil(real(256) / 3.0))))))-1 downto 0);
        signal tb_nrnmem1_wdata : std_logic_vector(35 downto 0);
        signal tb_nrnmem2_we    : std_logic;
        signal tb_nrnmem2_waddr : std_logic_vector(integer(ceil(log2(real(integer(ceil(real(128) / 3.0))))))-1 downto 0);
        signal tb_nrnmem2_wdata : std_logic_vector(35 downto 0);
        signal tb_nrnmem3_we    : std_logic;
        signal tb_nrnmem3_waddr : std_logic_vector(integer(ceil(log2(real(integer(ceil(real(10) / 3.0))))))-1 downto 0);
        signal tb_nrnmem3_wdata : std_logic_vector(35 downto 0);

    begin

        FC1 : entity work.FC_LAYER
        generic map (
            IN_SIZE         => 256,
            OUT_SIZE        => 256,
            OUT_FIFO_DEPTH  => 256,
            IS_LAST         => 0,
            SYN_MEM_WIDTH   => 32,
            BITS_PER_SYN    => 4,
            SYN_INIT_FILE   => "data/fc1_syn.data",
            NRN_INIT_FILE   => ""
        )
        port map (
            -- config
            i_cfg_en                => fc1_cfg_en,
            i_cfg_addr              => fc1_cfg_addr,
            i_cfg_val               => fc1_cfg_val,
            -- input
            i_in_fifo_we            => fc1_in_we,
            i_in_fifo_wdata         => fc1_in_wdata,
            -- output
            o_out_fifo_we           => fc1_out_we,
            o_out_fifo_wdata        => fc1_out_wdata,
            -- status
            o_in_fifo_empty         => fc1_empty,
            o_in_fifo_full          => fc1_full,
            o_in_fifo_fill_count    => open,
            i_out_fifo_full         => fc2_full,
            i_out_fifo_empty        => fc2_empty,
            i_out_fifo_fill_count   => fc2_fill_count,
            o_busy                  => fc1_busy,
            -- control
            i_enable                => fc1_en,
            i_rst                   => rst,
            i_clk                   => clk,
            -- debug
            o_sched_tstep           => open,
            o_nrnmem_we             => tb_nrnmem1_we,
            o_nrnmem_waddr          => tb_nrnmem1_waddr,
            o_nrnmem_wdata          => tb_nrnmem1_wdata
        );

        FC2 : entity work.FC_LAYER
        generic map (
            IN_SIZE         => 256,
            OUT_SIZE        => 128,
            OUT_FIFO_DEPTH  => 256,
            IS_LAST         => 0,
            SYN_MEM_WIDTH   => 32,
            BITS_PER_SYN    => 4,
            SYN_INIT_FILE   => "data/fc2_syn.data",
            NRN_INIT_FILE   => ""
        )
        port map (
            -- config
            i_cfg_en                => fc2_cfg_en,
            i_cfg_addr              => fc2_cfg_addr,
            i_cfg_val               => fc2_cfg_val,
            -- input
            i_in_fifo_we            => fc1_out_we,
            i_in_fifo_wdata         => fc1_out_wdata,
            -- output
            o_out_fifo_we           => fc2_out_we,
            o_out_fifo_wdata        => fc2_out_wdata,
            -- status
            o_in_fifo_empty         => fc2_empty,
            o_in_fifo_full          => fc2_full,
            o_in_fifo_fill_count    => fc2_fill_count,
            i_out_fifo_full         => fc3_full,
            i_out_fifo_empty        => fc3_empty,
            i_out_fifo_fill_count   => fc3_fill_count,
            o_busy                  => fc2_busy,
            -- control
            i_enable                => fc2_en,
            i_rst                   => rst,
            i_clk                   => clk,
            -- debug
            o_sched_tstep           => open,
            o_nrnmem_we             => tb_nrnmem2_we,
            o_nrnmem_waddr          => tb_nrnmem2_waddr,
            o_nrnmem_wdata          => tb_nrnmem2_wdata
        );

        FC3 : entity work.FC_LAYER
        generic map (
            IN_SIZE         => 128,
            OUT_SIZE        => 10,
            OUT_FIFO_DEPTH  => 512,
            IS_LAST         => 0,
            SYN_MEM_WIDTH   => 40,
            BITS_PER_SYN    => 4,
            SYN_INIT_FILE   => "data/fc3_syn.data",
            NRN_INIT_FILE   => ""
        )
        port map (
            -- config
            i_cfg_en                => fc3_cfg_en,
            i_cfg_addr              => fc3_cfg_addr,
            i_cfg_val               => fc3_cfg_val,
            -- input
            i_in_fifo_we            => fc2_out_we,
            i_in_fifo_wdata         => fc2_out_wdata,
            -- output
            o_out_fifo_we           => fc3_out_we,
            o_out_fifo_wdata        => fc3_out_wdata,
            -- status
            o_in_fifo_empty         => fc3_empty,
            o_in_fifo_full          => fc3_full,
            o_in_fifo_fill_count    => fc3_fill_count,
            i_out_fifo_full         => out_fifo_full,
            i_out_fifo_empty        => '1',
            i_out_fifo_fill_count   => "000000000",
            o_busy                  => fc3_busy,
            -- control
            i_enable                => fc3_en,
            i_rst                   => rst,
            i_clk                   => clk,
            -- debug
            o_sched_tstep           => open,
            o_nrnmem_we             => tb_nrnmem3_we,
            o_nrnmem_waddr          => tb_nrnmem3_waddr,
            o_nrnmem_wdata          => tb_nrnmem3_wdata
        );

        OUTPUT_FIFO : entity work.BRAM_FIFO
            generic map (
                DEPTH => 512,
                WIDTH => 13
            )
            port map (
                i_we                => fc3_out_we,
                i_wdata             => fc3_out_wdata,
                i_re                => '0',
                o_rvalid            => open,
                o_rdata             => open,
                o_empty             => out_fifo_empty,
                o_empty_next        => open,
                o_full              => out_fifo_full,
                o_full_next         => open,
                o_fill_count        => open,
                i_clk               => clk,
                i_rst               => rst
            );

        clk <= not clk after clk_period / 2;

        SPKREC1_WRITE_PROCESS : process(clk)
            file result : text open write_mode is ("spk_rec1.csv");
            variable lo : line;
        begin
            if rising_edge(clk) then
                if fc1_out_we = '1' then
                    write(lo, tb_fc1_tstep);
                    write(lo, ',');
                    write(lo, fc1_out_wdata);
                    writeline(result, lo);
                end if;
            end if;
        end process;

        MEMREC1_WRITE_PROCESS : process(clk)
            file result : text open write_mode is ("mem_rec1.csv");
            variable lo : line;
        begin
            if rising_edge(clk) then
                if (tb_nrnmem1_we = '1') then
                    write(lo, tb_fc1_tstep);
                    write(lo, ',');
                    write(lo, tb_nrnmem1_waddr);
                    write(lo, ',');
                    write(lo, tb_nrnmem1_wdata);
                    writeline(result, lo);
                end if;
            end if;
        end process;

        SPKREC2_WRITE_PROCESS : process(clk)
            file result : text open write_mode is ("spk_rec2.csv");
            variable lo : line;
        begin
            if rising_edge(clk) then
                if fc2_out_we = '1' then
                    write(lo, tb_fc2_tstep);
                    write(lo, ',');
                    write(lo, fc2_out_wdata);
                    writeline(result, lo);
                end if;
            end if;
        end process;

        MEMREC2_WRITE_PROCESS : process(clk)
            file result : text open write_mode is ("mem_rec2.csv");
            variable lo : line;
        begin
            if rising_edge(clk) then
                if (tb_nrnmem2_we = '1') then
                    write(lo, tb_fc2_tstep);
                    write(lo, ',');
                    write(lo, tb_nrnmem2_waddr);
                    write(lo, ',');
                    write(lo, tb_nrnmem2_wdata);
                    writeline(result, lo);
                end if;
            end if;
        end process;

        SPKREC3_WRITE_PROCESS : process(clk)
            file result : text open write_mode is ("spk_rec3.csv");
            variable lo : line;
        begin
            if rising_edge(clk) then
                if fc3_out_we = '1' then
                    write(lo, tb_fc3_tstep);
                    write(lo, ',');
                    write(lo, fc3_out_wdata);
                    writeline(result, lo);
                end if;
            end if;
        end process;

        MEMREC3_WRITE_PROCESS : process(clk)
            file result : text open write_mode is ("mem_rec3.csv");
            variable lo : line;
        begin
            if rising_edge(clk) then
                if (tb_nrnmem3_we = '1') then
                    write(lo, tb_fc3_tstep);
                    write(lo, ',');
                    write(lo, tb_nrnmem3_waddr);
                    write(lo, ',');
                    write(lo, tb_nrnmem3_wdata);
                    writeline(result, lo);
                end if;
            end if;
        end process;

        TSTEP_INCREMENT : process(clk)
        begin
            if rising_edge(clk) then
                if (fc1_out_we = '1' and fc1_out_wdata(12) = '1') then
                    tb_fc1_tstep <= tb_fc1_tstep + 1;
                end if;
                if (fc2_out_we = '1' and fc2_out_wdata(12) = '1') then
                    tb_fc2_tstep <= tb_fc2_tstep + 1;
                end if;
                if (fc3_out_we = '1' and fc3_out_wdata(12) = '1') then
                    tb_fc3_tstep <= tb_fc3_tstep + 1;
                end if;
            end if;
        end process;
        
        PROC_SEQUENCER : process

            file bin_file           : text open read_mode is "C:/home/university/8-semester/fenrir/src/design_sources/data/nmnist_data.txt";
            variable line_buffer    : line;
            variable bv_data        : bit_vector(12 downto 0);
            variable slv_data       : std_logic_vector(12 downto 0);

        begin

            -- Reset Synapse and Neuron Loader
            fc1_en      <= '0';
            fc2_en      <= '0';
            fc3_en      <= '0';
            fc1_in_we   <= '0';
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
                std_logic_vector(to_unsigned(256, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc1_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure neuron loader
            fc1_cfg_en      <= '1';
            fc1_cfg_addr    <= "00010000";
            fc1_cfg_val     <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(256, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc1_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure lif
            fc1_cfg_en      <= '1';
            fc1_cfg_addr    <= "00100000";
            fc1_cfg_val     <=
                std_logic_vector(to_unsigned(10, 8))   &   -- weight scalar
                std_logic_vector(to_unsigned(103, 12))  &   -- beta
                std_logic_vector(to_unsigned(75, 12));     -- threshold
            wait until rising_edge(clk);
            fc1_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure neuron writer
            fc1_cfg_en      <= '1';
            fc1_cfg_addr    <= "00110000";
            fc1_cfg_val     <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(256, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc1_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure synapse loader
            fc2_cfg_en      <= '1';
            fc2_cfg_addr    <= "00000000";
            fc2_cfg_val     <=
                "00000000"                              &   -- zero padding
                std_logic_vector(to_unsigned(1, 2))     &   -- bits per weight
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(128, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc2_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure neuron loader
            fc2_cfg_en      <= '1';
            fc2_cfg_addr    <= "00010000";
            fc2_cfg_val     <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(128, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc2_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure lif
            fc2_cfg_en      <= '1';
            fc2_cfg_addr    <= "00100000";
            fc2_cfg_val     <=
                std_logic_vector(to_unsigned(10, 8))   &   -- weight scalar
                std_logic_vector(to_unsigned(56, 12))  &   -- beta
                std_logic_vector(to_unsigned(47, 12));     -- threshold
            wait until rising_edge(clk);
            fc2_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure neuron writer
            fc2_cfg_en      <= '1';
            fc2_cfg_addr    <= "00110000";
            fc2_cfg_val     <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(128, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc2_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure synapse loader
            fc3_cfg_en      <= '1';
            fc3_cfg_addr    <= "00000000";
            fc3_cfg_val     <=
                "00000000"                              &   -- zero padding
                std_logic_vector(to_unsigned(1, 2))     &   -- bits per weight
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc3_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure neuron loader
            fc3_cfg_en      <= '1';
            fc3_cfg_addr    <= "00010000";
            fc3_cfg_val     <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc3_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure lif
            fc3_cfg_en      <= '1';
            fc3_cfg_addr    <= "00100000";
            fc3_cfg_val     <=
                std_logic_vector(to_unsigned(10, 8))   &   -- weight scalar
                std_logic_vector(to_unsigned(244, 12))  &   -- beta
                std_logic_vector(to_unsigned(228, 12));     -- threshold
            wait until rising_edge(clk);
            fc3_cfg_en      <= '0';
            wait until rising_edge(clk);

            -- configure neuron writer
            fc3_cfg_en      <= '1';
            fc3_cfg_addr    <= "00110000";
            fc3_cfg_val     <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(10, 11));      -- neurons per layer
            wait until rising_edge(clk);
            fc3_cfg_en      <= '0';
            wait until rising_edge(clk);
            
            fc1_en <= '1';
            fc2_en <= '1';
            fc3_en <= '1';

            while not endfile(bin_file) loop

                readline(bin_file, line_buffer);
                read(line_buffer, bv_data);

                slv_data := to_stdlogicvector(bv_data);

                -- wait until fc1_busy = '0';

                if (slv_data(12) = '1') then
                    tb_tstep        <= tb_tstep + 1;
                    fc1_in_we       <= '1';
                    fc1_in_wdata    <= '1' & "000000000000";
                else
                    fc1_in_we       <= '1';
                    fc1_in_wdata    <= '0' & slv_data(11 downto 0);
                end if;

                wait until rising_edge(clk);

                fc1_in_we       <= '0';
                fc1_in_wdata    <= (others => '0');

                wait until fc1_busy = '1';
                while (fc1_busy = '1') loop
                    wait until rising_edge(clk);
                end loop;

                -- wait for 10 * clk_period;

            end loop;

            while (fc3_empty = '1' or fc2_empty = '1' or fc1_empty = '1') loop
                wait until rising_edge(clk);
            end loop;

            finish;
        end process;

    end behavior;
