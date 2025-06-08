    ---------------------------------------------------------------------------------------------------
    --  Aarhus University (AU, Denmark)
    ---------------------------------------------------------------------------------------------------
    --
    --  File: tb_fenrir.vhd
    --  Description: Testbench for FENRIR/
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

    entity TB_FENRIR is
    end TB_FENRIR;

    architecture behavior of TB_FENRIR is

        constant clk_period     : time      := 10 ns;

        signal clk              : std_logic := '0';
        signal ctrl             : std_logic_vector(3 downto 0);
        signal led              : std_logic_vector(3 downto 0);
        signal ps_fifo          : std_logic_vector(31 downto 0);
        signal flags            : std_logic_vector(2 downto 0);
        signal fc2_fifo_empty   : std_logic;

        signal fc1_synldr_reg_cfg_0 : std_logic_vector(31 downto 0);
        signal fc1_nrnldr_reg_cfg_0 : std_logic_vector(31 downto 0);
        signal fc1_lif_reg_cfg_0    : std_logic_vector(31 downto 0);
        signal fc1_nrnwrt_reg_cfg_0 : std_logic_vector(31 downto 0);

        signal fc2_synldr_reg_cfg_0 : std_logic_vector(31 downto 0);
        signal fc2_nrnldr_reg_cfg_0 : std_logic_vector(31 downto 0);
        signal fc2_lif_reg_cfg_0    : std_logic_vector(31 downto 0);
        signal fc2_nrnwrt_reg_cfg_0 : std_logic_vector(31 downto 0);

        signal class_count_0        : std_logic_vector(31 downto 0);
        signal class_count_1        : std_logic_vector(31 downto 0);
        signal class_count_2        : std_logic_vector(31 downto 0);
        signal class_count_3        : std_logic_vector(31 downto 0);
        signal class_count_4        : std_logic_vector(31 downto 0);
        signal class_count_5        : std_logic_vector(31 downto 0);
        signal class_count_6        : std_logic_vector(31 downto 0);
        signal class_count_7        : std_logic_vector(31 downto 0);
        signal class_count_8        : std_logic_vector(31 downto 0);
        signal class_count_9        : std_logic_vector(31 downto 0);
        signal class_count_10       : std_logic_vector(31 downto 0);

        signal pendulum             : std_logic;

    begin

    FENRIR : entity work.FENRIR_TOP
        port map (
            i_sysclk                => clk,
            i_ctrl                  => ctrl,
            o_flags                 => flags,
            o_led                   => led,
            i_ps_fifo               => ps_fifo,

            i_fc1_synldr_reg_cfg_0  => fc1_synldr_reg_cfg_0,
            i_fc1_nrnldr_reg_cfg_0  => fc1_nrnldr_reg_cfg_0,
            i_fc1_lif_reg_cfg_0     => fc1_lif_reg_cfg_0,
            i_fc1_nrnwrt_reg_cfg_0  => fc1_nrnwrt_reg_cfg_0,

            i_fc2_synldr_reg_cfg_0  => fc2_synldr_reg_cfg_0,
            i_fc2_nrnldr_reg_cfg_0  => fc2_nrnldr_reg_cfg_0,
            i_fc2_lif_reg_cfg_0     => fc2_lif_reg_cfg_0,
            i_fc2_nrnwrt_reg_cfg_0  => fc2_nrnwrt_reg_cfg_0,

            o_class_count_0         => class_count_0,
            o_class_count_1         => class_count_1,
            o_class_count_2         => class_count_2,
            o_class_count_3         => class_count_3,
            o_class_count_4         => class_count_4,
            o_class_count_5         => class_count_5,
            o_class_count_6         => class_count_6,
            o_class_count_7         => class_count_7,
            o_class_count_8         => class_count_8,
            o_class_count_9         => class_count_9,
            o_class_count_10        => class_count_10
        );

        clk <= not clk after clk_period / 2;

        PROC_SEQUENCER : process

            file bin_file           : text open read_mode is "C:/home/university/8-semester/fenrir/src/design_sources/data/gesture_data.txt";
            variable line_buffer    : line;
            variable bv_data        : bit_vector(12 downto 0);
            variable slv_data       : std_logic_vector(12 downto 0);

        begin

            ctrl    <= "0000";
            ps_fifo <= (others => '0');
            pendulum <= '0';
            
            fc1_synldr_reg_cfg_0 <=
                "00000000"                              &   -- zero padding
                std_logic_vector(to_unsigned(1, 2))     &   -- bits per weight
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(64, 11));      -- neurons per layer

            fc1_nrnldr_reg_cfg_0 <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(64, 11));      -- neurons per 

            fc1_lif_reg_cfg_0 <=
                std_logic_vector(to_unsigned(10, 8))    &   -- weight scalar
                std_logic_vector(to_unsigned(18, 12))  &   -- beta
                std_logic_vector(to_unsigned(390, 12));      -- thresholdlayer

            fc1_nrnwrt_reg_cfg_0 <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(64, 11));      -- neurons per layer

            fc2_synldr_reg_cfg_0 <=
                "00000000"                              &   -- zero padding
                std_logic_vector(to_unsigned(1, 2))     &   -- bits per weight
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(11, 11));      -- neurons per layer

            --
            fc2_nrnldr_reg_cfg_0 <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(11, 11));      -- neurons per 

            fc2_lif_reg_cfg_0 <=
                std_logic_vector(to_unsigned(10, 8))    &   -- weight scalar
                std_logic_vector(to_unsigned(16, 12))  &   -- beta
                std_logic_vector(to_unsigned(477, 12));      -- thresholdlayer

            fc2_nrnwrt_reg_cfg_0 <=
                "0000000000"                            &   -- zero padding
                std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
                std_logic_vector(to_unsigned(11, 11));      -- neurons per layer

            wait until rising_edge(clk);

            ctrl    <= "0011";

            wait for 10 * clk_period;

            while not endfile(bin_file) loop

                readline(bin_file, line_buffer);
                read(line_buffer, bv_data);

                slv_data := to_stdlogicvector(bv_data);

                ps_fifo <= pendulum & "000000000000000000" & slv_data;
                pendulum <= not pendulum;

                wait until rising_edge(clk);

                while (flags(2) = '1') loop
                    wait until rising_edge(clk);
                end loop;

            end loop;

            while (flags(0) = '0') loop
                wait for 100 * clk_period;
            end loop;
            
            finish;
        end process;

    end behavior;
