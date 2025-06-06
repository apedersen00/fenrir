---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: fenrir_top.vhd
--  Description: Top module for FENRIR.
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

entity FENRIR_TOP is
    port (
        i_sysclk                : in std_logic;
        i_ctrl                  : in std_logic_vector(3 downto 0);  -- fsm control
        o_flags                 : out std_logic_vector(2 downto 0); -- in_fifo_full_next & busy & idle
        o_led                   : out std_logic_vector(3 downto 0);

        -- PS FIFO
        i_ps_fifo               : in std_logic_vector(31 downto 0);

        -- fc1 config
        i_fc1_synldr_reg_cfg_0  : in std_logic_vector(31 downto 0);
        i_fc1_nrnldr_reg_cfg_0  : in std_logic_vector(31 downto 0);
        i_fc1_lif_reg_cfg_0     : in std_logic_vector(31 downto 0);
        i_fc1_nrnwrt_reg_cfg_0  : in std_logic_vector(31 downto 0);

        -- fc2 config
        i_fc2_synldr_reg_cfg_0  : in std_logic_vector(31 downto 0);
        i_fc2_nrnldr_reg_cfg_0  : in std_logic_vector(31 downto 0);
        i_fc2_lif_reg_cfg_0     : in std_logic_vector(31 downto 0);
        i_fc2_nrnwrt_reg_cfg_0  : in std_logic_vector(31 downto 0);

        -- event counters
        o_class_count_0         : out std_logic_vector(31 downto 0);
        o_class_count_1         : out std_logic_vector(31 downto 0);
        o_class_count_2         : out std_logic_vector(31 downto 0);
        o_class_count_3         : out std_logic_vector(31 downto 0);
        o_class_count_4         : out std_logic_vector(31 downto 0);
        o_class_count_5         : out std_logic_vector(31 downto 0);
        o_class_count_6         : out std_logic_vector(31 downto 0);
        o_class_count_7         : out std_logic_vector(31 downto 0);
        o_class_count_8         : out std_logic_vector(31 downto 0);
        o_class_count_9         : out std_logic_vector(31 downto 0);
        o_class_count_10        : out std_logic_vector(31 downto 0)

    );

end FENRIR_TOP;

architecture behavior of FENRIR_TOP is

    -- fsm
    type state is (
        IDLE,
        CONFIG,
        RUN,
        INVALID
    );
    signal present_state        : state;
    signal next_state           : state;

    -- general
    signal reset                : std_logic;

    -- fc1
    signal fc1_en               : std_logic;
    signal fc1_in_we            : std_logic;
    signal fc1_in_wdata         : std_logic_vector(12 downto 0);
    signal fc1_out_we           : std_logic;
    signal fc1_out_wdata        : std_logic_vector(12 downto 0);
    signal fc1_busy             : std_logic;
    signal fc1_fifo_full        : std_logic;
    signal fc1_fifo_full_next   : std_logic;
    signal fc1_fifo_empty       : std_logic;

    -- fc2
    signal fc2_en               : std_logic;
    signal fc2_in_we            : std_logic;
    signal fc2_in_wdata         : std_logic_vector(12 downto 0);
    signal fc2_out_we           : std_logic;
    signal fc2_out_wdata        : std_logic_vector(12 downto 0);
    signal fc2_fifo_empty       : std_logic;
    signal fc2_fifo_full        : std_logic;
    signal fc2_fifo_fill_count  : std_logic_vector(7 downto 0);
    signal fc2_busy             : std_logic;

    -- output fifo
    signal out_fifo_empty       : std_logic;
    signal out_fifo_full        : std_logic;
    signal out_fifo_fill_count  : std_logic_vector(7 downto 0);
    signal out_fifo_re          : std_logic;
    signal out_fifo_rdata       : std_logic_vector(12 downto 0);
    signal out_fifo_rvalid      : std_logic;

    -- flags
    signal busy_flag            : std_logic;
    signal idle_flag            : std_logic;

begin

    busy_flag   <= fc1_busy or fc2_busy;
    idle_flag   <= fc2_fifo_empty and fc1_fifo_empty;
    o_flags     <= fc1_fifo_full_next & busy_flag & idle_flag;

    FIFO_IN_ADAPTER : entity work.FIFO_ADAPTER
    generic map (
        WIDTH           => 13
    )
    port map (
        i_clk           => i_sysclk,
        i_rst           => reset,
        i_ps_write      => i_ps_fifo,
        i_fifo_full     => fc1_fifo_full_next,
        o_fifo_we       => fc1_in_we,
        o_fifo_wdata    => fc1_in_wdata
    );

    FC1 : entity work.FC_LAYER
    generic map (
        IN_SIZE         => 3600,
        OUT_SIZE        => 64,
        OUT_FIFO_DEPTH  => 256,
        IS_LAST         => 0,
        SYN_MEM_WIDTH   => 32,
        BITS_PER_SYN    => 4,
        SYN_INIT_FILE   => "../data/FenrirFC_fc1.data",
        NRN_INIT_FILE   => ""
    )
    port map (
        -- config
        i_synldr_reg_cfg_0      => i_fc1_synldr_reg_cfg_0,
        i_nrnldr_reg_cfg_0      => i_fc1_nrnldr_reg_cfg_0,
        i_lif_reg_cfg_0         => i_fc1_lif_reg_cfg_0,
        i_nrnwrt_reg_cfg_0      => i_fc1_nrnwrt_reg_cfg_0,
        -- input
        i_in_fifo_we            => fc1_in_we,
        i_in_fifo_wdata         => fc1_in_wdata,
        -- output
        o_out_fifo_we           => fc1_out_we,
        o_out_fifo_wdata        => fc1_out_wdata,
        -- status
        o_in_fifo_empty         => fc1_fifo_empty,
        o_in_fifo_full          => fc1_fifo_full,
        o_in_fifo_fill_count    => open,
        o_in_fifo_full_next     => fc1_fifo_full_next,
        i_out_fifo_full         => fc2_fifo_full,
        i_out_fifo_empty        => fc2_fifo_empty,
        i_out_fifo_fill_count   => fc2_fifo_fill_count,
        o_busy                  => fc1_busy,
        -- control
        i_enable                => fc1_en,
        i_rst                   => reset,
        i_clk                   => i_sysclk,
        -- debug
        o_sched_tstep           => open,
        o_nrnmem_we             => open,
        o_nrnmem_waddr          => open,
        o_nrnmem_wdata          => open
    );

    FC2 : entity work.FC_LAYER
    generic map (
        IN_SIZE         => 64,
        OUT_SIZE        => 11,
        OUT_FIFO_DEPTH  => 256,
        IS_LAST         => 1,
        SYN_MEM_WIDTH   => 44,
        BITS_PER_SYN    => 4,
        SYN_INIT_FILE   => "../data/FenrirFC_fc2.data",
        NRN_INIT_FILE   => ""
    )
    port map (
        -- config
        i_synldr_reg_cfg_0      => i_fc2_synldr_reg_cfg_0,
        i_nrnldr_reg_cfg_0      => i_fc2_nrnldr_reg_cfg_0,
        i_lif_reg_cfg_0         => i_fc2_lif_reg_cfg_0,
        i_nrnwrt_reg_cfg_0      => i_fc2_nrnwrt_reg_cfg_0,
        -- input
        i_in_fifo_we            => fc1_out_we,
        i_in_fifo_wdata         => fc1_out_wdata,
        -- output
        o_out_fifo_we           => fc2_out_we,
        o_out_fifo_wdata        => fc2_out_wdata,
        -- status
        o_in_fifo_empty         => fc2_fifo_empty,
        o_in_fifo_full          => fc2_fifo_full,
        o_in_fifo_full_next     => open,
        o_in_fifo_fill_count    => fc2_fifo_fill_count,
        i_out_fifo_full         => '0',
        i_out_fifo_empty        => '1',
        i_out_fifo_fill_count   => "00000000",
        o_busy                  => fc2_busy,
        -- control
        i_enable                => fc2_en,
        i_rst                   => reset,
        i_clk                   => i_sysclk,
        -- debug
        o_sched_tstep           => open,
        o_nrnmem_we             => open,
        o_nrnmem_waddr          => open,
        o_nrnmem_wdata          => open
    );

        FC_OUTPUT_COUNTERS : entity work.FC_OUTPUT
        generic map (
            FIFO_WIDTH  => 13
        )
        port map (
            -- output fifo interface
            i_fifo_we       => fc2_out_we,
            i_fifo_wdata    => fc2_out_wdata,
            -- counter outputs
            o_class_count_0 => o_class_count_0,
            o_class_count_1 => o_class_count_1,
            o_class_count_2 => o_class_count_2,
            o_class_count_3 => o_class_count_3,
            o_class_count_4 => o_class_count_4,
            o_class_count_5 => o_class_count_5,
            o_class_count_6 => o_class_count_6,
            o_class_count_7 => o_class_count_7,
            o_class_count_8 => o_class_count_8,
            o_class_count_9 => o_class_count_9,
            i_clk           => i_sysclk,
            i_rst           => reset
        );

    reset_proc : process(i_sysclk)
    begin
        if rising_edge(i_sysclk) then
            if present_state = IDLE then
                reset   <= '1';
            else
                reset   <= '0';
            end if;
        end if;
    end process;

    -- FSM state register process
    state_reg : process(i_sysclk)
    begin
        if rising_edge(i_sysclk) then
            if (i_ctrl = "0000") then
                present_state <= IDLE;
            else
                present_state <= next_state;
            end if;
        end if;
    end process;

    -- FSM next state process
    nxt_state : process(present_state, i_ctrl)
    begin
        next_state <= present_state;
        case i_ctrl is
            when "0001" =>
                next_state <= IDLE;
            when "0010" =>
                next_state <= CONFIG;
            when "0011" =>
                next_state <= RUN;
            when others =>
                next_state <= INVALID;
        end case;
    end process;

    outputs : process(present_state)
    begin
        case present_state is    
            when IDLE =>
                o_led   <= "0001";
                fc1_en  <= '0';
                fc2_en  <= '0';
            when CONFIG =>
                o_led   <= "0010";
                fc1_en  <= '0';
                fc2_en  <= '0';
            when RUN =>
                o_led   <= "0011";
                fc1_en  <= '1';
                fc2_en  <= '1';
            when INVALID =>
                o_led <= "0000";
            when others =>
                fc1_en <= '0';
                fc2_en  <= '0';
        end case;
    end process;

end behavior;
