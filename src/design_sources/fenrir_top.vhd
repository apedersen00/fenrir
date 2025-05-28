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
        sysclk  : in std_logic;
        ctrl    : in std_logic_vector(3 downto 0);
        led     : out std_logic_vector(3 downto 0);

        -- output spike counters
        spike_0 : out std_logic_vector(31 downto 0);
        spike_1 : out std_logic_vector(31 downto 0);
        spike_2 : out std_logic_vector(31 downto 0);

        -- PS FIFO
        ps_fifo : in std_logic_vector(31 downto 0)
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

    signal spike_0_reg  : unsigned(31 downto 0);
    signal spike_1_reg  : unsigned(31 downto 0);
    signal spike_2_reg  : unsigned(31 downto 0);
    signal clk_div      : unsigned(1 downto 0);

    signal fifo_we      : std_logic;
    signal fifo_wdata   : std_logic_vector(15 downto 0);
    signal fifo_re      : std_logic;
    signal fifo_rdata   : std_logic_vector(15 downto 0);
    signal fifo_empty   : std_logic;
    signal fifo_rvalid  : std_logic;

    signal reset        : std_logic;

begin

    TEST_ADAPTER : entity work.FIFO_ADAPTER
    generic map (
        WIDTH           => 16
    )
    port map (
        i_clk           => sysclk,
        i_rst           => reset,
        i_ps_write      => ps_fifo,
        o_fifo_we       => fifo_we,
        o_fifo_wdata    => fifo_wdata
    );

    TEST_FIFO : entity work.BRAM_FIFO
    generic map (
        DEPTH   => 256,
        WIDTH   => 16
    )
    port map (
        i_we            => fifo_we,
        i_wdata         => fifo_wdata,
        i_re            => fifo_re,
        o_rvalid        => fifo_rvalid,
        o_rdata         => fifo_rdata,
        o_empty         => fifo_empty,
        o_empty_next    => open,
        o_full          => open,
        o_full_next     => open,
        o_fill_count    => open,
        i_clk           => sysclk,
        i_rst           => reset
    );

    READ_FIFO : process(sysclk)
    begin
        if rising_edge(sysclk) then
            if (fifo_empty = '0') and (fifo_re = '0') then
                fifo_re         <= '1';
            elsif (fifo_re = '1') and (fifo_rvalid = '1') then
                spike_2         <= "0000000000000000" & fifo_rdata;
                fifo_re         <= '0';
            end if;
        end if;
    end process;

    -- Debugging
    spike_0 <= std_logic_vector(spike_0_reg);
    spike_1 <= std_logic_vector(spike_1_reg);
    -- spike_2 <= std_logic_vector(spike_2_reg);

    increment : process(sysclk)
    begin
        if rising_edge(sysclk) then
            if (ctrl = "0000") then
                spike_0_reg <= (others => '0');
                spike_1_reg <= (others => '0');
                spike_2_reg <= (others => '0');
                clk_div     <= (others => '0');
            else
                clk_div     <= clk_div + 1;
                spike_0_reg <= spike_0_reg + 1;

                if clk_div(0) = '0' then
                    spike_1_reg <= spike_1_reg + 1;
                end if;

                if clk_div = "00" then
                    spike_2_reg <= spike_2_reg + 1;
                end if;
            end if;
        end if;
    end process;

    reset_proc : process(sysclk)
    begin
        if rising_edge(sysclk) then
            if present_state = IDLE then
                reset <= '1';
            else
                reset <= '0';
            end if;
        end if;
    end process;

    -- FSM state register process
    state_reg : process(sysclk)
    begin
        if rising_edge(sysclk) then
            if (ctrl = "0000") then
                present_state <= IDLE;
            else
                present_state <= next_state;
            end if;
        end if;
    end process;

    -- FSM next state process
    nxt_state : process(sysclk)
    begin
        case ctrl is
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

    outputs : process(sysclk)
    begin
        case present_state is    
            when IDLE =>
                led <= "0001";
            when CONFIG =>
                led <= "0011";
            when RUN =>
                if fifo_empty = '1' then
                    led <= "0111";
                else
                    led <= "1111";
                end if;
            when INVALID =>
                led <= "0000";
        end case;
    end process;

end behavior;
