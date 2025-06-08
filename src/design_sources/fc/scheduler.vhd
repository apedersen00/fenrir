---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: scheduler.vhd
--  Description: Scheduler for the fully-connected network of FENRIR.
--  VHDL Version: VHDL-2008
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

--  Instantiation Template:
--  INST_NAME : entity work.FC_SCHEDULER
--  port map (
--      i_enable        =>
--      i_synldr_busy   =>
--      i_nrnldr_busy   =>
--      o_synldr_start  =>
--      o_nrnldr_start  =>
--      o_timestep      =>
--      o_write_timeste =>
--      -- input fifo
--      i_fifo_in_empty =>
--      o_fifo_re       =>
--      i_fifo_rdata    =>
--      i_re            =>
--      o_rdata         =>
--      -- output fifo
--      i_fifo_out_full =>
--      o_busy          =>
--      i_clk           =>
--      i_rst           =>
--  );

entity FC_SCHEDULER is
    port (
        i_enable            : in std_logic;
        i_synldr_busy       : in std_logic;
        i_nrnldr_busy       : in std_logic;

        o_synldr_start      : out std_logic;
        o_nrnldr_start      : out std_logic;
        o_timestep          : out std_logic;
        o_write_timestep    : out std_logic;

        -- input fifo
        i_fifo_in_empty     : in std_logic;
        o_fifo_re           : out std_logic;                        -- read enable fifo
        i_fifo_rdata        : in std_logic_vector(12 downto 0);     -- read from fifo, 1b tstep & 12b neuron index
        i_re                : in std_logic;                         -- read enable from synapse loader
        o_rdata             : out std_logic_vector(11 downto 0);    -- data to synapse loader, 12b neuron index

        -- output fifo
        i_fifo_out_full     : in std_logic;

        o_busy              : out std_logic;
        i_clk               : in std_logic;
        i_rst               : in std_logic
    );
end FC_SCHEDULER;

architecture Behavioral of FC_SCHEDULER is

    -- fsm
    type state is (
        IDLE,
        READ_FIFO,
        WAIT_FIFO,
        PROCESS_FIFO,
        PROCESS_EVENT,
        PROCESS_EVENT_BUSY,
        PROPAGATE_TIMESTEP,
        TIMESTEP,
        TIMESTEP_BUSY
    );
    signal present_state    : state;
    signal next_state       : state;

    signal event_buf        : std_logic_vector(12 downto 0);
    signal tstep_buf        : std_logic;

    -- debug
    signal dbg_synldr_start : std_logic;
    signal dbg_nrnldr_start : std_logic;
    signal dbg_timestep     : std_logic;
    signal dbg_state        : std_logic_vector(3 downto 0);
    signal dbg_synldr_busy  : std_logic;
    signal dbg_nrnldr_busy  : std_logic;
    signal dbg_rst          : std_logic;
    
begin

    o_timestep      <= dbg_timestep;
    o_nrnldr_start  <= dbg_nrnldr_start;
    o_synldr_start  <= dbg_synldr_start;
    dbg_synldr_busy <= i_synldr_busy;
    dbg_nrnldr_busy <= i_nrnldr_busy;
    dbg_rst         <= i_rst;

    with present_state select dbg_state <=
        "0000" when IDLE,
        "0001" when READ_FIFO,
        "0010" when WAIT_FIFO,
        "0011" when PROCESS_FIFO,
        "0100" when PROCESS_EVENT,
        "0101" when PROCESS_EVENT_BUSY,
        "0111" when PROPAGATE_TIMESTEP,
        "1000" when TIMESTEP,
        "1001" when TIMESTEP_BUSY;

    o_busy      <= i_synldr_busy or i_nrnldr_busy;
    tstep_buf   <= event_buf(12);

    read_interface : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if  (i_rst = '1') then
                o_rdata <= (others => '0');
            else
                if (i_re = '1') then
                    o_rdata <= event_buf(11 downto 0);
                else
                    o_rdata <= (others => '0');
                end if;
            end if;
        end if;
    end process;

    -- FSM state register process
    state_reg : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                present_state <= IDLE;
            else
                present_state <= next_state;
            end if;
        end if;
    end process;

    -- FSM next state process
    nxt_state : process(all)
    begin
        next_state <= present_state;
        case present_state is
            when IDLE =>
                if  (i_enable           = '1')  and
                    (i_fifo_in_empty    = '0')  and
                    (i_fifo_out_full    = '0')  and
                    (i_synldr_busy      = '0')  and
                    (i_nrnldr_busy      = '0')  then
                    next_state <= READ_FIFO;
                end if;

            when READ_FIFO =>
                next_state <= WAIT_FIFO;

            when WAIT_FIFO =>
                next_state <= PROCESS_FIFO;

            when PROCESS_FIFO =>
                if  (tstep_buf = '1') then
                    next_state <= PROPAGATE_TIMESTEP;
                else
                    next_state <= PROCESS_EVENT;
                end if;

            when PROCESS_EVENT =>
                if  (i_synldr_busy  = '1') and
                    (i_nrnldr_busy  = '1') then
                    next_state <= PROCESS_EVENT_BUSY;
                end if;
            
            when PROCESS_EVENT_BUSY =>
                if  (i_synldr_busy  = '0') and
                    (i_nrnldr_busy  = '0') then
                    next_state <= IDLE;
                end if;

            when PROPAGATE_TIMESTEP =>
                next_state <= TIMESTEP;

            when TIMESTEP =>
                if  (i_nrnldr_busy  = '1') then
                    next_state <= TIMESTEP_BUSY;
                end if;

            when TIMESTEP_BUSY =>
                if  (i_nrnldr_busy  = '0') then
                    next_state <= IDLE;
                end if;

            when others =>
                next_state <= IDLE;
        end case;
    end process;

    read_in_fifo : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                event_buf <= (others => '0');
            elsif present_state = WAIT_FIFO then
                event_buf <= i_fifo_rdata;
            end if;
        end if;
    end process;

    outputs : process(all)
    begin
        case present_state is    
            when IDLE =>
                dbg_timestep        <= '0';
                dbg_synldr_start    <= '0';
                dbg_nrnldr_start    <= '0';
                o_fifo_re           <= '0';
                o_write_timestep    <= '0';

            when READ_FIFO =>
                dbg_timestep        <= '0';
                dbg_synldr_start    <= '0';
                dbg_nrnldr_start    <= '0';
                o_fifo_re           <= '1';
                o_write_timestep    <= '0';

            when WAIT_FIFO =>
                dbg_timestep        <= '0';
                dbg_synldr_start    <= '0';
                dbg_nrnldr_start    <= '0';
                o_fifo_re           <= '0';
                o_write_timestep    <= '0';

            when PROCESS_FIFO =>
                dbg_timestep        <= '0';
                dbg_synldr_start    <= '0';
                dbg_nrnldr_start    <= '0';
                o_fifo_re           <= '0';
                o_write_timestep    <= '0';

            when PROCESS_EVENT =>
                dbg_timestep        <= '0';
                dbg_synldr_start    <= '1';
                dbg_nrnldr_start    <= '1';
                o_fifo_re           <= '0';
                o_write_timestep    <= '0';

            when PROCESS_EVENT_BUSY =>
                dbg_timestep        <= '0';
                dbg_synldr_start    <= '0';
                dbg_nrnldr_start    <= '0';
                o_fifo_re           <= '0';
                o_write_timestep    <= '0';

            when PROPAGATE_TIMESTEP =>
                dbg_timestep        <= '1';
                dbg_synldr_start    <= '0';
                dbg_nrnldr_start    <= '1';
                o_fifo_re           <= '0';
                o_write_timestep    <= '1';

            when TIMESTEP =>
                dbg_timestep        <= '1';
                dbg_synldr_start    <= '0';
                dbg_nrnldr_start    <= '1';
                o_fifo_re           <= '0';
                o_write_timestep    <= '0';

            when TIMESTEP_BUSY =>
                dbg_timestep        <= '1';
                dbg_synldr_start    <= '0';
                dbg_nrnldr_start    <= '0';
                o_fifo_re           <= '0';
                o_write_timestep    <= '0';
        end case;
    end process;

end Behavioral;
