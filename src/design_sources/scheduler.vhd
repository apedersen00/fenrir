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
--  INST_NAME : entity work.SCHEDULER
--  port map (
--      i_enable        =>
--      i_timestep      =>
--      i_synldr_busy   =>
--      i_nrnldr_busy   =>
--      o_synldr_start  =>
--      o_nrnldr_start  =>
--      i_fifo_in_empty =>
--      i_fifo_out_full =>
--      o_busy          =>
--      i_clk           =>
--      i_rst           =>
--  );

entity SCHEDULER is
    port (
        i_enable        : in std_logic;
        i_timestep      : in std_logic;
        i_synldr_busy   : in std_logic;
        i_nrnldr_busy   : in std_logic;

        o_synldr_start  : out std_logic;
        o_nrnldr_start  : out std_logic;
        o_timestep      : out std_logic;

        -- input fifo
        i_fifo_in_empty : in std_logic;

        -- output fifo
        i_fifo_out_full : in std_logic;

        o_busy          : out std_logic;
        i_clk           : in std_logic;
        i_rst           : in std_logic
    );
end SCHEDULER;

architecture Behavioral of SCHEDULER is

    -- fsm
    type state is (
        IDLE,
        PROCESS_EVENT,
        TIMESTEP
    );
    signal present_state        : state;
    signal next_state           : state;

begin

    o_busy <= i_synldr_busy or i_nrnldr_busy;

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
    nxt_state : process(i_clk)
    begin
        case present_state is
            when IDLE =>
                if      (i_rst = '1') then
                    next_state <= present_state;
                elsif   (i_enable           = '1')  and
                        (i_timestep         = '1')  and 
                        (i_fifo_out_full    = '0')  and
                        (i_synldr_busy      = '0')  and
                        (i_nrnldr_busy      = '0')  then
                    next_state <= TIMESTEP;

                elsif   (i_enable           = '1')  and
                        (i_fifo_in_empty    = '0')  and
                        (i_fifo_out_full    = '0')  and
                        (i_synldr_busy      = '0')  and
                        (i_nrnldr_busy      = '0')  then
                    next_state <= PROCESS_EVENT;
                end if;

            when PROCESS_EVENT =>
                if      (i_rst = '1') then
                    next_state <= IDLE;
                elsif   (i_synldr_busy  = '1') and
                        (i_nrnldr_busy  = '1') then
                    next_state <= IDLE;
                end if;
            
            when TIMESTEP =>
                if      (i_rst = '1') then
                    next_state <= IDLE;
                elsif   (i_synldr_busy  = '1') and
                        (i_nrnldr_busy  = '1') then
                    next_state <= IDLE;
                end if;
        end case;
    end process;

    outputs : process(i_clk)
    begin
        case present_state is    
            when IDLE =>
                o_timestep      <= '0';
                o_synldr_start  <= '0';
                o_nrnldr_start  <= '0';

            when PROCESS_EVENT =>
                o_timestep      <= '0';
                o_synldr_start  <= '1';
                o_nrnldr_start  <= '1';

            when TIMESTEP =>
                o_timestep      <= '1';
                o_synldr_start  <= '1';
                o_nrnldr_start  <= '1';
        end case;
    end process;

end Behavioral;
