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
--  INST_NAME : entity work.MASTER_SCHEDULER
--  generic map (
--      NUM_LAYERS      =>
--  )
--  port map (
--      i_enable        =>
--      i_timestep      =>
--      i_fc1_busy      =>
--      i_fc2_busy      =>
--      i_fc3_busy      =>
--      i_fc1_full      =>
--      i_fc2_full      =>
--      i_fc3_full      =>
--      i_fc1_empty     =>
--      i_fc2_empty     =>
--      i_fc3_empty     =>
--      o_fc1_start     =>
--      o_fc2_start     =>
--      o_fc3_start     =>
--      o_fc1_timestep  =>
--      o_fc2_timestep  =>
--      o_fc3_timestep  =>
--      o_busy          =>
--      i_clk           =>
--      i_rst           =>
--  );

entity MASTER_SCHEDULER is
    generic (
        NUM_LAYERS      : integer
    );
    port (
        i_enable        : in std_logic;
        i_timestep      : in std_logic;
        i_fc1_busy      : in std_logic;
        i_fc2_busy      : in std_logic;
        i_fc3_busy      : in std_logic;
        i_fc1_full      : in std_logic;
        i_fc2_full      : in std_logic;
        i_fc3_full      : in std_logic;
        i_fc1_empty     : in std_logic;
        i_fc2_empty     : in std_logic;
        i_fc3_empty     : in std_logic;
        o_fc1_start     : out std_logic;
        o_fc2_start     : out std_logic;
        o_fc3_start     : out std_logic;
        o_fc1_timestep  : out std_logic;
        o_fc2_timestep  : out std_logic;
        o_fc3_timestep  : out std_logic;
        o_busy          : out std_logic;
        i_clk           : in std_logic;
        i_rst           : in std_logic
    );
end MASTER_SCHEDULER;

architecture Behavioral of MASTER_SCHEDULER is

    -- fsm
    type state is (
        IDLE,
        START,
        RUN,
        INIT_TIMESTEP,
        WAIT_FC2_EMPTY,
        START_TIMESTEP,
        TIMESTEP
    );
    signal present_state        : state;
    signal next_state           : state;

    signal fc1_busy             : std_logic;
    signal fc2_busy             : std_logic;
    signal fc3_busy             : std_logic;
    signal fc1_full             : std_logic;
    signal fc2_full             : std_logic;
    signal fc3_full             : std_logic;
    signal fc1_empty            : std_logic;
    signal fc2_empty            : std_logic;
    signal fc3_empty            : std_logic;

begin

    fc1_busy    <= i_fc1_busy;
    fc1_full    <= i_fc1_full;
    fc1_empty   <= i_fc1_empty;
    fc2_busy    <= i_fc2_busy  when NUM_LAYERS > 1 else '0';
    fc2_full    <= i_fc2_full  when NUM_LAYERS > 1 else '0';
    fc2_empty   <= i_fc2_empty when NUM_LAYERS > 1 else '1';
    fc3_busy    <= i_fc3_busy  when NUM_LAYERS > 2 else '0';
    fc3_full    <= i_fc3_full  when NUM_LAYERS > 2 else '0';
    fc3_empty   <= i_fc3_empty when NUM_LAYERS > 2 else '1';

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
                if (i_enable = '1' and i_rst = '0') then
                    if (i_timestep = '0') then
                        next_state <= START;
                    else
                        next_state <= INIT_TIMESTEP;
                    end if;
                end if;

            when START =>
                if  (fc1_busy = '1' or fc1_empty = '1') and
                    (fc2_busy = '1' or fc2_empty = '1') then
                    next_state <= RUN;
                end if;

            when RUN =>
                if (fc1_busy = '0' and fc2_busy = '0') then
                    next_state <= IDLE;
                end if;
            
            when INIT_TIMESTEP =>
                if  (fc1_busy = '0')                    and
                    (fc2_busy = '1' or fc2_empty = '1') then
                    next_state <= WAIT_FC2_EMPTY;
                end if;

            when WAIT_FC2_EMPTY =>
                if (fc2_empty = '1' and fc2_busy = '0') then
                    next_state <= START_TIMESTEP;
                end if;

            when START_TIMESTEP =>
                if  (fc1_busy = '1' or fc1_empty = '1') and
                    (fc2_busy = '1' or fc2_empty = '1') then
                    next_state <= TIMESTEP;
                end if;

            when TIMESTEP =>
                if (fc1_busy = '0' and fc2_busy = '0') then
                    next_state <= IDLE;
                end if;
        end case;
    end process;

    outputs : process(i_clk)
    begin

        o_fc1_start     <= '0';
        o_fc2_start     <= '0';
        o_fc3_start     <= '0';
        o_fc1_timestep  <= '0';
        o_fc2_timestep  <= '0';
        o_fc3_timestep  <= '0';

        case present_state is    
            when IDLE =>
                o_busy          <= '0';

            when START =>
                o_busy          <= '1';
                o_fc1_start     <= '1';
                o_fc2_start     <= '1';

            when RUN =>
                o_busy          <= '1';
            
            when INIT_TIMESTEP =>
                o_busy          <= '1';
                o_fc2_start     <= '1';

            when WAIT_FC2_EMPTY =>
                o_busy          <= '1';
                o_fc2_start     <= '1';
                
            when START_TIMESTEP =>
                o_busy          <= '1';
                o_fc1_start     <= '1';
                o_fc1_timestep  <= '1';
                o_fc2_start     <= '1';
                o_fc2_timestep  <= '1';

            when TIMESTEP =>
                o_busy          <= '1';
        end case;
    end process;

end Behavioral;
