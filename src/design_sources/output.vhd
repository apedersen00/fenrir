---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: output.vhd
--  Description: Module for counting output events from a FIFO.
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
--  INST_NAME : entity work.FC_OUTPUT
--  generic map (
--      FIFO_WIDTH  =>
--  )
--  port map (
--      -- output fifo interface
--      o_fifo_re       =>
--      i_fifo_rdata    =>
--      i_fifo_empty    =>
--      i_fifo_rvalid   =>
--      -- counter outputs
--      o_class_count_0 =>
--      o_class_count_1 =>
--      o_class_count_2 =>
--      o_class_count_3 =>
--      o_class_count_4 =>
--      o_class_count_5 =>
--      o_class_count_6 =>
--      o_class_count_7 =>
--      o_class_count_8 =>
--      o_class_count_9 =>
--      i_clk           =>
--      i_rst           =>
--  );

entity FC_OUTPUT is
    generic (
        FIFO_WIDTH  : integer
    );
    port (
        -- output fifo interface
        o_fifo_re           : out std_logic;
        i_fifo_rdata        : in std_logic_vector(FIFO_WIDTH - 1 downto 0);
        i_fifo_empty        : in std_logic;
        i_fifo_rvalid       : in std_logic;

        -- counter outputs
        o_class_count_0     : out std_logic_vector(31 downto 0);
        o_class_count_1     : out std_logic_vector(31 downto 0);
        o_class_count_2     : out std_logic_vector(31 downto 0);
        o_class_count_3     : out std_logic_vector(31 downto 0);
        o_class_count_4     : out std_logic_vector(31 downto 0);
        o_class_count_5     : out std_logic_vector(31 downto 0);
        o_class_count_6     : out std_logic_vector(31 downto 0);
        o_class_count_7     : out std_logic_vector(31 downto 0);
        o_class_count_8     : out std_logic_vector(31 downto 0);
        o_class_count_9     : out std_logic_vector(31 downto 0);

        i_clk               : in std_logic;
        i_rst               : in std_logic
    );
end FC_OUTPUT;

architecture Behavioral of FC_OUTPUT is
    -- fsm
    type state is (
        IDLE,
        GET_EVENT,
        WRITE_COUNTER
    );
    signal present_state    : state;
    signal next_state       : state;
    
    signal class_count_0    : unsigned(31 downto 0);
    signal class_count_1    : unsigned(31 downto 0);
    signal class_count_2    : unsigned(31 downto 0);
    signal class_count_3    : unsigned(31 downto 0);
    signal class_count_4    : unsigned(31 downto 0);
    signal class_count_5    : unsigned(31 downto 0);
    signal class_count_6    : unsigned(31 downto 0);
    signal class_count_7    : unsigned(31 downto 0);
    signal class_count_8    : unsigned(31 downto 0);
    signal class_count_9    : unsigned(31 downto 0);

begin

    o_class_count_0 <= std_logic_vector(class_count_0);
    o_class_count_1 <= std_logic_vector(class_count_1);
    o_class_count_2 <= std_logic_vector(class_count_2);
    o_class_count_3 <= std_logic_vector(class_count_3);
    o_class_count_4 <= std_logic_vector(class_count_4);
    o_class_count_5 <= std_logic_vector(class_count_5);
    o_class_count_6 <= std_logic_vector(class_count_6);
    o_class_count_7 <= std_logic_vector(class_count_7);
    o_class_count_8 <= std_logic_vector(class_count_8);
    o_class_count_9 <= std_logic_vector(class_count_9);

    incr_counters : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                class_count_0 <= (others => '0');
                class_count_1 <= (others => '0');
                class_count_2 <= (others => '0');
                class_count_3 <= (others => '0');
                class_count_4 <= (others => '0');
                class_count_5 <= (others => '0');
                class_count_6 <= (others => '0');
                class_count_7 <= (others => '0');
                class_count_8 <= (others => '0');
                class_count_9 <= (others => '0');
            else
                if (present_state = WRITE_COUNTER) and (i_fifo_rvalid = '1') then
                    case to_integer(unsigned(i_fifo_rdata)) is
                        when 0 =>
                            class_count_0 <= class_count_0 + 1;
                        when 1 =>
                            class_count_1 <= class_count_1 + 1;
                        when 2 =>
                            class_count_2 <= class_count_2 + 1;
                        when 3 =>
                            class_count_3 <= class_count_3 + 1;
                        when 4 =>
                            class_count_4 <= class_count_4 + 1;
                        when 5 =>
                            class_count_5 <= class_count_5 + 1;
                        when 6 =>
                            class_count_6 <= class_count_6 + 1;
                        when 7 =>
                            class_count_7 <= class_count_7 + 1;
                        when 8 =>
                            class_count_8 <= class_count_8 + 1;
                        when 9 =>
                            class_count_9 <= class_count_9 + 1;
                        when others =>
                            null;
                    end case;
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
                if i_fifo_empty = '0' then
                    next_state <= GET_EVENT;
                end if;

            when GET_EVENT =>
                next_state <= WRITE_COUNTER;

            when WRITE_COUNTER =>
                if i_fifo_rvalid = '1' then
                    next_state <= IDLE;
                end if;
        end case;
    end process;

    -- FSM output process
    outputs : process(all)
    begin
        case present_state is    

            when IDLE =>
                o_fifo_re <= '0';

            when GET_EVENT =>
                o_fifo_re <= '1';

            when WRITE_COUNTER =>
                o_fifo_re <= '0';
    
        end case;
    end process;

end Behavioral;
