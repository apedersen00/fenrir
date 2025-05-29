---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: bram_multiplexer.vhd
--  Description: Multiplexer for PS/PL access to BRAM.
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
--  INST_NAME : entity work.BRAM_MUX
--  generic map (
--      RAM_WIDTH   =>
--      RAM_DEPTH   =>
--      PS_DEPTH    =>
--  )
--  port map (
--      i_clk               =>
--      i_sel               =>
--      -- BRAM interface
--      o_we                =>
--      o_waddr             =>
--      o_wdata             =>
--      o_re                =>
--      o_raddr             =>
--      i_rdata             =>
--      -- BRAM port A interface PS
--      i_ps_bram_addr_a    =>
--      i_ps_bram_clk_a     =>
--      i_ps_bram_wrdata_a  =>
--      o_ps_bram_rddata_a  =>
--      i_ps_bram_en_a      =>
--      i_ps_bram_rst_a     =>
--      i_ps_bram_we_a      =>
--      -- BRAM interface PL
--      i_pl_we             =>
--      i_pl_waddr          =>
--      i_pl_wdata          =>
--      i_pl_re             =>
--      i_pl_raddr          =>
--      o_pl_rdata          =>
--  );

entity BRAM_MUX is
    generic (
        RAM_WIDTH       : integer;
        RAM_DEPTH       : integer;
        SPLIT_FACTOR    : integer
    );
    port (
        i_clk               : in std_logic;
        i_sel               : in std_logic;
        i_rst               : in std_logic;

        -- BRAM interface
        o_we                : out std_logic;
        o_waddr             : out std_logic_vector(integer(ceil(log2(real(RAM_DEPTH)))) - 1 downto 0);
        o_wdata             : out std_logic_vector(RAM_WIDTH - 1 downto 0);
        o_re                : out std_logic;
        o_raddr             : out std_logic_vector(integer(ceil(log2(real(RAM_DEPTH)))) - 1 downto 0);
        i_rdata             : in std_logic_vector(RAM_WIDTH - 1 downto 0);

        -- BRAM port A interface PS
        i_ps_write          : in std_logic_vector(31 downto 0);

        -- BRAM interface PL
        i_pl_we             : in std_logic;
        i_pl_waddr          : in std_logic_vector(integer(ceil(log2(real(RAM_DEPTH)))) - 1 downto 0);
        i_pl_wdata          : in std_logic_vector(RAM_WIDTH - 1 downto 0);
        i_pl_re             : in std_logic;
        i_pl_raddr          : in std_logic_vector(integer(ceil(log2(real(RAM_DEPTH)))) - 1 downto 0);
        o_pl_rdata          : out std_logic_vector(RAM_WIDTH - 1 downto 0)
    );
end BRAM_MUX;

architecture behavior of BRAM_MUX is

    -- fsm
    type state is (
        IDLE,
        PL_PASSTHROUGH,
        PS_RST,
        PS_WRT
    );
    signal present_state        : state;
    signal next_state           : state;

    signal ps_reset         : std_logic;
    signal pendulum         : std_logic;
    signal pendulum_last    : std_logic;
    signal write_counter    : integer;

begin

    ps_reset    <= i_ps_write(RAM_WIDTH / SPLIT_FACTOR);
    pendulum    <= i_ps_write(RAM_WIDTH / SPLIT_FACTOR - 1);

    -- o_we        <= i_pl_we;
    -- o_waddr     <= i_pl_waddr;
    -- o_wdata     <= i_pl_wdata;
    -- o_re        <= i_pl_re;
    -- o_raddr     <= i_pl_raddr;
    -- o_pl_rdata  <= i_rdata;


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
                next_state <= PL_PASSTHROUGH when i_sel = '1' else PS_RST;
            when PL_PASSTHROUGH =>
                next_state <= PL_PASSTHROUGH when i_sel = '1' else PS_RST;
            when PS_RST =>
                next_state <= PS_WRT when ps_reset ='1' else PS_RST;
            when PS_WRT =>
                next_state <= PL_PASSTHROUGH when write_counter >= (RAM_DEPTH * SPLIT_FACTOR) else PS_WRT;
        end case;
    end process;

    outputs : process(i_clk)
    begin
        case present_state is    
            when IDLE =>
                o_we            <= '0';
                o_waddr         <= (others => '0');
                o_wdata         <= (others => '0');
                o_re            <= '0';
                o_raddr         <= (others => '0');
                o_pl_rdata      <= (others => '0');
                write_counter   <= 0;
            when PL_PASSTHROUGH =>
                o_we            <= i_pl_we;
                o_waddr         <= i_pl_waddr;
                o_wdata         <= i_pl_wdata;
                o_re            <= i_pl_re;
                o_raddr         <= i_pl_raddr;
                o_pl_rdata      <= i_rdata;
                write_counter   <= 0;
            when PS_RST =>
                o_we            <= '0';
                o_waddr         <= (others => '0');
                o_wdata         <= (others => '0');
                o_re            <= '0';
                o_raddr         <= (others => '0');
                o_pl_rdata      <= (others => '0');
                write_counter   <= 0;
                pendulum_last   <= pendulum;
        end case;
    end process;

end behavior;
