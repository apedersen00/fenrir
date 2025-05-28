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

        -- BRAM 1 interface PS
        bram_addr_a    : in std_logic_vector(11 downto 0);
        bram_clk_a     : in std_logic;
        bram_wrdata_a  : in std_logic_vector(32 - 1 downto 0);
        bram_rddata_a  : out std_logic_vector(32 - 1 downto 0);
        bram_en_a      : in std_logic;
        bram_rst_a     : in std_logic;
        bram_we_a      : in std_logic_vector((32 / 8) - 1 downto 0)
    );

end FENRIR_TOP;

architecture behavior of FENRIR_TOP is

    ATTRIBUTE X_INTERFACE_INFO : string;
    ATTRIBUTE X_INTERFACE_INFO OF bram_addr_a    : SIGNAL IS "xilinx.com:interface:bram:1.0 BRAM_PORTA ADDR";
    ATTRIBUTE X_INTERFACE_INFO OF bram_clk_a     : SIGNAL IS "xilinx.com:interface:bram:1.0 BRAM_PORTA CLK";
    ATTRIBUTE X_INTERFACE_INFO OF bram_wrdata_a  : SIGNAL IS "xilinx.com:interface:bram:1.0 BRAM_PORTA DIN";
    ATTRIBUTE X_INTERFACE_INFO OF bram_rddata_a  : SIGNAL IS "xilinx.com:interface:bram:1.0 BRAM_PORTA DOUT";
    ATTRIBUTE X_INTERFACE_INFO OF bram_en_a      : SIGNAL IS "xilinx.com:interface:bram:1.0 BRAM_PORTA EN";
    ATTRIBUTE X_INTERFACE_INFO OF bram_rst_a     : SIGNAL IS "xilinx.com:interface:bram:1.0 BRAM_PORTA RST";
    ATTRIBUTE X_INTERFACE_INFO OF bram_we_a      : SIGNAL IS "xilinx.com:interface:bram:1.0 BRAM_PORTA WE";

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

    signal testmem_we       : std_logic;
    signal testmem_waddr    : std_logic_vector(7 downto 0);
    signal testmem_wdata    : std_logic_vector(31 downto 0);
    signal testmem_re       : std_logic;
    signal testmem_raddr    : std_logic_vector(7 downto 0);
    signal testmem_rdata    : std_logic_vector(31 downto 0);
    signal testmem_clk      : std_logic;

    signal synmem_we        : std_logic;
    signal synmem_waddr     : std_logic_vector(7 downto 0);
    signal synmem_wdata     : std_logic_vector(31 downto 0);
    signal synmem_re        : std_logic;
    signal synmem_raddr     : std_logic_vector(7 downto 0);
    signal synmem_rdata     : std_logic_vector(31 downto 0);

    signal bram_sel         : std_logic;

begin

    bram_sel <= '0' when present_state = CONFIG else '1';
    synmem_re       <=  '1';
    synmem_raddr    <=  std_logic_vector(to_unsigned(1, synmem_raddr'length));

    TEST_MEMORY : entity work.DUAL_PORT_BRAM
    generic map (
        DEPTH       => 256,
        WIDTH       => 32,
        FILENAME    => ""
    )
    port map (
        i_we        => testmem_we,
        i_waddr     => testmem_waddr,
        i_wdata     => testmem_wdata,
        i_re        => testmem_re,
        i_raddr     => testmem_raddr,
        o_rdata     => testmem_rdata,
        i_clk       => testmem_clk
    );

    BRAM_MUX_0 : entity work.BRAM_MUX
    generic map (
        RAM_WIDTH   => 32,
        RAM_DEPTH   => 256,
        PS_DEPTH    => 2048
    )
    port map (
        i_clk               => sysclk,
        i_sel               => bram_sel,
        -- BRAM interface
        o_we                => testmem_we,
        o_waddr             => testmem_waddr,
        o_wdata             => testmem_wdata,
        o_re                => testmem_re,
        o_raddr             => testmem_raddr,
        i_rdata             => testmem_rdata,
        o_clk               => testmem_clk,
        -- BRAM port A interface PS
        i_ps_bram_addr_a    => bram_addr_a,
        i_ps_bram_clk_a     => bram_clk_a,
        i_ps_bram_wrdata_a  => bram_wrdata_a,
        o_ps_bram_rddata_a  => bram_rddata_a,
        i_ps_bram_en_a      => bram_en_a,
        i_ps_bram_rst_a     => bram_rst_a,
        i_ps_bram_we_a      => bram_we_a,
        -- BRAM interface PL
        i_pl_we             => synmem_we, 
        i_pl_waddr          => synmem_waddr,
        i_pl_wdata          => synmem_wdata,
        i_pl_re             => synmem_re,
        i_pl_raddr          => synmem_raddr,
        o_pl_rdata          => synmem_rdata,
        i_pl_clk            => sysclk
    );

    -- READ_BRAM : process(sysclk)
    -- begin
    --     if rising_edge(sysclk) then
    --         if present_state = RUN then
    --             synmem_we       <= '0';
    --             synmem_waddr    <=  (others => '0');
    --             synmem_wdata    <=  (others => '0');
    --             synmem_re       <=  '1';
    --             synmem_raddr    <=  std_logic_vector(to_unsigned(1, synmem_raddr'length));
    --         else
    --             synmem_we       <= '0';
    --             synmem_waddr    <=  (others => '0');
    --             synmem_wdata    <=  (others => '0');
    --             synmem_re       <=  '0';
    --             synmem_raddr    <=  (others => '0');
    --         end if;
    --     end if;
    -- end process;

    -- Debugging
    spike_0 <= std_logic_vector(spike_0_reg);
    spike_1 <= std_logic_vector(spike_1_reg);
    spike_2 <= testmem_rdata;

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
                led <= "0111";
            when INVALID =>
                led <= "0000";
        end case;
    end process;

end behavior;
