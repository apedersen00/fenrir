---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: neuron_loader.vhd
--  Description: FSM for loading the LIF logic with neurons.
--
--  Author(s):
--      - A. Pedersen, Aarhus University
--      - A. Cherencq, Aarhus University
--
---------------------------------------------------------------------------------------------------
--
--  Configuration Registers:
--      - (reg_cfg_0):
--          - <11b> [10:0]  layer_size          : number of neurons in the layer
--          - <11b> [21:11] layer_offset        : neuron address layer offset
--
---------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

--  Instantiation Template:
--  INST_NAME : entity work.NEURON_LOADER
--  generic map (
--      SYN_MEM_DEPTH   =>
--      SYN_MEM_WIDTH   =>
--  )
--  port map (
--      i_cfg_en        =>
--      i_cfg_addr      =>
--      i_cfg_val       =>
--      o_nrn_re        =>
--      o_nrn_addr      =>
--      i_nrn_data      =>
--      o_nrn_state     =>
--      o_nrn_valid     =>
--      i_start         =>
--      o_busy          =>
--      i_clk           =>
--      i_rst           =>
--  );

entity NEURON_LOADER is
    generic (
        NRN_MEM_DEPTH   : integer; -- depth of the neuron memory
        NRN_MEM_WIDTH   : integer  -- width of the neuron memory
    );
    port (
        -- configuration interface
        i_cfg_en    : in std_logic;                         -- enable configuration
        i_cfg_addr  : in std_logic_vector(3 downto 0);      -- register to configure
        i_cfg_val   : in std_logic_vector(31 downto 0);     -- value to configure

        -- neuron memory interface
        o_nrn_re    : out std_logic;                        -- neuron memory read enable
        o_nrn_addr  : out std_logic_vector(integer(ceil(log2(real(NRN_MEM_DEPTH))))-1 downto 0);
        i_nrn_data  : in std_logic_vector(35 downto 0);     -- neuron memory data in (3x12b)

        -- output
        o_nrn_state : out std_logic_vector(11 downto 0);    -- multiplexed output for LIF
        o_nrn_valid : out std_logic;                        -- output valid

        -- control signals
        i_start     : in std_logic;                         -- start signal
        o_busy      : out std_logic;                        -- busy signal

        i_clk       : in std_logic;
        i_rst       : in std_logic;
    );
end NEURON_LOADER;

architecture Behavioral of NEURON_LOADER is
    -- fsm
    type state is (
        IDLE,
        GET_NEURONS,
        WAIT_FOR_BRAM,
        ITERATE
    );
    signal present_state        : state;
    signal next_state           : state;

    -- registers
    signal reg_cfg_0            : std_logic_vector(31 downto 0);    -- configuration register 0

    -- configuration
    signal cfg_layer_size       : std_logic_vector(10 downto 0);    -- number of neurons in the layer
    signal cfg_layer_offset     : std_logic_vector(10 downto 0);    -- neuron address layer offset

    -- counters
    signal counter_enable       : std_logic;
    signal counter_reset        : std_logic;
    signal nrn_index            : integer range 0 to 1024;
    signal nrn_addr_cntr        : integer range 0 to 512;
    signal fsm_counter          : integer range 0 to 1;

    -- constants
    signal neurons_per_addr     : integer range 0 to 3;

begin

    -- configuration decoding
    cfg_layer_size      <= reg_cfg_0(10 downto 0);
    cfg_layer_offset    <= reg_cfg_0(21 downto 11);

    -- configuration interface
    config : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                reg_cfg_0   <= (others => '0');
            elsif i_cfg_en = '1' then
                case i_cfg_addr is
                    when "0000" => reg_cfg_0 <= i_cfg_val;
                    when others => null;
                end case;
            end if;
        end if;
    end process;

    -- neuron counter
    nrn_counter : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if counter_reset = '1' then
                nrn_index       <= 0;
                nrn_addr_cntr   <= 0;
            elsif counter_enable = '1' then
                nrn_index       <= nrn_index + 1;
                nrn_addr_cntr   <= (nrn_index + 1) / neurons_per_addr;
            else
                nrn_index       <= nrn_index;
                syn_addr_cntr   <= syn_addr_cntr;
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
    nxt_state : process(present_state, i_start, syn_index)
    begin
        case present_state is

            when IDLE =>
                if i_start = '1' then
                    next_state <= GET_NEURONS;
                end if;

            when GET_NEURONS =>
                next_state  <= GET_WEIGHTS;

            when WAIT_FOR_BRAM =>
                next_state <= ITERATE;

            when ITERATE =>
                if syn_index >= unsigned(cfg_layer_size) - 1 then
                    next_state <= GET_EVENT;
                elsif (syn_index /= 0) and ((syn_index + 1) mod weights_per_addr = 0) then
                    next_state <= GET_WEIGHTS;
                end if;

        end case;
    end process;

    outputs: process(present_state, <inputs>) is
    begin
        case present_state is
            -- insert output logic here
        end case;
    end process;

end Behavioral;
