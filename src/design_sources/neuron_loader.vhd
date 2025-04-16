/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: neuron_loader.vhd
    Description: FSM for loading the *synaptic shotgun* with neurons.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------

    Configuration Registers:
        - (reg_cfg_0):
            - [3:0]     state_width         : width of the state vector
            - [4]       per_neuron_beta_en  : enable per neuron beta
            - [11:4]    common_beta         : beta value when per neuron beta is disabled
            - [12]      per_neuron_thr_en   : enable per neuron threshold
            - [23:12]   common_thr          : threshold value when per neuron threshold is disabled
        - (reg_cfg_1)
            - [10:0]    layer_size          : number of neurons in the layer
            - [21:11]   layer_offset        : neuron address layer offset

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neuron_loader is
    generic (
        SHOTGUN_NUM_REG : integer -- number of registers in the shotgun
    );
    port (
        -- configuration interface
        i_cfg_en    : in std_logic;                         -- enable configuration
        i_cfg_addr  : in std_logic_vector(3 downto 0);      -- register to configure
        i_cfg_val   : in std_logic_vector(31 downto 0);     -- value to configure

        -- shotgun interface
        o_reg_addr  : out std_logic_vector(3 downto 0);     -- shotgun register address
        o_reg_out   : out std_logic_vector(31 downto 0);    -- shotgun register value
        o_primed    : out std_logic;                        -- shotgun primed signal
        i_fired     : in std_logic;                         -- shotgun fired signal

        -- neuron memory interface
        o_nrn_addr  : out std_logic_vector(10 downto 0);    -- neuron address
        o_nrn_data  : out std_logic_vector(31 downto 0);    -- neuron data
        o_nrn_en    : out std_logic;                        -- enable neuron memory access

        -- control signals
        i_start     : in std_logic;                         -- start signal
        o_busy      : out std_logic;                        -- busy signal

        i_clk       : in std_logic;
        i_nrst      : in std_logic;

        o_fault     : out std_logic
    );
end neuron_loader;

architecture Behavioral of neuron_loader is
    -- fsm
    type state is (
        IDLE,
        RUN,
        WAIT
    );
    signal present_state        : state;
    signal next_state           : state;

    -- registers
    signal reg_cfg_0            : std_logic_vector(31 downto 0);    -- configuration register 0
    signal reg_cfg_1            : std_logic_vector(31 downto 0);    -- configuration register 1
    signal reg_shotgun_counter  : std_logic_vector(3 downto 0);     -- shotgun counter

    -- configuration
    signal cfg_state_width      : std_logic_vector(3 downto 0);     -- state width
    signal cfg_per_neuron_beta  : std_logic_vector(1 downto 0);     -- per neuron beta enable
    signal cfg_common_beta      : std_logic_vector(7 downto 0);     -- common beta value
    signal cfg_per_neuron_thr   : std_logic_vector(1 downto 0);     -- per neuron threshold enable
    signal cfg_common_thr       : std_logic_vector(11 downto 0);    -- common threshold value
    signal cfg_layer_size       : std_logic_vector(10 downto 0);    -- layer size
    signal cfg_layer_offset     : std_logic_vector(10 downto 0);    -- layer offset

begin

    -- configuration decoding
    cfg_state_width     <= reg_cfg_0(3 downto 0);
    cfg_per_neuron_beta <= reg_cfg_0(4 downto 3);
    cfg_common_beta     <= reg_cfg_0(11 downto 4);
    cfg_per_neuron_thr  <= reg_cfg_0(12 downto 11);
    cfg_common_thr      <= reg_cfg_0(23 downto 12);
    cfg_layer_size      <= reg_cfg_1(10 downto 0);
    cfg_layer_offset    <= reg_cfg_1(21 downto 11);

    process(i_clk, i_nrst) is
    begin
        if rising_edge(i_clk) then
            if i_cfg_en = '1' then
                case i_cfg_addr is
                    when "0000" => reg_cfg_0 <= i_cfg_val;
                    when "0001" => reg_cfg_1 <= i_cfg_val;
                    when others => null;
                end case;
            end if;
        end if;
    end process;

    state_reg: process(i_clk) is
    begin
        if rising_edge(i_clk) then
            if i_nrst = '0' then
                -- present_state <= initial_state
            else
                -- present_state <= next_state
            end if;
        end if;
    end process;

    nxt_state: process(present_state, <inputs>) is
    begin
        case present_state is
            -- insert FSM here
        end case;
    end process;

    outputs: process(present_state, <inputs>) is
    begin
        case present_state is
            -- insert output logic here
        end case;
    end process;

end Behavioral;
