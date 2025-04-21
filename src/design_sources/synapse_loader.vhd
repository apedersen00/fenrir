/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: synapse_loader.vhd
    Description: FSM for loading the *synaptic shotgun* with synapses.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------

    Configuration Registers:
        - (reg_cfg_0):
            - <11b> [10:0]  layer_size          : number of neurons in the layer
            - <11b> [21:11] layer_offset        : neuron address layer offset
            -  <3b> [24:22] syn_bits            : number of bits per synapse

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity neuron_loader is
    generic (
        SHOTGUN_NUM_REG : integer -- number of registers in the shotgun
        SYN_MEM_DEPTH   : integer -- depth of the synapse memory
    );
    port (
        -- configuration interface
        i_cfg_en    : in std_logic;                         -- enable configuration
        i_cfg_addr  : in std_logic_vector(3 downto 0);      -- register to configure
        i_cfg_val   : in std_logic_vector(31 downto 0);     -- value to configure

        -- FIFO interface
        o_re        : out std_logic;                        -- read enable
        i_rvalid    : in std_logic;                         -- read valid
        i_rdata     : in std_logic_vector(31 downto 0);     -- read data
        i_empty     : in std_logic;                         -- FIFO empty flag
        i_empty_next: in std_logic;                         -- FIFO empty next flag

        -- pack interface
        o_reg_addr  : out std_logic_vector(3 downto 0);     -- shotgun register address
        o_reg_out   : out std_logic_vector(31 downto 0);    -- shotgun register value
        o_primed    : out std_logic;                        -- shotgun primed signal
        i_fired     : in std_logic;                         -- shotgun fired signal

        -- synapse memory interface
        o_syn_addr  : out std_logic_vector(integer(ceil(log2(real(SYN_MEM_DEPTH))))-1 downto 0);
        i_syn_data  : in std_logic_vector(31 downto 0);     -- neuron data

        -- control signals
        i_start     : in std_logic;                         -- start signal
        o_busy      : out std_logic;                        -- busy signal

        i_clk       : in std_logic;
        i_rst       : in std_logic;

        o_fault     : out std_logic
    );
end neuron_loader;

architecture Behavioral of neuron_loader is
    -- fsm
    type state is (
        IDLE,
        GET_EVENT,
        ITERATE
    );
    signal present_state        : state;
    signal next_state           : state;

    -- registers
    signal reg_cfg_0            : std_logic_vector(31 downto 0);    -- configuration register 0

    -- configuration
    signal cfg_layer_size      : std_logic_vector(10 downto 0);    -- number of neurons in the layer
    signal cfg_layer_offset    : std_logic_vector(10 downto 0);    -- neuron address layer offset

    signal syn_index           : integer range 0 to 1023;

begin

    -- configuration decoding
    cfg_layer_size      <= reg_cfg_0(10 downto 0);
    cfg_layer_offset    <= reg_cfg_0(21 downto 11);

    process(i_clk, i_nrst) is
    begin
        if rising_edge(i_clk) then
            if i_cfg_en = '1' then
                case i_cfg_addr is
                    when "0000" => reg_cfg_0 <= i_cfg_val;
                    when others => null;
                end case;
            end if;
        end if;
    end process;

    state_reg : process(i_clk) is
    begin
        if rising_edge(i_clk) then
            if i_nrst = '0' then
                present_state <= IDLE;
            else
                present_state <= next_state;
            end if;
        end if;
    end process;

    nxt_state : process(present_state, <inputs>) is
    begin
        case present_state is

            when IDLE =>
                if not i_empty then
                    next_state <= GET_EVENT;
                else
                    next_state <= IDLE;
                end if;

            when GET_EVENT =>
                if i_rvalid = '1' then
                    next_state <= ITERATE;
                else
                    next_state <= GET_EVENT;
                end if;

            when ITERATE =>
                if syn_count >= unsigned(cfg_layer_size) then
                    next_state <= GET_EVENT;
                else
                    next_state <= ITERATE;
                end if;

            when others =>
                next_state <= IDLE;

        end case;
    end process;

    outputs : process(present_state, <inputs>) is
    begin
        case present_state is
            -- insert output logic here
        end case;
    end process;

end Behavioral;
