/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: synapse_loader.vhd
    Description: FSM for loading the *synaptic shotgun* with synapses. Address decoding for the
                 synapse memory.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------

    Configuration Registers:
        - (reg_cfg_0):
            - <11b> [10:0]  layer_size          : number of neurons in the layer
            - <11b> [21:11] layer_offset        : neuron address layer offset
            -  <2b> [23:22] syn_bits            : number of bits per synapse (2b, 4b, 8b)

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity SYNAPSE_LOADER is
    generic (
        SHOTGUN_DEPTH   : integer -- number of registers in the shotgun
        SYN_MEM_DEPTH   : integer -- depth of the synapse memory
        SYN_MEM_WIDTH   : integer -- width of the synapse memory
    );
    port (
        -- configuration interface
        i_cfg_en            : in std_logic;                         -- enable configuration
        i_cfg_addr          : in std_logic_vector(3 downto 0);      -- register to configure
        i_cfg_val           : in std_logic_vector(31 downto 0);     -- value to configure

        -- FIFO interface
        o_fifo_re           : out std_logic;                        -- read enable
        i_fifo_rvalid       : in std_logic;                         -- read valid
        i_fifo_rdata        : in std_logic_vector(31 downto 0);     -- read data
        i_fifo_empty        : in std_logic;                         -- FIFO empty flag
        i_fifo_empty_next   : in std_logic;                         -- FIFO empty next flag

        -- pack interface
        o_syn_weights       : out std_logic_vector((SHOTGUN_DEPTH * 8) - 1 downto 0);   -- synaptic weigh out to packer (8x8b)
        o_nrn_indices       : out std_logic_vector((SHOTGUN_DEPTH * 10 - 1 downto 0);   -- neuron indices out to packer (8x10b)
        i_halt              : in std_logic;                                             -- halt signal from packer

        -- synapse memory interface
        o_syn_addr          : out std_logic_vector(integer(ceil(log2(real(SYN_MEM_DEPTH))))-1 downto 0);
        i_syn_data          : in std_logic_vector(31 downto 0);     -- neuron data

        -- control signals
        i_start             : in std_logic;
        o_busy              : out std_logic;
        i_clk               : in std_logic;
        i_rst               : in std_logic;
    );
end SYNAPSE_LOADER;

architecture Behavioral of SYNAPSE_LOADER is
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
    signal cfg_layer_size       : std_logic_vector(10 downto 0);    -- number of neurons in the layer
    signal cfg_layer_offset     : std_logic_vector(10 downto 0);    -- neuron address layer offset
    signal cfg_syn_bits         : std_logic_vector(1 downto 0);     -- number of bits per synapse (2b, 4b, 8b, 16b)

    -- counters
    signal syn_counter          : integer range 0 to 1024 * 1024 / 2; -- synapse address counter

begin

    -- configuration decoding
    cfg_layer_size      <= reg_cfg_0(10 downto 0);
    cfg_layer_offset    <= reg_cfg_0(21 downto 11);
    cfg_syn_bits        <= reg_cfg_0(23 downto 22);


    config : process(i_clk, i_nrst) is
    begin
        if rising_edge(i_clk) then
            if cur_state = IDLE then
                if i_cfg_en = '1' then
                    case i_cfg_addr is
                        when "0000" => reg_cfg_0 <= i_cfg_val;
                    end case;
                end if;
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
                if i_start then
                    next_state <= GET_EVENT;
                end if;

            when GET_EVENT =>
                next_state <= ITERATE;

            when ITERATE =>
                if 
                next_state <= GET_EVENT;

        end case;
    end process;

    outputs : process(present_state, <inputs>) is
    begin
        case present_state is
            
            when IDLE =>
                o_busy          <= '0';
                o_fifo_re       <= '0';
                o_syn_addr      <= (others => '0');
                o_syn_weights   <= (others => '0');
                o_nrn_indices   <= (others => '0');
                syn_counter     <= 0;

            when GET_EVENT =>
                o_busy          <= '1';
                o_fifo_re       <= '1';
                o_syn_addr      <= (others => '0');
                o_syn_weights   <= (others => '0');
                o_nrn_indices   <= (others => '0');
                syn_counter     <= 0;

            when ITERATE =>
                o_busy          <= '1';
                o_fifo_re       <= '1';
                o_syn_addr      <= 

        end case;
    end process;

end Behavioral;
