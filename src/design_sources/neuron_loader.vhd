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
--      NRN_MEM_DEPTH   =>
--  )
--  port map (
--      i_cfg_en            =>
--      i_cfg_addr          =>
--      i_cfg_val           =>
--      o_nrn_re            =>
--      o_nrn_addr          =>
--      i_nrn_data          =>
--      o_nrn_state         =>
--      o_nrn_index         =>
--      o_nrn_valid         =>
--      o_nrn_valid_next    =>
--      o_nrn_valid_last    =>
--      i_start             =>
--      i_continue          =>
--      o_busy              =>
--      i_clk               =>
--      i_rst               =>
--  );

entity NEURON_LOADER is
    generic (
        NRN_MEM_DEPTH   : integer                           -- depth of the neuron memory
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
        o_nrn_state         : out std_logic_vector(11 downto 0);    -- multiplexed output for LIF
        o_nrn_index         : out std_logic_vector(11 downto 0);    -- index of outputtet neuron
        o_nrn_valid         : out std_logic;                        -- output valid
        o_nrn_valid_next    : out std_logic;                        -- next output is valid
        o_nrn_valid_last    : out std_logic;

        -- control signals
        i_start     : in std_logic;                         -- start signal
        i_continue  : in std_logic;                         -- continue iteration
        o_busy      : out std_logic;                        -- busy signal
        i_goto_idle : in std_logic;

        i_clk       : in std_logic;
        i_rst       : in std_logic
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

    -- constants
    signal neurons_per_addr     : integer range 0 to 3;
    signal bits_per_neuron      : integer range 0 to 12;

begin

    -- configuration decoding
    cfg_layer_size      <= reg_cfg_0(10 downto 0);
    cfg_layer_offset    <= reg_cfg_0(21 downto 11);
    
    -- constants
    neurons_per_addr    <= 3;
    bits_per_neuron     <= 12;

    o_nrn_index         <= std_logic_vector(to_unsigned(nrn_index, o_nrn_index'length));

    addr_decoding : process(i_clk)
    begin
        if rising_edge(i_clk) then
            o_nrn_addr <= std_logic_vector(to_unsigned(nrn_addr_cntr, o_nrn_addr'length));
        end if;
    end process;

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
                nrn_addr_cntr   <= nrn_addr_cntr;
            end if;
        end if;
    end process;

    -- state signal generation
    state_sig : process(i_clk)
    begin
        if rising_edge(i_clk) then

            o_nrn_valid <= '1' when present_state = ITERATE else '0';

            -- if fetching in BRAM the next immediate value is always valid
            if (present_state = WAIT_FOR_BRAM) then
                o_nrn_valid_next <= '1';
            elsif (present_state = ITERATE) then
                -- if the next neuron is the last in current fetch
                if (nrn_index /= 0) and ((nrn_index + 1) mod neurons_per_addr = 0) then
                    o_nrn_valid_next <= '0';
                -- if the next neuron is the last in layer
                elsif (nrn_index /= 0) and ((nrn_index + 1) >= unsigned(cfg_layer_size)) then
                    o_nrn_valid_next <= '0';
                else
                    o_nrn_valid_next <= '1';
                end if;
            else
                o_nrn_valid_next <= '0';
            end if;

            if (present_state = ITERATE) then
                if (nrn_index /= 0) and (nrn_index + 1 >= unsigned(cfg_layer_size)) then
                    o_nrn_valid_last <= '1';
                else
                    o_nrn_valid_last <= '0';
                end if;
            else
                o_nrn_valid_last <= '0';
            end if;

        end if;
    end process;

    -- output multiplexer
    output_mux : process(i_clk)
        variable v_word_index : integer;
    begin
        if rising_edge(i_clk) then
            if neurons_per_addr /= 0 then
                -- wrap around nrn_index so we always extract one of the neurons
                v_word_index    := nrn_index mod neurons_per_addr;
                o_nrn_state     <= (others => '0');
                o_nrn_state(bits_per_neuron - 1 downto 0) <=
                    i_nrn_data((v_word_index + 1) * bits_per_neuron - 1 downto v_word_index * bits_per_neuron);
            else
                o_nrn_state     <= (others => '0');
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
    nxt_state : process(i_clk)
    begin
        case present_state is

            when IDLE =>
                if i_start = '1' then
                    next_state <= GET_NEURONS;
                end if;

            when GET_NEURONS =>
                next_state  <= WAIT_FOR_BRAM;

            when WAIT_FOR_BRAM =>
                next_state <= ITERATE;

            when ITERATE =>
                if (i_goto_idle = '1') then
                    next_state <= IDLE;
                elsif (nrn_index /= 0) and ((nrn_index + 1) mod neurons_per_addr = 0) then
                    next_state <= GET_NEURONS;
                end if;

        end case;
    end process;

    outputs: process(i_clk)
    begin

        case present_state is    
            when IDLE =>
                o_busy          <= '0';
                counter_enable  <= '0';
                counter_reset   <= '1';
                o_nrn_re        <= '0';

            when GET_NEURONS =>
                o_busy          <= '1';
                counter_enable  <= '0';
                counter_reset   <= '0';
                o_nrn_re        <= '1';

            when WAIT_FOR_BRAM =>
                o_busy          <= '1';
                counter_enable  <= '0';
                counter_reset   <= '0';
                o_nrn_re        <= '0';

            when ITERATE    =>
                o_busy          <= '1';
                counter_enable  <= '1' when i_continue = '1' else '0';
                counter_reset   <= '0';
                o_nrn_re        <= '0';
        end case;
    end process;

end Behavioral;
