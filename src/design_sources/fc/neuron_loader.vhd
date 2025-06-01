---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: neuron_loader.vhd
--  Description: FSM for loading the LIF logic with neurons.
--  VHDL Version: VHDL-2008
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
--  INST_NAME : entity work.FC_NEURON_LOADER
--  generic map (
--      NRN_MEM_DEPTH   =>
--  )
--  port map (
--      i_cfg_en                =>
--      i_cfg_addr              =>
--      i_cfg_val               =>
--      o_nrn_re                =>
--      o_nrn_addr              =>
--      i_nrn_data              =>
--      i_out_fifo_fill_count   =>
--      o_nrn_state             =>
--      o_nrn_index             =>
--      o_nrn_valid             =>
--      o_nrn_valid_next        =>
--      o_nrn_valid_last        =>
--      i_start                 =>
--      i_continue              =>
--      o_busy                  =>
--      i_clk                   =>
--      i_rst                   =>
--  );

entity FC_NEURON_LOADER is
    generic (
        NRN_MEM_DEPTH   : integer;
        OUT_FIFO_DEPTH  : integer
    );
    port (
        -- configuration interface
        i_reg_cfg_0             : in std_logic_vector(31 downto 0);

        -- neuron memory interface
        o_nrn_re                : out std_logic;
        o_nrn_addr              : out std_logic_vector(integer(ceil(log2(real(NRN_MEM_DEPTH))))-1 downto 0);
        i_nrn_data              : in std_logic_vector(35 downto 0);

        -- fc layer output fifo
        i_out_fifo_fill_count   : in std_logic_vector(integer(ceil(log2(real(OUT_FIFO_DEPTH))))-1 downto 0);

        -- output
        o_nrn_state             : out std_logic_vector(11 downto 0);
        o_nrn_index             : out std_logic_vector(11 downto 0);
        o_nrn_valid             : out std_logic;
        o_nrn_valid_next        : out std_logic;
        o_nrn_valid_last        : out std_logic;

        -- control signals
        i_start                 : in std_logic;
        i_continue              : in std_logic;
        o_busy                  : out std_logic;
        i_goto_idle             : in std_logic;

        i_clk                   : in std_logic;
        i_rst                   : in std_logic
    );
end FC_NEURON_LOADER;

architecture Behavioral of FC_NEURON_LOADER is
    -- fsm
    type state is (
        IDLE,
        GET_NEURONS,
        WAIT_FOR_BRAM,
        ITERATE
    );
    signal present_state        : state;
    signal next_state           : state;

    -- status
    signal out_fifo_ready       : std_logic;

    -- configuration
    signal cfg_layer_size       : std_logic_vector(10 downto 0);    -- number of neurons in the layer

    -- counters
    signal counter_enable       : std_logic;
    signal counter_reset        : std_logic;
    signal nrn_index            : integer range 0 to 2047;
    signal nrn_addr_cntr        : integer range 0 to 1023;

    -- constants
    constant neurons_per_addr   : integer := 3;
    constant bits_per_neuron    : integer := 12;

begin

    -- configuration decoding
    cfg_layer_size      <= i_reg_cfg_0(10 downto 0);

    fifo_not_full : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if (to_integer(unsigned(i_out_fifo_fill_count)) < OUT_FIFO_DEPTH - 5) then
                out_fifo_ready <= '1';
            else
                out_fifo_ready <= '0';
            end if;
        end if;
    end process;

    nrn_index_out : process(i_clk)
    begin
        if rising_edge(i_clk) then
            o_nrn_index <= std_logic_vector(to_unsigned(nrn_index, o_nrn_index'length));
        end if;
    end process;

    addr_decoding : process(i_clk)
    begin
        if rising_edge(i_clk) then
            o_nrn_addr <= std_logic_vector(to_unsigned(nrn_addr_cntr, o_nrn_addr'length));
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

            o_nrn_valid <= ('1' and not i_goto_idle) when present_state = ITERATE else '0';

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
    nxt_state : process(all)
    begin
        case present_state is
            when IDLE =>
                if i_start = '1' then
                    next_state <= GET_NEURONS;
                end if;

            when GET_NEURONS =>
                if (i_goto_idle = '1') then
                    next_state <= IDLE;
                elsif (out_fifo_ready = '1') then
                    next_state  <= WAIT_FOR_BRAM;
                end if;

            when WAIT_FOR_BRAM =>
                if (i_goto_idle = '1') then
                    next_state <= IDLE;
                else
                    next_state <= ITERATE;
                end if;

            when ITERATE =>
                if (i_goto_idle = '1') then
                    next_state <= IDLE;
                elsif (nrn_index /= 0) and ((nrn_index + 1) mod neurons_per_addr = 0) then
                    next_state <= GET_NEURONS;
                end if;
        end case;
    end process;

    outputs: process(all)
    begin
        case present_state is
            when IDLE           =>
                o_busy          <= '0';
                counter_enable  <= '0';
                counter_reset   <= '1';
                o_nrn_re        <= '0';

            when GET_NEURONS    =>
                o_busy          <= '1';
                counter_enable  <= '0';
                counter_reset   <= '0';
                o_nrn_re        <= '1';

            when WAIT_FOR_BRAM  =>
                o_busy          <= '1';
                counter_enable  <= '0';
                counter_reset   <= '0';
                o_nrn_re        <= '1';

            when ITERATE        =>
                o_busy          <= '1';
                counter_enable  <= '1' when i_continue = '1' else '0';
                counter_reset   <= '0';
                o_nrn_re        <= '1';
        end case;
    end process;

end Behavioral;
