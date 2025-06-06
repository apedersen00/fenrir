---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: synapse_loader.vhd
--  Description: FSM for loading the LIF logic with synapses. Address decoding for the
--               synapse memory.
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
--          -  <2b> [23:22] syn_bits            : number of bits per synapse (2b, 4b, 8b)
--
---------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

--  Instantiation Template:
--  INST_NAME : entity work.FC_SYNAPSE_LOADER
--  generic map (
--      SYN_MEM_DEPTH   =>
--      SYN_MEM_WIDTH   =>
--  )
--  port map (
--      -- configuration interface
--      i_reg_cfg_0     =>
--      -- FIFO interface
--      o_fifo_re       =>
--      i_fifo_rdata    =>
--      -- LIF interface
--      o_syn_weight    =>
--      o_syn_valid     =>
--      o_syn_valid_nex =>
--      o_syn_valid_las =>
--      i_goto_idle     =>
--      -- synapse memory interface
--      o_synmem_re     =>
--      o_synmem_raddr  =>
--      i_synmem_rdata  =>
--      -- control signals
--      i_start         =>
--      i_continue      =>
--      o_busy          =>
--      i_clk           =>
--      i_rst           =>
--  );

entity FC_SYNAPSE_LOADER is
    generic (
        SYN_MEM_DEPTH   : integer;
        SYN_MEM_WIDTH   : integer
    );
    port (
        -- configuration interface
        i_reg_cfg_0         : in std_logic_vector(31 downto 0);

        -- FIFO interface
        o_fifo_re           : out std_logic;
        i_fifo_rdata        : in std_logic_vector(11 downto 0);

        -- LIF interface
        o_syn_weight        : out std_logic_vector(7 downto 0);
        o_syn_valid         : out std_logic;
        o_syn_valid_next    : out std_logic;
        o_syn_valid_last    : out std_logic;
        i_goto_idle         : in std_logic;

        -- synapse memory interface
        o_synmem_re         : out std_logic;
        o_synmem_raddr      : out std_logic_vector(integer(ceil(log2(real(SYN_MEM_DEPTH))))-1 downto 0);
        i_synmem_rdata      : in std_logic_vector(SYN_MEM_WIDTH - 1 downto 0);     -- data from memory

        -- control signals
        i_start             : in std_logic;
        i_continue          : in std_logic;
        o_busy              : out std_logic;
        i_clk               : in std_logic;
        i_rst               : in std_logic
    );
end FC_SYNAPSE_LOADER;

architecture Behavioral of FC_SYNAPSE_LOADER is

    -- fsm
    type state is (
        IDLE,
        GET_EVENT,
        STORE_EVENT,
        GET_WEIGHTS,
        WAIT_FOR_BRAM,
        ITERATE
    );
    signal present_state        : state;
    signal next_state           : state;

    -- registers
    signal reg_event_buf        : std_logic_vector(11 downto 0);    -- buffers the event for address decoding

    -- configuration
    signal cfg_layer_size       : std_logic_vector(10 downto 0);    -- number of neurons in the layer
    signal cfg_layer_offset     : std_logic_vector(10 downto 0);    -- neuron address layer offset
    signal cfg_syn_bits         : std_logic_vector(1 downto 0);     -- number of bits per synapse (2b, 4b, 8b, 16b)

    -- counters
    signal counter_enable       : std_logic;
    signal counter_reset        : std_logic;
    -- maximum value should be maximum number of neurons in layer
    signal syn_index            : integer range 0 to 2047;
    -- maximum value should be max number of neurons in layer / min number of weights per address
    signal syn_addr_cntr        : integer range 0 to 2047;

    -- constants
    signal weights_per_addr     : integer range 0 to 2047;
    signal bits_per_weight      : integer range 0 to 2047;
    signal addr_per_event       : integer range 0 to 2047;

    constant SYN_MEM_ADDR_WIDTH : integer := integer(ceil(log2(real(SYN_MEM_DEPTH))));

    signal dbg_syn_state    : std_logic_vector(3 downto 0);
    signal dbg_o_weight     : std_logic_vector(7 downto 0);
    signal dbg_syn_index    : std_logic_vector(11 downto 0);
    signal dbg_syn_valid    : std_logic;
    signal dbg_syn_valid_next : std_logic;
    signal dbg_syn_valid_last : std_logic;

begin

    o_syn_valid_last <= dbg_syn_valid_last;
    o_syn_valid <= dbg_syn_valid;
    o_syn_valid_next <= dbg_syn_valid_next;
    o_syn_weight <= dbg_o_weight;
    dbg_syn_index <= std_logic_vector(to_unsigned(syn_index, dbg_syn_index'length));

    with present_state select dbg_syn_state <=
        "0000" when IDLE,
        "0001" when GET_EVENT,
        "0010" when STORE_EVENT,
        "0011" when GET_WEIGHTS,
        "0100" when WAIT_FOR_BRAM,
        "0101" when ITERATE;

    -- configuration decoding
    cfg_layer_size      <= i_reg_cfg_0(10 downto 0);
    cfg_layer_offset    <= i_reg_cfg_0(21 downto 11);
    cfg_syn_bits        <= i_reg_cfg_0(23 downto 22);

    -- 137 * 3600
    addr_decoding : process(i_clk)
    begin
        if rising_edge(i_clk) then
            o_synmem_raddr <= std_logic_vector(to_unsigned(to_integer(unsigned(reg_event_buf)) * addr_per_event + syn_addr_cntr, SYN_MEM_ADDR_WIDTH));
        end if;
    end process;

    cfg_decode : process(all)
    begin
        case cfg_syn_bits is
            -- 2 bits per synapse
            when "00"   =>
                weights_per_addr <= SYN_MEM_WIDTH / 2;
                bits_per_weight  <= 2;
                addr_per_event   <= to_integer(unsigned(cfg_layer_size)) / (SYN_MEM_WIDTH / 2);

            -- 4 bits per synapse
            when "01"   =>
                weights_per_addr <= SYN_MEM_WIDTH / 4;
                bits_per_weight  <= 4;
                addr_per_event   <= to_integer(unsigned(cfg_layer_size)) / (SYN_MEM_WIDTH / 4);

            -- 8 bits per synapse
            when "10"   =>
                weights_per_addr <= SYN_MEM_WIDTH / 8;
                bits_per_weight  <= 8;
                addr_per_event   <= to_integer(unsigned(cfg_layer_size)) / (SYN_MEM_WIDTH / 8);

            -- default to 4 bits per synapse
            when others =>
                weights_per_addr <= SYN_MEM_WIDTH / 4;
                bits_per_weight  <= 4;
                addr_per_event   <= to_integer(unsigned(cfg_layer_size)) / (SYN_MEM_WIDTH / 4);
        end case;
    end process;

    -- synapse counter
    syn_counter : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if counter_reset = '1' then
                syn_index       <= 0;
                syn_addr_cntr   <= 0;
            elsif counter_enable = '1' then
                syn_index       <= syn_index + 1;
                syn_addr_cntr   <= (syn_index + 1) / weights_per_addr;
            else
                syn_index       <= syn_index;
                syn_addr_cntr   <= syn_addr_cntr;
            end if;
        end if;
    end process;

    -- state signal generation
    state_sig : process(i_clk)
    begin
        if rising_edge(i_clk) then

            dbg_syn_valid <= ('1' and not i_goto_idle) when present_state = ITERATE else '0';

            -- if fetching in BRAM the next immediate value is always valid
            if (present_state = WAIT_FOR_BRAM) then
                dbg_syn_valid_next <= '1';
            elsif (present_state = ITERATE) then
                if (syn_index /= 0) and ((syn_index + 1) mod weights_per_addr = 0) then
                    dbg_syn_valid_next <= '0';
                elsif (syn_index /= 0) and ((syn_index + 1) >= unsigned(cfg_layer_size)) then
                    dbg_syn_valid_next <= '0';
                else
                    dbg_syn_valid_next <= '1';
                end if;
            else
                dbg_syn_valid_next <= '0';
            end if;

            if (present_state = ITERATE) then
                if (syn_index /= 0) and (syn_index + 1 >= unsigned(cfg_layer_size)) then
                    dbg_syn_valid_last <= '1';
                else
                    dbg_syn_valid_last <= '0';
                end if;
            else
                dbg_syn_valid_last <= '0';
            end if;

        end if;
    end process;

    -- output multiplexer
    output_mux : process(i_clk)
        variable v_word_index : integer range 0 to 2047;
        variable v_rev_index  : integer range 0 to 2047;
    begin
        if rising_edge(i_clk) then
            if (i_rst = '1') then
                o_synmem_re     <= '0';
                dbg_o_weight    <= (others => '0');
            elsif (weights_per_addr /= 0) then
                -- wrap around syn_index so we always extract one of the weights per address
                v_word_index    := syn_index mod weights_per_addr;
                v_rev_index     := weights_per_addr - 1 - v_word_index;

                o_synmem_re     <= '1';
                dbg_o_weight    <= (others => '0');
                dbg_o_weight(bits_per_weight - 1 downto 0) <=
                    i_synmem_rdata((v_rev_index + 1) * bits_per_weight - 1 downto v_rev_index * bits_per_weight);
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
                if i_start = '1' then
                    next_state <= GET_EVENT;
                end if;

            when GET_EVENT =>
                if (i_goto_idle = '1') then
                    next_state <= IDLE;
                else
                    next_state  <= STORE_EVENT;
                end if;

            when STORE_EVENT =>
                if (i_goto_idle = '1') then
                    next_state <= IDLE;
                else
                    next_state  <= GET_WEIGHTS;
                end if;

            when GET_WEIGHTS =>
                if (i_goto_idle = '1') then
                    next_state <= IDLE;
                else
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
                elsif (syn_index /= 0) and ((syn_index + 1) mod weights_per_addr = 0) then
                    next_state <= GET_WEIGHTS;
                end if;
        end case;
    end process;

    read_in_fifo : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                reg_event_buf <= (others => '0');
            else
                if present_state = STORE_EVENT then
                    reg_event_buf <= i_fifo_rdata;
                end if;
            end if;
        end if;
    end process;

    -- FSM output process
    outputs : process(all)
    begin
        case present_state is
            when IDLE =>
                o_busy          <= '0';
                o_fifo_re       <= '0';
                counter_enable  <= '0';
                counter_reset   <= '1';

            when GET_EVENT =>
                o_busy          <= '1';
                o_fifo_re       <= '1';
                counter_enable  <= '0';
                counter_reset   <= '1';

            when STORE_EVENT =>
                o_busy          <= '1';
                o_fifo_re       <= '0';
                counter_enable  <= '0';
                counter_reset   <= '0';

            when GET_WEIGHTS =>
                o_busy          <= '1';
                o_fifo_re       <= '0';
                counter_enable  <= '0';
                counter_reset   <= '0';

            when WAIT_FOR_BRAM =>
                o_busy          <= '1';
                o_fifo_re       <= '0';
                counter_enable  <= '0';
                counter_reset   <= '0';

            when ITERATE    =>
                o_busy          <= '1';
                o_fifo_re       <= '0';
                counter_enable  <= '1' when i_continue = '1' else '0';
                counter_reset   <= '0';
        end case;
    end process;

end Behavioral;
