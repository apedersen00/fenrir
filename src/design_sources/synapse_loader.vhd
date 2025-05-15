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
--  INST_NAME : entity work.SYNAPSE_LOADER
--  generic map (
--      SYN_MEM_DEPTH   =>
--      SYN_MEM_WIDTH   =>
--  )
--  port map (
--      i_cfg_en            =>
--      i_cfg_addr          =>
--      i_cfg_val           =>
--      o_fifo_re           =>
--      i_fifo_rvalid       =>
--      i_fifo_rdata        =>
--      o_syn_weight        =>
--      o_syn_valid         =>
--      o_syn_valid_next    =>
--      o_syn_valid_last    =>
--      i_syn_goto_idle     =>
--      o_syn_addr          =>
--      i_syn_data          =>
--      i_start             =>
--      i_continue          =>
--      o_busy              =>
--      i_clk               =>
--      i_rst               =>
--  );

entity SYNAPSE_LOADER is
    generic (
        SYN_MEM_DEPTH   : integer; -- depth of the synapse memory
        SYN_MEM_WIDTH   : integer  -- width of the synapse memory
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

        -- LIF interface
        o_syn_weight        : out std_logic_vector(7 downto 0);     -- synapse weight
        o_syn_valid         : out std_logic;                        -- valid weight on ouptut
        o_syn_valid_next    : out std_logic;                        -- next weight valid on output
        o_syn_valid_last    : out std_logic;
        i_goto_idle     : in std_logic;

        -- synapse memory interface
        o_syn_addr          : out std_logic_vector(integer(ceil(log2(real(SYN_MEM_DEPTH))))-1 downto 0);
        i_syn_data          : in std_logic_vector(31 downto 0);     -- neuron data

        -- control signals
        i_start             : in std_logic;
        i_continue          : in std_logic;
        o_busy              : out std_logic;
        i_clk               : in std_logic;
        i_rst               : in std_logic
    );
end SYNAPSE_LOADER;

architecture Behavioral of SYNAPSE_LOADER is
    -- fsm
    type state is (
        IDLE,
        GET_EVENT,
        GET_WEIGHTS,
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
    signal cfg_syn_bits         : std_logic_vector(1 downto 0);     -- number of bits per synapse (2b, 4b, 8b, 16b)

    -- counters
    signal counter_enable       : std_logic;
    signal counter_reset        : std_logic;
    -- maximum value should be maximum number of neurons in layer
    signal syn_index            : integer range 0 to 1024;
    -- maximum value should be max number of neurons in layer / min number of weights per address
    signal syn_addr_cntr        : integer range 0 to 512;

    -- constants
    signal weights_per_addr     : integer range 0 to 16;
    signal bits_per_weight      : integer range 0 to 8;

begin

    -- configuration decoding
    cfg_layer_size      <= reg_cfg_0(10 downto 0);
    cfg_layer_offset    <= reg_cfg_0(21 downto 11);
    cfg_syn_bits        <= reg_cfg_0(23 downto 22);
    
    addr_decoding : process(i_clk)
    begin
        -- TODO: Fix the number of bits used for syn_addr_cntr
        -- since syn_mem must know for instantation the size of o_syn_addr could be used. 
        if rising_edge(i_clk) then
            o_syn_addr <= i_fifo_rdata(9 downto 0) & std_logic_vector(to_unsigned(syn_addr_cntr, 1));
        end if;
    end process;

    -- determine how many weights per address
    cfg_decode : process(cfg_syn_bits)
    begin
        case cfg_syn_bits is
            -- 2 bits per synapse
            when "00"   =>
                weights_per_addr <= 16;
                bits_per_weight  <= 2;

            -- 4 bits per synapse
            when "01"   =>
                weights_per_addr <= 8;
                bits_per_weight  <= 4;

            -- 8 bits per synapse
            when "10"   =>
                weights_per_addr <= 4;
                bits_per_weight  <= 8;

            when others =>
                weights_per_addr <= 0;
                bits_per_weight  <= 0;
        end case;
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

            o_syn_valid <= ('1' and not i_goto_idle) when present_state = ITERATE else '0';

            -- if fetching in BRAM the next immediate value is always valid
            if (present_state = WAIT_FOR_BRAM) then
                o_syn_valid_next <= '1';
            elsif (present_state = ITERATE) then
                if (syn_index /= 0) and ((syn_index + 1) mod weights_per_addr = 0) then
                    o_syn_valid_next <= '0';
                elsif (syn_index /= 0) and ((syn_index + 1) >= unsigned(cfg_layer_size)) then
                    o_syn_valid_next <= '0';
                else
                    o_syn_valid_next <= '1';
                end if;
            else
                o_syn_valid_next <= '0';
            end if;

            if (present_state = ITERATE) then
                if (syn_index /= 0) and (syn_index + 1 >= unsigned(cfg_layer_size)) then
                    o_syn_valid_last <= '1';
                else
                    o_syn_valid_last <= '0';
                end if;
            else
                o_syn_valid_last <= '0';
            end if;

        end if;
    end process;

    -- output multiplexer
    output_mux : process(i_clk)
        variable v_word_index : integer;
    begin
        if rising_edge(i_clk) then
            if weights_per_addr /= 0 then
                -- wrap around syn_index so we always extract one of the weights per address
                v_word_index    := syn_index mod weights_per_addr;
                o_syn_weight    <= (others => '0');
                o_syn_weight(bits_per_weight - 1 downto 0) <=
                    i_syn_data((v_word_index + 1) * bits_per_weight - 1 downto v_word_index * bits_per_weight);
            else
                o_syn_weight        <= (others => '0');
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
                    next_state <= GET_EVENT;
                end if;

            when GET_EVENT =>
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

    -- FSM output process
    outputs : process(i_clk)
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
