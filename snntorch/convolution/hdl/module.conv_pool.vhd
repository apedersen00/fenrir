library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_pool_pkg.all;

entity conv_pool is
    generic(
        CHANNELS_IN : integer := 1;
        CHANNELS_OUT : integer := 1;
        BITS_PER_NEURON : integer := 6;
        BITS_PER_WEIGHT : integer := 6;
        IMG_WIDTH : integer := 32;
        IMG_HEIGHT : integer := 32;
        BITS_PER_COORD : integer := 8
    );
    port(
        -- Standard ports needed for digital components
        clk : in std_logic;
        rst_i : in std_logic;
        enable_i : in std_logic;

        -- Control signals
        timestep_i : in std_logic;

        -- Event signals
        event_fifo_empty_i : in std_logic;
        -- shape of event bus: [x_coord(8), y_coord(8), channel(1), ..., channel]
        event_fifo_bus_i : in std_logic_vector(2 * BITS_PER_COORD + CHANNELS_IN - 1 downto 0);
        event_fifo_read_o : out std_logic;

        -- pragma translate_off
        debug_main_state : out main_state_et;
        debug_next_state : out main_state_et;
        debug_last_state : out main_state_et;
        
        debug_timestep_pending : out std_logic;
        debug_current_event : out event_tensor_t;
        debug_event_valid : out std_logic;
        debug_main_state_vec, debug_next_state_vec, debug_last_state_vec : out std_logic_vector(2 downto 0)
        -- pragma translate_on

    );
end entity conv_pool;

architecture rtl of conv_pool is

    signal main_state, main_next_state, main_last_state : main_state_et := IDLE;
    signal timestep_pending : std_logic := '0';
    signal fifo_read_request : std_logic := '0';

    signal current_event : event_tensor_t := (
        x_coord => 0,
        y_coord => 0,
        channel => 0
    );
    signal event_valid : std_logic := '0';

begin

    -- signals for output ports
    event_fifo_read_o <= fifo_read_request;


    state_register_update : process (clk, rst_i)
    begin
        if rising_edge(clk) then
            if rst_i = '1' then
                main_state <= RESET;
            else
                main_state <= main_next_state;
            end if;
        end if;
    end process state_register_update;

    update_last_state : process (clk, rst_i)
    begin
        if rising_edge(clk) then
            if rst_i = '1' then
                main_last_state <= IDLE;
            elsif main_state /= PAUSE then
                main_last_state <= main_state;
            end if;
        end if;
    end process update_last_state;

    state_machine_control : process (all)
    begin

        main_next_state <= main_state; -- assign current state as next by default

        -- check for enable signal
        if enable_i = '0' then
            main_next_state <= PAUSE;
        else
            case main_state is
            when IDLE =>
            when EVENT_CONV => 
            when PAUSE => main_next_state <= main_last_state;
            when POOL =>
            when CONFIG =>
            WHEN RESET => main_next_state <= IDLE;
            end case;
        end if;

    end process state_machine_control;

    timestep_buffer : process (clk, rst_i, timestep_i)
    begin
    if rising_edge(clk) then
        if rst_i = '1' then 
            timestep_pending <= '0';
        else
            if timestep_i = '1' then
                timestep_pending <= '1';
            elsif main_state = POOL then
                timestep_pending <= '0';
            end if;
        end if;
    end if;
    end process timestep_buffer;

    fifo_read_control : process (clk, rst_i)
    begin
    if rising_edge(clk) then
        if rst_i = '1' then 
            fifo_read_request <= '0';
            event_valid <= '0';
        else
            
            fifo_read_request <= '0'; -- default to no read request
            event_valid <= '0'; -- default to no valid event

            if main_state = IDLE and event_fifo_empty_i = '0' then
                fifo_read_request <= '1'; -- request to read from FIFO
            end if;

            if fifo_read_request = '1' then
                current_event <= bus_to_event_tensor(
                    event_fifo_bus_i,
                    BITS_PER_COORD,
                    CHANNELS_IN
                );
                event_valid <= '1'; -- set event as valid
            else 
                event_valid <= '0';
            end if;
        end if;
    end if;
    end process fifo_read_control;
    -- pragma translate_off
    debug_main_state <= main_state;
    debug_next_state <= main_next_state;
    debug_last_state <= main_last_state;

    debug_timestep_pending <= timestep_pending;
    debug_current_event <= current_event;
    debug_event_valid <= event_valid;

    debug_main_state_vec <= state_to_slv(main_state);
    debug_next_state_vec <= state_to_slv(main_next_state);
    debug_last_state_vec <= state_to_slv(main_last_state);
    -- pragma translate_on


end architecture rtl;