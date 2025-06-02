library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_pool_pkg.all;

entity event_capture is
    generic(
        BITS_PER_COORD : integer := 8;
        IMG_WIDTH      : integer := 32;
        IMG_HEIGHT     : integer := 32
    );
    port(
        -- Standard control signals
        clk             : in  std_logic;
        rst_i           : in  std_logic;
        enable_i        : in  std_logic;
        
        -- FIFO interface
        fifo_empty_i    : in  std_logic;
        fifo_bus_i      : in  std_logic_vector(2 * BITS_PER_COORD - 1 downto 0);
        fifo_read_o     : out std_logic;
        
        -- Data output interface
        data_valid_o    : out std_logic;
        data_out_o      : out vector2_t;
        data_consumed_i : in  std_logic;
        
        -- Debug signals
        debug_state_o   : out std_logic_vector(1 downto 0)
    );
end entity event_capture;

architecture rtl of event_capture is

    type capture_state_t is (
        IDLE,
        READ_REQUEST, 
        DATA_READY,
        RESET
    );
    
    signal current_state, next_state : capture_state_t := IDLE;
    signal captured_data : vector2_t := (x => 0, y => 0);
    signal data_valid_reg : std_logic := '0';

    -- Helper function to convert state to std_logic_vector for debug
    function state_to_slv(state : capture_state_t) return std_logic_vector is
    begin
        case state is
            when IDLE         => return "00";
            when READ_REQUEST => return "01";
            when DATA_READY   => return "10";
            when RESET        => return "11";
        end case;
    end function;

    -- Helper function to extract coordinates from bus
    function extract_coordinates(
        bus_data : std_logic_vector;
        bits_per_coord : integer
    ) return vector2_t is
        variable coords : vector2_t;
    begin
        -- Extract coordinates: [x_coord(MSBs)][y_coord(LSBs)]
        coords.y := to_integer(unsigned(bus_data(bits_per_coord - 1 downto 0)));
        coords.x := to_integer(unsigned(bus_data(2 * bits_per_coord - 1 downto bits_per_coord)));
        return coords;
    end function;

begin

    -- Output assignments
    data_valid_o <= data_valid_reg;
    data_out_o <= captured_data;
    debug_state_o <= state_to_slv(current_state);
    
    -- FIFO read signal - only high for one cycle during READ_REQUEST state
    fifo_read_o <= '1' when (current_state = READ_REQUEST) else '0';

    -- State register update process
    state_register : process(clk, rst_i)
    begin
        if rst_i = '1' then
            current_state <= RESET;
        elsif rising_edge(clk) then
            current_state <= next_state;
        end if;
    end process state_register;

    -- Next state logic
    state_machine : process(all)
    begin
        -- Default: stay in current state
        next_state <= current_state;
        
        if enable_i = '0' then
            -- When disabled, go to idle (or stay in reset if in reset)
            if current_state /= RESET then
                next_state <= IDLE;
            end if;
        else
            case current_state is
                when IDLE =>
                    -- Check if FIFO has data available
                    if fifo_empty_i = '0' then
                        next_state <= READ_REQUEST;
                    end if;
                    
                when READ_REQUEST =>
                    -- Always move to DATA_READY after one cycle
                    next_state <= DATA_READY;
                    
                when DATA_READY =>
                    -- Wait for top module to consume the data
                    if data_consumed_i = '1' then
                        next_state <= IDLE;
                    end if;
                    
                when RESET =>
                    -- Exit reset when enable is high
                    if enable_i = '1' then
                        next_state <= IDLE;
                    end if;
            end case;
        end if;
    end process state_machine;

    -- Data capture and valid flag management
    data_capture : process(clk, rst_i)
    begin
        if rst_i = '1' then
            captured_data <= (x => 0, y => 0);
            data_valid_reg <= '0';
        elsif rising_edge(clk) then
            -- Handle data capture - capture when transitioning FROM READ_request TO data_ready
            if current_state = READ_REQUEST and next_state = DATA_READY then
                -- Capture the data from FIFO bus (FIFO responds after 1 cycle)
                captured_data <= extract_coordinates(fifo_bus_i, BITS_PER_COORD);
            end if;
            
            -- Handle data valid flag
            if next_state = DATA_READY and current_state /= DATA_READY then
                -- Entering DATA_READY state
                data_valid_reg <= '1';
            elsif current_state = DATA_READY and next_state /= DATA_READY then
                -- Leaving DATA_READY state
                data_valid_reg <= '0';
            end if;
        end if;
    end process data_capture;

end architecture rtl;