library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

PACKAGE conv_control_t IS 

    CONSTANT IMAGE_WIDTH            :integer:= 10;
    CONSTANT IMAGE_HEIGHT           :integer:= 10; 
    CONSTANT BITS_PER_NEURON        :integer:= 4;
    CONSTANT FEATURE_MAPS           :integer:= 2;
    CONSTANT TIMESTAMP_WIDTH        :integer:= 4;
    CONSTANT NEURON_ADDRESS_WIDTH   :integer:= 7;
    CONSTANT MEM_SIZE_OF_ADDRESS    :integer:=    BITS_PER_NEURON 
                                                * FEATURE_MAPS 
                                                + TIMESTAMP_WIDTH;

    CONSTANT NEURON_RESET_WIDTH      :integer:= 4;
    CONSTANT NEURON_THRESHOLD_WIDTH  :integer:= 4;
    CONSTANT KERNEL_BIT_WIDTH        :integer:= 4;
    CONSTANT KERNEL_SIZE             :integer:= 3 * 3; -- Simulating a 3x3 kernel window
    CONSTANT LEAKAGE_PARAM_WIDTH     :integer:= 4;
    CONSTANT TIME_SCALING_FACTOR     :integer:= 200;
    -- PLaceholder for params
    CONSTANT PARAM_NEURON_RESET      :integer:= 0;
    CONSTANT PARAM_NEURON_THRESHOLD  :integer:= 14;
    CONSTANT PARAM_LEAKAGE           :integer:= 0;

    CONSTANT RAW_EVENT_X_WIDTH       :integer:= 4 ;
    CONSTANT RAW_EVENT_Y_WIDTH       :integer:= 4;
    CONSTANT RAW_EVENT_POLARITY_WIDTH:integer:= 2;
    
    CONSTANT FIFO_IN_DATA_WIDTH      :integer:=   RAW_EVENT_X_WIDTH 
                                                + RAW_EVENT_Y_WIDTH 
                                                + RAW_EVENT_POLARITY_WIDTH 
                                                + TIMESTAMP_WIDTH;

    CONSTANT EVENT_OUT_X_WIDTH      :integer:= 10;
    CONSTANT EVENT_OUT_Y_WIDTH      :integer:= 8;
    CONSTANT FIFO_OUT_DATA_WIDTH    :integer:=    EVENT_OUT_X_WIDTH
                                                + EVENT_OUT_Y_WIDTH
                                                + TIMESTAMP_WIDTH;
    type main_states_t is (
        IDLE,
        PROCESS_EVENT_START,
        PROCESS_EVENT,
        DELAY,
        UPDATE_ALL_NEURON_TIMESTAMPS,
        INITIALIZE
    );

    type kernels_t is array (0 to KERNEL_SIZE - 1) of std_logic_vector(FEATURE_MAPS * KERNEL_BIT_WIDTH - 1 downto 0);

    type event_raw is record
        x : integer range 0 to IMAGE_WIDTH - 1;
        y : integer range 0 to IMAGE_HEIGHT - 1;
        polarity : integer range -1 to 1;
        timestamp : std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0);
    end record;

    procedure convert_vector_to_event(
        signal data_vector : in std_logic_vector(FIFO_IN_DATA_WIDTH - 1 downto 0);
        signal event : out event_raw
    );

    component mem_neuron_potentials
        port(
            clka    : in std_logic;
            clkb    : in std_logic;
            ena     : in std_logic;
            enb     : in std_logic;
            wea     : in std_logic_vector(0 downto 0);
            web     : in std_logic_vector(0 downto 0);
            addra   : in std_logic_vector(NEURON_ADDRESS_WIDTH - 1 downto 0);
            addrb   : in std_logic_vector(NEURON_ADDRESS_WIDTH - 1 downto 0);
            dina    : in std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
            dinb    : in std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
            douta   : out std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
            doutb   : out std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0)
        );
    end component;

END PACKAGE conv_control_t;

package body conv_control_t is 

    procedure convert_vector_to_event(
    signal data_vector : in std_logic_vector(FIFO_IN_DATA_WIDTH - 1 downto 0);
    signal event : out event_raw
) is
    constant X_MSB         : integer := FIFO_IN_DATA_WIDTH - 1;
    constant X_LSB         : integer := X_MSB - RAW_EVENT_X_WIDTH + 1;

    constant Y_MSB         : integer := X_LSB - 1;
    constant Y_LSB         : integer := Y_MSB - RAW_EVENT_Y_WIDTH + 1;

    constant POLARITY_MSB  : integer := Y_LSB - 1;
    constant POLARITY_LSB  : integer := POLARITY_MSB - RAW_EVENT_POLARITY_WIDTH + 1;

    constant TIMESTAMP_MSB : integer := POLARITY_LSB - 1;
    constant TIMESTAMP_LSB : integer := TIMESTAMP_MSB - TIMESTAMP_WIDTH + 1;
begin
    -- Proper signal assignment with correct slicing
    event.x <= to_integer(unsigned(data_vector(X_MSB downto X_LSB)));
    event.y <= to_integer(unsigned(data_vector(Y_MSB downto Y_LSB)));
    
    -- Assuming polarity is signed 2-bit value (range -1 to 1)
    event.polarity <= to_integer(signed(data_vector(POLARITY_MSB downto POLARITY_LSB)));
    
    event.timestamp <= data_vector(TIMESTAMP_MSB downto TIMESTAMP_LSB);
end procedure;
    

end package body conv_control_t;

use work.conv_control_t.all;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity conv_control is
    port(
        clk : in std_logic;
        reset : in std_logic;
        fifo_empty : in std_logic; -- just NOT this signal to see if data is ready. 
        data_from_fifo : in std_logic_vector(FIFO_IN_DATA_WIDTH - 1 downto 0);
        read_from_fifo : out std_logic; -- signal to read from FIFO
        init : in std_logic; -- signal to initialize the neuron potentials
        initialize_value : in std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0)
    );
end entity conv_control;

 architecture fsm of conv_control is

    signal ram_ena:     std_logic;
    signal ram_enb:     std_logic;
    signal ram_wea:     std_logic_vector(0 downto 0);
    signal ram_web:     std_logic_vector(0 downto 0);
    signal ram_addra:   std_logic_vector(NEURON_ADDRESS_WIDTH - 1 downto 0);
    signal ram_addrb:   std_logic_vector(NEURON_ADDRESS_WIDTH - 1 downto 0);
    signal ram_dina:    std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
    signal ram_dinb:    std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
    signal ram_douta:   std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
    signal ram_doutb:   std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);

    signal state : main_states_t := IDLE;
    signal next_state : main_states_t := IDLE;
    signal data_ready : std_logic;
    signal kernels : kernels_t := (others => (others => '0'));
    signal event : event_raw;

    signal dx : integer range -1 to 1;
    signal dy : integer range -1 to 1;
    signal counter : integer range 0 to KERNEL_SIZE - 1;

    signal init_counter : integer range 0 to IMAGE_HEIGHT * IMAGE_WIDTH - 1;

    signal enable_conv_unit : std_logic;
    signal kernels_for_conv_unit : std_logic_vector(KERNEL_BIT_WIDTH * FEATURE_MAPS - 1 downto 0);
    signal neuron_reset_value : std_logic_vector(NEURON_RESET_WIDTH - 1 downto 0);
    signal neuron_threshold_value : std_logic_vector(NEURON_THRESHOLD_WIDTH - 1 downto 0);
    signal leakage_param : std_logic_vector(LEAKAGE_PARAM_WIDTH - 1 downto 0);
    signal timestamp_event : std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0);
    signal spike_events : std_logic_vector(FEATURE_MAPS - 1 downto 0);
    signal event_happened_flag : std_logic;

begin
    -- port a for reading, port b for writing. 
    mem_neurons : mem_neuron_potentials
    port map(
        clka => clk,
        clkb => clk,
        ena  => ram_ena,
        enb  => ram_enb,
        wea  => ram_wea,
        web  => ram_web,
        addra => ram_addra,
        addrb => ram_addrb,
        dina  => ram_dina,
        dinb  => ram_dinb,
        douta => ram_douta,
        doutb => ram_doutb
    );
    conv_unit : entity work.conv_unit
    generic map(
        BITS_PER_NEURON => BITS_PER_NEURON,
        FEATURE_MAPS => FEATURE_MAPS,
        TIMESTAMP_WIDTH => TIMESTAMP_WIDTH,
        NEURON_RESET_WIDTH => NEURON_RESET_WIDTH,
        NEURON_THRESHOLD_WIDTH => NEURON_THRESHOLD_WIDTH,
        KERNEL_BIT_WIDTH => KERNEL_BIT_WIDTH,
        LEAKAGE_PARAM_WIDTH => LEAKAGE_PARAM_WIDTH,
        TIME_SCALING_FACTOR => TIME_SCALING_FACTOR
    )
    port map(
        enable => enable_conv_unit,
        input_data => ram_douta,
        output_data => ram_dinb,
        kernels => kernels_for_conv_unit,
        neuron_reset_value => neuron_reset_value,
        neuron_threshold_value => neuron_threshold_value,
        leakage_param => leakage_param,
        timestamp_event => timestamp_event,
        spike_events => spike_events,
        event_happened_flag => event_happened_flag
    );
    
data_ready <= not fifo_empty;
-- always convert the data vector to event.
convert_vector_to_event(data_from_fifo, event);
-- Connect doutb to our signal
ram_doutb <= ram_doutb;

process (clk)
begin
    IF RISING_EDGE(CLK) THEN
    IF RESET = '1' then

        -- Reset all signals
        ram_ena <= '0';
        ram_enb <= '0';
        ram_wea <= "0";
        ram_web <= "0";
        ram_addra <= (others => 'W');
        ram_addrb <= (others => 'W');
        ram_dina <= (others => 'W');
        kernels_for_conv_unit <= (others => 'W');
        dx <= 0;
        dy <= 0;
        read_from_fifo <= '0';
        enable_conv_unit <= '0';

        -- test values
        for i in 0 to KERNEL_SIZE - 1 loop
            kernels(i) <= std_logic_vector(to_unsigned(i, FEATURE_MAPS * KERNEL_BIT_WIDTH));
        end loop;
        neuron_reset_value <= std_logic_vector(to_unsigned(PARAM_NEURON_RESET, NEURON_RESET_WIDTH));
        neuron_threshold_value <= std_logic_vector(to_unsigned(PARAM_NEURON_THRESHOLD, NEURON_THRESHOLD_WIDTH));
        leakage_param <= std_logic_vector(to_unsigned(PARAM_LEAKAGE, LEAKAGE_PARAM_WIDTH));
        -- lets just put some value in the timestamp
        timestamp_event <= std_logic_vector(to_unsigned(3, TIMESTAMP_WIDTH));

        if init = '1' then
            STATE <= INITIALIZE;
        else 
            STATE <= IDLE;
        end if;
        
    ELSE
    CASE state is 
        WHEN IDLE =>
            IF data_ready = '1' then
                read_from_fifo <= '1';
                
                state <= PROCESS_EVENT_START;

            END IF;
            ram_wea <= "0";
            ram_web <= "0";
            ram_ena <= '0';
            ram_enb <= '0';
            ram_addra <= (others => 'W');
            ram_addrb <= (others => 'W');
            

        WHEN PROCESS_EVENT_START => 
            dy <= -1;
            dx <= -1;
            counter <= 0;
            read_from_fifo <= '0';
            
            state <= PROCESS_EVENT;
            
        WHEN DELAY => 

            STATE <= next_state;

        WHEN PROCESS_EVENT =>
            ram_ena <= '1';

            if counter < KERNEL_SIZE then
                ram_addra <= std_logic_vector(
                        to_unsigned(event.x + dx + (event.y + dy) * IMAGE_WIDTH, NEURON_ADDRESS_WIDTH)
                );
            else 
                ram_ena <= '0';
                ram_addra <= (others => 'W');
            end if;
            
            ram_addrb <= ram_addra;

            if counter > 0 and counter < KERNEL_SIZE then
                ram_enb <= '1';
                ram_web <= "1";
                kernels_for_conv_unit <= kernels(counter-1);
                enable_conv_unit <= '1';
            end if;
            
            -- update counters
            
            if counter = (KERNEL_SIZE + 1) THEN
                dx <= 0;
                dy <= 0;
                ram_enb <= '0';
                counter <= 0;
                kernels_for_conv_unit <= (others => 'W');
                enable_conv_unit <= '0';
                state <= idle;
            
            else
                if dx = 1 and dy = 1 then

                    dx <= 0;
                    dy <= 0;
                
                elsif dx = 1 then

                    dx <= -1;
                    dy <= dy + 1;

                else 

                    dx <= dx + 1;

                end if;

            counter <= counter + 1;

            end if;

        WHEN UPDATE_ALL_NEURON_TIMESTAMPS =>
        WHEN INITIALIZE => 
        
            ram_ena <= '1';
            ram_wea <= "1";

            ram_addra <= std_logic_vector(to_unsigned(init_counter, NEURON_ADDRESS_WIDTH));
            ram_dina <= initialize_value;

            init_counter <= init_counter + 1;
            IF init_counter = IMAGE_HEIGHT * IMAGE_WIDTH - 1 THEN
                ram_ena <= '0';
                ram_wea <= "0";
                init_counter <= 0;
                ram_addra <= (others => '0');
                state <= IDLE;
            END IF;


    END CASE;

    END IF;
    END IF;

end process;


end architecture fsm;
 