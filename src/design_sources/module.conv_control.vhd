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

    CONSTANT RAW_EVENT_X_WIDTH       :integer:= 10;
    CONSTANT RAW_EVENT_Y_WIDTH       :integer:= 8;
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
        PROCESS_EVENT,
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
        variable x : integer;
        variable y : integer;
        variable polarity : integer;
        variable timestamp : std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0);
    begin
        x := to_integer(unsigned(data_vector(0 to RAW_EVENT_X_WIDTH - 1)));
        y := to_integer(unsigned(data_vector(RAW_EVENT_X_WIDTH to RAW_EVENT_X_WIDTH + RAW_EVENT_Y_WIDTH - 1)));
        polarity := to_integer(unsigned(data_vector(RAW_EVENT_X_WIDTH + RAW_EVENT_Y_WIDTH to FIFO_IN_DATA_WIDTH - 1)));
        timestamp := data_vector(FIFO_IN_DATA_WIDTH - 1 downto FIFO_IN_DATA_WIDTH - TIMESTAMP_WIDTH);
        
        event.x := x;
        event.y := y;
        event.polarity := polarity;
        event.timestamp := timestamp;
    end procedure convert_vector_to_event;
    

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
        read_from_fifo : out std_logic -- signal to read from FIFO
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
    signal data_ready : std_logic;
    signal kernels : kernels_t := (others => (others => '0'));
    signal event : event_raw;

    signal dx : integer range -1 to 1;
    signal dy : integer range -1 to 1;

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

process (clk)
begin
    IF RISING_EDGE(CLK) THEN
    IF RESET = '1' then

    ELSE

    CASE state is 
        WHEN IDLE =>
            IF data_ready = '1' then
                read_from_fifo <= '1';
                -- enable ram
                ram_ena <= '1';
                ram_enb <= '1';
                ram_wea <= "0";
                ram_web <= "0";
                enable_conv_unit <= '1';

                state <= PROCESS_EVENT;

            END IF;
        WHEN PROCESS_EVENT => 

            read_from_fifo <= '0';
            for dy in -1 to 1 loop
            for dx in -1 to 1 loop
            
                -- calculate address for kernel
                ram_addra <= std_logic_vector(
                    to_unsigned((event.x + dx) * IMAGE_WIDTH + (event.y + dy), NEURON_ADDRESS_WIDTH)
                );

                

            end loop;
            end loop;


        WHEN UPDATE_ALL_NEURON_TIMESTAMPS =>
        WHEN INITIALIZE => 
    END CASE;

    END IF;
    END IF;

end process;


end architecture fsm;
 