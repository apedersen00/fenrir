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
        UPDATE_ALL_NEURON_TIMESTAMPS
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

    signal enable_conv_unit : std_logic;
    signal kernels : std_logic_vector(FEATURE_MAPS * KERNEL_BIT_WIDTH - 1 downto 0);
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
        kernels => kernels,
        neuron_reset_value => neuron_reset_value,
        neuron_threshold_value => neuron_threshold_value,
        leakage_param => leakage_param,
        timestamp_event => timestamp_event,
        spike_events => spike_events,
        event_happened_flag => event_happened_flag
    );
    
data_ready <= not fifo_empty;


process (clk)
begin

    CASE state is 
        WHEN IDLE => 
        WHEN PROCESS_EVENT => 
        WHEN UPDATE_ALL_NEURON_TIMESTAMPS =>
    END CASE;

end process;


end architecture fsm;
 