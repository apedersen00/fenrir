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
    CONSTANT MEM_SIZE_OF_ADDRESS    :integer:= BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH;

    CONSTANT NEURON_RESET_WIDTH      :integer:= 4;
    CONSTANT NEURON_THRESHOLD_WIDTH  :integer:= 4;
    CONSTANT KERNEL_BIT_WIDTH        :integer:= 4;
    CONSTANT LEAKAGE_PARAM_WIDTH     :integer:= 4;
    CONSTANT TIME_SCALING_FACTOR     :integer:= 200;

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

entity conv_control is
    port(
        clk : in std_logic;
        reset : in std_logic
    );
end entity conv_control;

architecture fsm of conv_control is

    signal state : main_states_t;
    
    signal ram_ena : std_logic;
    signal ram_enb : std_logic;
    signal ram_wea : std_logic_vector(0 downto 0);
    signal ram_web : std_logic_vector(0 downto 0);
    signal ram_addra : std_logic_vector(NEURON_ADDRESS_WIDTH - 1 downto 0);
    signal ram_addrb : std_logic_vector(NEURON_ADDRESS_WIDTH - 1 downto 0);
    signal ram_dina : std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
    signal ram_dinb : std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
    signal ram_douta : std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);
    signal ram_doutb : std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0);

    mem_neuron_potentials : mem_neuron_potentials
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

begin



end architecture fsm;
 