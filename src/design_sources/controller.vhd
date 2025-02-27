library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity controller is
    port (

        clk             : in std_logic;
        -- Data
        input_vector    : in std_logic_vector(15 downto 0); -- 16-bit input
        input_select    : out std_logic_vector(3 downto 0); -- 4-bit selector for input

        -- neuron and synapse 
        neuron_address  : out std_logic_vector(7 downto 0); -- 8-bit address for neuron
        neuron_input    : in std_logic_vector(31 downto 0); -- 32-bit input for neuron

        synapse_address : out std_logic_vector(7 downto 0); -- 8-bit address for synapse
        synapse_in      : in std_logic_vector(31 downto 0); -- 32-bit input for synapse

        -- signals for active neuron
        param_leak_str  : out std_logic_vector(6 downto 0); -- leakage stength parameter
        param_thr       : out std_logic_vector(11 downto 0); -- neuron firing threshold parameter
        state_core      : out std_logic_vector(11 downto 0); -- core neuron state from SRAM
        syn_weight      : out std_logic_vector(3 downto 0); -- synaptic weight
        syn_event       : out std_logic; -- synaptic event trigger
        
        state_core_next : in std_logic_vector(11 downto 0); -- next core neuron state to SRAM
        spike_out       : in std_logic -- neuron spike event output

    );
end controller;

architecture Behavioral of controller is
    signal synapse_counter : integer := 0;
    signal neuron_counter : integer := 0;
    
begin

end Behavioral;
