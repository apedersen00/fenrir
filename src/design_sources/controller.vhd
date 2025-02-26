library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity controller is
    port (
        clk             : in std_logic;
        rst             : in std_logic;
        neuron_address  : out std_logic_vector(7 downto 0);
        synapse_address : out std_logic_vector(15 downto 0);

        input_in        : in std_logic_vector(7 downto 0);
        input_out       : out std_logic
    );
end controller;

architecture Behavioral of controller is

    signal counter_neuron   : std_logic_vector(7 downto 0)  := (others => '0');
    signal counter_synapse  : std_logic_vector(7 downto 0)  := (others => '0');
    signal counter_input    : std_logic_vector(7 downto 0)  := (others => '0');

    variable last_neuron    : std_logic_vector(7 downto 0)  := "00110000"; --dec: 48 -- 
    variable last_synapse   : std_logic_vector(7 downto 0)  := "00110000";

begin
    process (clk)
    begin
        if rising_edge(clk) then

            if rst = '1' then

                counter_neuron <= (others => '0');
                counter_synapse <= (others => '0');
                counter_input <= (others => '0');

            elsif counter_neuron = last_neuron then

                counter_neuron <= (others => '0');
                counter_synapse <= (others => '0');
                counter_input <= (others => '0');

            elsif counter_synapse = last_synapse then

                counter_neuron <= counter_neuron + 1;
                counter_synapse <= (others => '0');
                
            else 

                neuron_address <= counter_neuron;
                synapse_address(7 downto 0) <= counter_synapse;
                synapse_address(15 downto 8) <= counter_neuron;
                input_out <= input_in(counter_input);
                counter_synapse <= counter_synapse + 1;
                
            end if;
        end if;
    end process;
end Behavioral;
