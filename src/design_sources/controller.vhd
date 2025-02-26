library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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

    signal counter_neuron  : unsigned(7 downto 0) := (others => '0');
    signal counter_synapse : unsigned(7 downto 0) := (others => '0');
    signal counter_input   : unsigned(7 downto 0) := (others => '0');

begin
    process (clk)
        constant last_neuron  : unsigned(7 downto 0) := to_unsigned(48, 8);
        constant last_synapse : unsigned(7 downto 0) := to_unsigned(48, 8);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                counter_neuron  <= (others => '0');
                counter_synapse <= (others => '0');
                counter_input   <= (others => '0');
                neuron_address  <= (others => '0');
                synapse_address <= (others => '0');
                input_out       <= '0';

            elsif counter_neuron = last_neuron then
                counter_neuron  <= (others => '0');
                counter_synapse <= (others => '0');
                counter_input   <= (others => '0');

            elsif counter_synapse = last_synapse then
                counter_neuron  <= counter_neuron + 1;
                counter_synapse <= (others => '0');

            else 
                neuron_address  <= std_logic_vector(counter_neuron);
                synapse_address(7 downto 0) <= std_logic_vector(counter_synapse);
                synapse_address(15 downto 8) <= std_logic_vector(counter_neuron);
                
                if to_integer(counter_input) < 8 then
                    input_out <= input_in(to_integer(counter_input));
                else
                    input_out <= '0';
                end if;

                counter_synapse <= counter_synapse + 1;
            end if;
        end if;
    end process;
end Behavioral;
