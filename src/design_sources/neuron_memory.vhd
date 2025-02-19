library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity neuron_memory is
    port (
        clk : in std_logic;
        rst : in std_logic;
        neuron_address : in std_logic_vector(3 downto 0);
        
        param_leak_str : out std_logic_vector(6 downto 0);
        param_thr : out std_logic_vector(11 downto 0);

        state_core : out std_logic_vector(11 downto 0);
        state_core_next : in std_logic_vector(11 downto 0)
    );
end neuron_memory;

architecture Behavioral of neuron_memory is
begin
end Behavioral;