library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity controller is
    port (
        clk : in std_logic;
        rst : in std_logic;
        neuron_address : out std_logic_vector(3 downto 0);
        synapse_address : out std_logic_vector(3 downto 0)
    );
end controller;

architecture Behavioral of controller is
begin
end Behavioral;
