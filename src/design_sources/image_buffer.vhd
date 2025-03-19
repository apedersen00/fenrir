library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity INPUT_BUFFER is
    port (
        clk: in std_logic;
        data_in: in std_logic_vector(63 downto 0);
        data_out: out std_logic_vector(63 downto 0);
        write_enable: in std_logic;
        address: in std_logic_vector(7 downto 0)
    );
end entity INPUT_BUFFER;

architecture Behavioral of INPUT_BUFFER is
    type memory_type is array (0 to 31) of std_logic_vector(63 downto 0);
    signal memory: memory_type;
begin
    process(clk)
    begin
        if rising_edge(clk) then

            if write_enable = '1' then
                memory(to_integer(unsigned(address))) <= data_in;
            end if;

            data_out <= memory(to_integer(unsigned(address)));
        end if;
    end process;

    
end architecture Behavioral;
