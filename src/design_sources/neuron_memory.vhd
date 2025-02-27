library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neuron_bram is
    port(
        clk : in std_logic;
        we : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(7 downto 0);
        di : in std_logic_vector(31 downto 0);
        do : out std_logic_vector(31 downto 0)
    );
end neuron_bram;

architecture Behavioral of neuron_bram is
    type ram_type is array (255 downto 0) of std_logic_vector(31 downto 0);
    signal RAM : ram_type;
begin
    process(clk)
    begin
        if clk'event and clk = '1' then
            if en = '1' then
                if we = '1' then
                    RAM(to_integer(unsigned(addr))) <= di;
                end if;
            do <= RAM(to_integer(unsigned(addr)));
            end if;
        end if;
end process;

end Behavioral;