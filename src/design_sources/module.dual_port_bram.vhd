library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity dual_port_bram is
    generic(
        DATA_WIDTH : integer := 128;
        ADDR_WIDTH : integer := 10;
        DEPTH      : integer := 1024
    );
    port(
        clk: in std_logic; -- common clock for both ports
        ena: in std_logic; -- enable for port A
        enb: in std_logic; -- enable for port B
        we_a: in std_logic; -- write enable for port A
        addr_a: in std_logic_vector(ADDR_WIDTH - 1 downto 0); -- address for port A
        din_a: in std_logic_vector(DATA_WIDTH - 1 downto 0); -- data input for port A
        dout_a: out std_logic_vector(DATA_WIDTH - 1 downto 0); -- data output for port A
        we_b: in std_logic; -- write enable for port B
        addr_b: in std_logic_vector(ADDR_WIDTH - 1 downto 0); -- address for port B
        din_b: in std_logic_vector(DATA_WIDTH - 1 downto 0); -- data input for port B
        dout_b: out std_logic_vector(DATA_WIDTH - 1 downto 0) -- data output for port B
    );
end entity dual_port_bram;

architecture syn of dual_port_bram is
    type ram_type is array(0 to DEPTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    shared variable ram : ram_type;

begin

    process(CLK)
    begin
        if rising_edge(CLK) and ENA = '1' then
            dout_a <= ram(to_integer(unsigned(addr_a)));

            if we_a = '1' then
                ram(to_integer(unsigned(addr_a))) := din_a;
            end if;

        end if;

    end process;

    process(CLK)
    begin
        if rising_edge(CLK) and ENB = '1' then
            dout_b <= ram(to_integer(unsigned(addr_b)));

            if we_b = '1' then
                ram(to_integer(unsigned(addr_b))) := din_b;
            end if;

        end if;

    end process;

end architecture syn;