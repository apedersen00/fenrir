library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_control_t.all;

entity tb_conv_control is
end entity tb_conv_control;

architecture testbench of tb_conv_control is

    CONSTANT CLK_PERIOD : time := 10 ns;
    
    signal CLK : std_logic := '0';
    signal RESET : std_logic := '0';
    signal FIFO_EMPTY : std_logic := '0';
    signal FIFO_IN_DATA : std_logic_vector(FIFO_IN_DATA_WIDTH - 1 downto 0) := (others => '0');

begin

    CLK <= not CLK after CLK_PERIOD / 2;

    dut : entity work.conv_control
        port map(
            clk => CLK,
            reset => RESET,
            fifo_empty => FIFO_EMPTY,
            data_from_fifo => FIFO_IN_DATA
        );

    stimulus : process

    begin

        wait for 10 * CLK_PERIOD;

    end process;

end architecture testbench;