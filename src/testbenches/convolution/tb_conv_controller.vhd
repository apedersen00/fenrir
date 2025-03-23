library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_types.all;

entity tb_conv_controller is
end entity tb_conv_controller;

architecture testbench of tb_conv_controller is

    constant CLK_PERIOD : time := 10 ns;

    signal clk : std_logic := '1';
    signal data_ready : std_logic := '0';
    signal windows : window_array_8_t;
    signal kernel : kernel_t;
    signal img_row_data_in : std_logic_vector(63 downto 0);
    signal img_row_address : std_logic_vector(7 downto 0);
    signal img_write_en : std_logic;
    signal kernel_data_in : std_logic_vector(63 downto 0);
    signal kernel_address : std_logic_vector(7 downto 0);
    signal kernel_write_en : std_logic;

    subtype kernel_ram is std_logic_vector(63 downto 0);
    type test_kernels_t is array(0 to 7) of kernel_ram;
    signal test_kernels : test_kernels_t := (
        x"1FFFFFFFFFFFFFFF",
        x"2222222222222222",
        x"3333333333333333",
        x"4444444444444444",
        x"5555555555555555",
        x"6666666666666666",
        x"7777777777777777",
        x"8888888888888888"
    );
    


begin

    clk <= not clk after CLK_PERIOD/2;

    dut : entity work.conv_controller
        port map(
            clk => clk,
            data_ready => data_ready,
            windows => windows,
            kernel => kernel,
            img_row_data_in => img_row_data_in,
            img_row_address => img_row_address,
            img_write_en => img_write_en,
            kernel_data_in => kernel_data_in,
            kernel_address => kernel_address,
            kernel_write_en => kernel_write_en,
            debug_state => debug_state
        );
    
    stimulus: process
    
        procedure wait_cycles(n : in integer) is
        begin
            for i in 1 to n loop
                wait for CLK_PERIOD;
            end loop;
        end procedure;

        procedure sim_bram(address : std_logic_vector(7 downto 0)) is 
        begin
            kernel_data_in <= test_kernels(to_integer(unsigned(address)));
        end procedure;

    begin

        wait for CLK_PERIOD;

        data_ready <= '1';

        wait for CLK_PERIOD;

        data_ready <= '0';

        wait for CLK_PERIOD;

        sim_bram(kernel_address);

        wait;

    end process;

end architecture testbench;