library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_types.all;
use work.test_utils.all;

entity tb_parallel_conv is
end entity tb_parallel_conv;

architecture testbench of tb_parallel_conv is

    constant CLK_PERIOD : time := 10 ns;

    signal clk      : std_logic := '1';
    signal windows  : window_array_8_t;
    signal kernel   : kernel_t;
    signal conv_out : conv_out_array_t;

begin

    clk <= not clk after CLK_PERIOD/2;

    dut : entity work.parallel_conv
        port map (
            clk     => clk,
            window  => windows,
            kernel  => kernel,
            conv_out => conv_out
        );

    stimulus: process

        procedure wait_cycles(n : in integer) is
        begin
            for i in 1 to n loop
                wait for CLK_PERIOD;
            end loop;
        end procedure;

        procedure set_window_using_array(arr: in window_int_array; window_idx: integer) is
        begin 
            windows(window_idx) <= make_window_from_array(arr);
        end procedure;

        constant kernel_vals  : kernel_int_array := (1, 1, 1, 1, 1, 1, 1, 1, 1);
        variable w_arr : window_int_array(0 to 8);

    begin
        
        kernel <= make_kernel_from_array(kernel_vals);

        for win_idx in 0 to 7 loop
            for i in 0 to 8 loop
                if i < win_idx then
                    w_arr(i) := 1;
                else
                    w_arr(i) := 0;
                end if;
            end loop;

            set_window_using_array(w_arr, win_idx);
        end loop;
        wait_cycles(10);
        
        wait;
    end process;

end architecture testbench;
