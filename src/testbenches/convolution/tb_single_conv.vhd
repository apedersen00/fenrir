library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_types.all;

entity tb_single_conv is
end entity tb_single_conv;

architecture testbench of tb_single_conv is

    constant CLK_PERIOD : time := 10 ns;

    signal clk      : std_logic := '1';
    signal window   : window_t;
    signal kernel   : kernel_t;
    signal conv_out : signed(CONV_OUT_BIT_WIDTH-1 downto 0);

    signal sim_done : boolean := false;
    signal test_passed : boolean := true;

begin

    clk <= not clk after CLK_PERIOD/2 when not sim_done else '0';

    dut : entity work.convolution_single
    generic map (
        conv_out_bit_width => CONV_OUT_BIT_WIDTH
    )
    port map (
        clk     => clk,
        window  => window,
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

    procedure set_window_with_sum(sum : integer) is
        
        variable tmp_window : window_t;
    begin
        tmp_window.v00 := to_signed(sum,2);
        tmp_window.v01 := to_signed(sum,2);
        tmp_window.v02 := to_signed(sum,2);
        tmp_window.v10 := to_signed(sum,2);
        tmp_window.v11 := to_signed(sum,2);
        tmp_window.v12 := to_signed(sum,2);
        tmp_window.v20 := to_signed(sum,2);
        tmp_window.v21 := to_signed(sum,2);
        tmp_window.v22 := to_signed(sum,2);

        window <= tmp_window;
    end procedure;

    procedure set_kernel_with_sum(sum : integer) is
        
        variable tmp_kernel : kernel_t;
    begin
        tmp_kernel.k00 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);
        tmp_kernel.k01 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);
        tmp_kernel.k02 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);
        tmp_kernel.k10 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);
        tmp_kernel.k11 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);
        tmp_kernel.k12 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);
        tmp_kernel.k20 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);
        tmp_kernel.k21 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);
        tmp_kernel.k22 := to_signed(sum,DEFAULT_KERNEL_BIT_WIDTH);

        kernel <= tmp_kernel;
    end procedure;

    begin

        wait_cycles(2);

        set_window_with_sum(1);
        set_kernel_with_sum(1);
        
        wait_cycles(5);

        window.v00 <= to_signed(-1, 2);

        wait for CLK_PERIOD;

        window.v00 <= to_signed(0, 2);

        wait;

    end process;

end architecture testbench;