library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_system.all;

entity tb_downsampler is
end entity tb_downsampler;

architecture testbench of tb_downsampler is

    CONSTANT CLK_PERIOD : time := 10 ns;

    signal clk                      : std_logic := '1';
    signal timeref_clock            : timeref_clock := (others => '0');
    signal AER_IN                   : DUMMY_AER_IN;
    signal AER_OUT                  : EVENT_R;
    signal ready                    : std_logic;
    signal data_ready               : std_logic := '0';
    signal fifo_out_write_enable    : std_logic;
    signal fifo_full                : std_logic := '0';

begin

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.downsampler
        port map(
            clk                     => clk,
            timeref_clock           => timeref_clock,
            AER_In                  => AER_IN,
            AER_Out                 => AER_OUT,
            ready                   => ready,
            data_ready              => data_ready,
            fifo_out_write_enable   => fifo_out_write_enable,
            fifo_full               => fifo_full
        );

    -- timeref clock generation process
    process (clk)
        variable counter : integer := 1;
        constant divide_at : integer := 10;
    begin
        if rising_edge(clk) then

            if counter = divide_at then
                counter := 1;
                timeref_clock <= std_logic_vector(unsigned(timeref_clock) + 1);
            else
                counter := counter + 1;
            end if;
        end if;
    end process;

    -- dummy variable for checking
    
    stimulus : process
    begin

        wait for CLK_PERIOD;

        data_ready <= '1';

        wait for CLK_PERIOD;

        AER_IN.x <= std_logic_vector(to_unsigned(10, 16));
        AER_IN.y <= std_logic_vector(to_unsigned(11, 16));
        AER_IN.polarity <= std_logic_vector(to_unsigned(1, 2));

        wait for CLK_PERIOD;
        data_ready <= '0';
        wait for CLK_PERIOD * 34;

        data_ready <= '1';

        wait for CLK_PERIOD;

        AER_IN.x <= std_logic_vector(to_unsigned(25, 16));
        AER_IN.y <= std_logic_vector(to_unsigned(31, 16));
        AER_IN.polarity <= std_logic_vector(to_signed(-1, 2));

        wait for CLK_PERIOD;
        data_ready <= '0';
        wait;

    end process stimulus;

end architecture testbench;