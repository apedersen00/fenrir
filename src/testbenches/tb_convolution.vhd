library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_conv is
end entity tb_conv;

architecture testbench of tb_conv is

    CONSTANT CLK_PERIOD     : time                       := 10 ns;
    CONSTANT AER_BUS_WIDTH  : integer                    := 32;

    signal clk              : std_logic                  := '1';
    signal reset_o          : std_logic                  := '0';
    signal enable_o         : std_logic                  := '0';

    -- fifo signals
    signal aer_fifo_bus_o   : std_logic_vector(AER_BUS_WIDTH - 1 downto 0) := (others => '0');
    signal aer_empty_o      : std_logic := '1';
    signal aer_fifo_read_i  : std_logic;

    

begin
    
    -- UUT
    uut : entity work.Reverse_Convolution_Layer
        generic map(
            AerBusWidth => AER_BUS_WIDTH
        )
        port map(
            clk                => clk,
            reset_i            => reset_o,
            enable_i           => enable_o,
            aer_fifo_bus_i     => aer_fifo_bus_o,
            aer_empty_i        => aer_empty_o,
            aer_fifo_read_o    => aer_fifo_read_i
        );

    -- Clock Generation
    clk <= not clk after CLK_PERIOD / 2;

    -- Test reset
    stimulus : process

        procedure waitf(n : in integer) is
        begin
            for i in 1 to n loop
                wait for CLK_PERIOD;
            end loop;
        end procedure;

    begin
        -- Reset the system
        reset_o <= '1';
        waitf(5);
        reset_o <= '0';
        waitf(2);
        -- Enable the system
        enable_o <= '1';
        waitf(2);
        -- Disable the system
        enable_o <= '0';
        aer_empty_o <= '0';
        waitf(2);
        -- Enable the system again
        aer_empty_o <= '1';
        enable_o <= '1';

        waitf(4);
        -- Test the FIFO read
        -- simulate fifo not empty
        aer_empty_o <= '0';


        wait;

    end process stimulus;

end architecture testbench;