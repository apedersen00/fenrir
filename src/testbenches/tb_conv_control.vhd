library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_control_t.all;

entity tb_conv_control is
end entity tb_conv_control;

architecture testbench of tb_conv_control is

    CONSTANT CLK_PERIOD : time := 10 ns;
    
    signal CLK : std_logic := '1';
    signal RESET : std_logic := '0';
    signal FIFO_EMPTY : std_logic := '1';
    signal FIFO_IN_DATA : std_logic_vector(FIFO_IN_DATA_WIDTH - 1 downto 0) := (others => '0');
    signal READ_FIFO : std_logic := '0';
    signal INITIALIZE : std_logic := '0';
    signal initialize_value : std_logic_vector(MEM_SIZE_OF_ADDRESS - 1 downto 0) := (others => '0');
begin

    CLK <= not CLK after CLK_PERIOD / 2;

    dut : entity work.conv_control
        port map(
            clk => CLK,
            reset => RESET,
            fifo_empty => FIFO_EMPTY,
            data_from_fifo => FIFO_IN_DATA,
            read_from_fifo => READ_FIFO,
            init => INITIALIZE,
            initialize_value => initialize_value
        );

    stimulus : process

    begin
        FIFO_EMPTY <= '1'; -- Simulate FIFO being empty
        -- RESET THE DUT
        RESET <= '1';
        INITIALIZE <= '1';
        wait for clk_period;
        RESET <= '0';

        for i in 0 to 99 loop
            
            initialize_value <= x"A2" & std_logic_vector(to_unsigned(i MOD 4, 4));

            wait for clk_period;
        end loop;

        INITIALIZE <= '0';        

        wait for CLK_PERIOD * 2;

        FIFO_EMPTY <= '0'; -- Simulate data being available in FIFO
        
        wait for CLK_PERIOD * 2;
        FIFO_IN_DATA <= b"00100010011111";

        FIFO_EMPTY <= '1'; -- Simulate FIFO being empty
        wait for CLK_PERIOD * 2;
        FIFO_EMPTY <= '0';

        wait;

    end process;
 
end architecture testbench;