library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity neuron_memory_tb is
end neuron_memory_tb;

architecture tb of neuron_memory_tb is
    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    signal neuron_addr : std_logic_vector(3 downto 0) := (others => '0');
    signal neuron_we : std_logic := '0';
    signal neuron_in : std_logic_vector(31 downto 0) := (others => '0');
    signal param_leak_str : std_logic_vector(6 downto 0);
    signal param_thr : std_logic_vector(11 downto 0);
    signal state_core : std_logic_vector(11 downto 0);
    signal state_core_next : std_logic_vector(11 downto 0) := (others => '0');

    constant CLK_PERIOD : time := 10 ns;
    
    begin
        uut : entity work.neuron_memory
            port map(
                clk => clk,
                rst => rst,
                neuron_address => neuron_addr,
                we => neuron_we,
                neuron_in => neuron_in,
                param_leak_str => param_leak_str,
                param_thr => param_thr,
                state_core => state_core,
                state_core_next => state_core_next
            );

        clk_process : process
        begin
            while true loop
                clk <= '0';
                wait for CLK_PERIOD / 2;
                clk <= '1';
                wait for CLK_PERIOD / 2;
            end loop;
        end process;

        stim : process
        begin
            -- Reset
            rst <= '1';
            wait for 10*CLK_PERIOD;
            rst <= '0';
            neuron_we <= '0';
            wait for CLK_PERIOD;

            -- read all 16 neurons should be 0
            for i in 0 to 15 loop
                neuron_addr <= std_logic_vector(to_unsigned(i, 4));
                wait for CLK_PERIOD;
            end loop;

            -- write some values to the memory
            neuron_we <= '1';
            neuron_addr <= "0000";
            wait;

        end process;
end architecture;

