library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity synapse_memory_tb is 
end synapse_memory_tb;

architecture tb of synapse_memory_tb is
    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    signal syn_idx : std_logic_vector(3 downto 0) := (others => '0');
    signal syn_we : std_logic := '0';
    signal syn_in : std_logic_vector(3 downto 0) := (others => '0');
    signal syn_out : std_logic_vector(3 downto 0);

    constant CLK_PERIOD : time := 10 ns;

    begin
        uut : entity work.synapse_memory
            port map(
                clk => clk,
                rst => rst,
                synapse_address => syn_idx,
                we => syn_we,
                syn_in => syn_in,
                syn_out => syn_out
            );
        
        clk_process : process
        begin
            while true loop
                clk <= '0';
                wait for clk_period / 2;
                clk <= '1';
                wait for clk_period / 2;
            end loop;
        end process;

        stim: process
    begin
        -- Reset
        rst <= '1';
        wait for 10*CLK_PERIOD;
        rst <= '0';
        syn_we <= '0';
        wait for CLK_PERIOD;

        -- read all 16 weights should be 0
        for i in 0 to 15 loop
            syn_idx <= std_logic_vector(to_unsigned(i, 4));
            wait for CLK_PERIOD;
        end loop;

        -- write some values to the memory
        syn_we <= '1';
        syn_idx <= "0000";
        syn_in <= "0001";
        wait for CLK_PERIOD;

        syn_idx <= "0001";
        syn_in <= "0010";
        wait for CLK_PERIOD;

        --read the values back
        syn_we <= '0';
        syn_idx <= "0000";
        wait for CLK_PERIOD;
        syn_idx <= "0001";
        wait for CLK_PERIOD;

        wait;
    end process;

end architecture;