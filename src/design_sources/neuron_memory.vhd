library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity neuron_memory is
    port (
        clk : in std_logic;
        rst : in std_logic;
        neuron_address : in std_logic_vector(3 downto 0);
        
        param_leak_str : out std_logic_vector(6 downto 0);
        param_thr : out std_logic_vector(11 downto 0);

        state_core : out std_logic_vector(11 downto 0);
        state_core_next : in std_logic_vector(11 downto 0);

        we : in std_logic;
        neuron_in : in std_logic_vector(31 downto 0);
    );
end neuron_memory;

architecture Behavioral of neuron_memory is
    type neuron_mem_t is array (0 to 15) of std_logic_vector(31 downto 0);
    signal mem : neuron_mem_t := (
        others => (others => '0')
    );
begin

    process(clk)
        variable tmp_neuron : std_logic_vector(31 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                
                param_leak_str <= (others => '0');
                param_thr      <= (others => '0');
                state_core     <= (others => '0');
                
            else

                if we = '1' then
                    tmp_neuron := neuron_in;
                    mem(to_integer(unsigned(neuron_address))) <= tmp_neuron;
                else
                
                    tmp_neuron := mem(to_integer(unsigned(neuron_address)));
                    
                    param_leak_str <= tmp_neuron(7 downto 1);
                    param_thr      <= tmp_neuron(19 downto 8);
                    state_core     <= tmp_neuron(31 downto 20);

                    
                    tmp_neuron(31 downto 20) := state_core_next;
                    mem(to_integer(unsigned(neuron_address))) <= tmp_neuron;
                end if;
            end if;
        end if;
    end process;

end Behavioral;