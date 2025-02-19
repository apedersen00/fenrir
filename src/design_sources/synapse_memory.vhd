library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity synapse_memory is
    port (
        clk          : in  std_logic;
        rst          : in  std_logic;
        
        -- 13-bit address = { pre_neur<7:0>, post_neur<7:3> }
        synapse_addr : in  std_logic_vector(12 downto 0);
        
        -- Output nibble (mapping bit + 3-bit weight)
        synapse_nibble : out std_logic_vector(3 downto 0)
    );
end synapse_memory;

architecture Behavioral of synapse_memory is
    type synapse_mem_t is array (0 to 127) of std_logic_vector(31 downto 0);
    signal syn_mem : synapse_mem_t := (
        0   => x"0000000B",   
        32  => x"00000001",   
        64  => x"00000005",   
        96  => x"00000007",   
        others => (others => '0')
    );
    signal word_address : unsigned(6 downto 0);
    signal raw_word : std_logic_vector(31 downto 0);

begin 

    process(synapse_addr)
        variable v_pre : unsigned(7 downto 0);
        variable v_post : unsigned(4 downto 0);
    begin
        v_pre := unsigned(synapse_addr(12 downto 5));
        v_post := unsigned(synapse_addr(4 downto 0));
        word_address <= v_pre * 32 + v_post;
    end process;

    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                raw_word <= (others => '0');
            else
                raw_word <= syn_mem(to_integer(word_address));
            end if;
        end if;
    end process;

    synapse_nibble <= raw_word(3 downto 0);

end Behavioral;