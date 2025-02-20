library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity synapse_memory is 
    port (
        clk                 : in std_logic;
        rst                 : in std_logic;

        synapse_address     : in std_logic_vector(3 downto 0);  -- synapse address

        we                  : in std_logic;                     -- write enable
        syn_in              : in std_logic_vector(3 downto 0);  -- synapse input data

        syn_out             : out std_logic_vector(3 downto 0)  -- synapse output data
    );
end synapse_memory;

architecture Behavioral of synapse_memory is
    type synapse_mem_t is array (0 to 1) of std_logic_vector(31 downto 0);
    signal mem : synapse_mem_t := (
        others => (others => '0')
    ); 

    signal word_idx : unsigned(0 downto 0);
    signal nibble_idx : unsigned(2 downto 0);
    signal read_word : std_logic_vector(31 downto 0);
    signal read_nibble : std_logic_vector(3 downto 0);

begin

    process(synapse_address)
    begin
        word_idx <= unsigned(synapse_address(3 downto 3));
        nibble_idx <= unsigned(synapse_address(2 downto 0));
    end process;

    process(clk)
        variable tmp_word : std_logic_vector(31 downto 0);
        variable idx : integer;
        variable nib_idx : integer;
        variable start_byte : integer;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                read_word <= (others => '0');
            else
                idx := to_integer(word_idx);
                tmp_word := mem(idx);

                if we = '1' then
                    nib_idx := to_integer(nibble_idx);
                    start_byte := nib_idx * 4;
                    tmp_word(start_byte + 3 downto start_byte) := syn_in;
                    mem(idx) <= tmp_word;
                end if;

                read_word <= tmp_word;

            end if;
        end if;
    end process;

    process(read_word, nibble_idx)
        variable nib_idx : integer;
        variable start_byte : integer;

    begin
        nib_idx := to_integer(nibble_idx);
        start_byte := nib_idx * 4;
        read_nibble <= read_word(start_byte + 3 downto start_byte);

        syn_out <= read_nibble;
    end process;
end Behavioral;