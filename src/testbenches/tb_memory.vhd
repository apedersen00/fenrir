/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: tb_memory.vhd
    Description: Testbench for BRAM initialization from external data file.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_neuron_bram is
end tb_neuron_bram;

architecture Behavioral of tb_neuron_bram is

    -- Component declaration for the Unit Under Test (UUT)
    component neuron_bram is
        port(
            clk  : in  std_logic;
            we   : in  std_logic;
            en   : in  std_logic;
            addr : in  std_logic_vector(7 downto 0);
            di   : in  std_logic_vector(31 downto 0);
            do   : out std_logic_vector(31 downto 0)
        );
    end component;

    -- Signals to connect to the UUT
    signal clk  : std_logic := '0';
    signal we   : std_logic;
    signal en   : std_logic;
    signal addr : std_logic_vector(7 downto 0);
    signal di   : std_logic_vector(31 downto 0);
    signal do   : std_logic_vector(31 downto 0);

    -- Clock period definition
    constant clk_period : time := 10 ns;

begin

    -- Instantiate the BRAM
    uut: neuron_bram port map(
        clk  => clk,
        we   => we,
        en   => en,
        addr => addr,
        di   => di,
        do   => do
    );

    -- Clock generation process
    clk_gen : process
    begin
        while true loop
            clk <= '0';
            wait for clk_period/2;
            clk <= '1';
            wait for clk_period/2;
        end loop;
    end process;

    -- Stimulus process
    stim_proc : process
    begin
        -- Initialize signals
        en   <= '1';
        we   <= '0';
        addr <= (others => '0');
        di   <= (others => '0');
        wait for 20 ns;

        -- Write operation: write 0xDEADBEEF at address 0x10
        addr <= x"10";
        di   <= x"DEADBEEF";
        we   <= '1';
        wait for clk_period;
        we   <= '0';
        wait for clk_period;

        -- Read operation: read from address 0x10
        addr <= x"10";
        wait for clk_period;
        -- The output do should now be 0xDEADBEEF

        wait for 20 ns;

        -- Write another value: write 0xAAAAAAAA at address 0x20
        addr <= x"20";
        di   <= x"AAAAAAAA";
        we   <= '1';
        wait for clk_period;
        we   <= '0';
        wait for clk_period;

        -- Read back from address 0x20
        addr <= x"20";
        wait for clk_period;
        -- The output do should now be 0xAAAAAAAA

        wait for 50 ns;
        -- End simulation
        wait;
    end process;

end Behavioral;
