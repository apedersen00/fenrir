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
use std.textio.all;

entity rams_init_file_tb is
end rams_init_file_tb;

architecture test of rams_init_file_tb is
    constant CLK_PERIOD : time := 10 ns;
    signal clk  : std_logic := '0';
    signal we   : std_logic := '0';
    signal addr : std_logic_vector(7 downto 0) := (others => '0');
    signal din  : std_logic_vector(31 downto 0) := (others => '0');
    signal dout : std_logic_vector(31 downto 0);
    
    component rams_init_file
        port (
            clk  : in  std_logic;
            we   : in  std_logic;
            addr : in  std_logic_vector(7 downto 0);
            din  : in  std_logic_vector(31 downto 0);
            dout : out std_logic_vector(31 downto 0)
        );
    end component;

begin
    -- Instantiate the RAM module
    uut: rams_init_file
        port map (
            clk  => clk,
            we   => we,
            addr => addr,
            din  => din,
            dout => dout
        );
    
    -- Clock generation
    process
    begin
        while now < 500 ns loop
            clk <= '0';
            wait for CLK_PERIOD / 2;
            clk <= '1';
            wait for CLK_PERIOD / 2;
        end loop;
        wait;
    end process;
    
    -- Test process
    process
    begin
        wait for 20 ns;
        
        -- Read initial value from address 0
        addr <= std_logic_vector(to_unsigned(0, 8));
        wait for CLK_PERIOD;
        report "Initial data at address 0: " & integer'image(to_integer(unsigned(dout)));

        -- Read initial value from address 1
        addr <= std_logic_vector(to_unsigned(1, 8));
        wait for CLK_PERIOD;
        report "Initial data at address 1: " & integer'image(to_integer(unsigned(dout)));
        
        -- Write new data to address 2
        addr <= std_logic_vector(to_unsigned(2, 8));
        we <= '1';
        din <= x"DEADBEEF";
        wait for CLK_PERIOD;
        we <= '0';
        
        -- Read back written data
        wait for CLK_PERIOD;
        report "New data at address 2: " & integer'image(to_integer(unsigned(dout)));
        
        -- Stop simulation
        wait;
    end process;
end test;
