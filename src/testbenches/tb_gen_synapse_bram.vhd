library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

use std.env.all;

entity mem_synapse_tb is
generic(
    TEST_DATA_FILE : string := "";
    REPORT_ONLY_ERRORS : boolean := true
);
end mem_synapse_tb;

architecture testbench of mem_synapse_tb is
    -- Constants
    constant CLK_PERIOD : time := 10 ns;
    constant MEM_DEPTH  : natural := 256;
    constant MEM_WIDTH  : natural := 32;
    constant ADDR_WIDTH : natural := 8;
    
    -- Signals
    signal clk          : std_logic := '0';
    signal we           : std_logic := '0';
    signal addr         : std_logic_vector(ADDR_WIDTH-1 downto 0) := (others => '0');
    signal din          : std_logic_vector(MEM_WIDTH-1 downto 0) := (others => '0');
    signal dout         : std_logic_vector(MEM_WIDTH-1 downto 0);
    
    -- Test control signals
    signal sim_done     : boolean := false;
    signal test_passed  : boolean := true;
begin
    -- Clock generation
    clk <= not clk after CLK_PERIOD/2 when not sim_done else '0';
    
    -- Device Under Test (DUT)
    dut: entity work.mem_synapse
    generic map (
        G_DEBUG             => false,
        G_DEBUG_COUNTER_INIT => 0,
        DEPTH               => MEM_DEPTH,
        WIDTH               => MEM_WIDTH,
        WIDTH_ADDR          => ADDR_WIDTH,
        FILENAME            => TEST_DATA_FILE
    )
    port map (
        clk                 => clk,
        we                  => we,
        addr                => addr,
        din                 => din,
        dout                => dout
    );
    
    -- Test process
    stimulus: process
    
    procedure wait_cycles(n : in integer) is
    begin
        for i in 1 to n loop
            wait for CLK_PERIOD;
        end loop;
    end procedure;

    procedure write_mem(address : natural; data : natural) is
    begin
        addr <= std_logic_vector(to_unsigned(address, ADDR_WIDTH));
        din <= std_logic_vector(to_unsigned(data, MEM_WIDTH));
        we <= '1';
        wait until rising_edge(clk);
        we <= '0';
    end procedure;

    procedure read_mem(a : natural; expected : natural) is
        variable read_value : natural;
    begin
        addr <= std_logic_vector(to_unsigned(a, ADDR_WIDTH));
        we <= '0';
        wait until rising_edge(clk);
        wait for 1 ns; -- Wait for output to stabilize
        
        read_value := to_integer(unsigned(dout));
        
        if read_value = expected then
            if not REPORT_ONLY_ERRORS then
                report "Read Test PASSED at address " & integer'image(a) & 
                        ": Expected " & integer'image(expected) & 
                        ", Got " & integer'image(read_value);
            end if;
            test_passed <= true;
        else
            report "Read Test FAILED at address " & integer'image(a) & 
                    ": Expected " & integer'image(expected) & 
                    ", Got " & integer'image(read_value)
                    severity error;
            test_passed <= false;
        end if;
    end procedure;

    impure function read_expected_value(addr_index : natural) return natural is
            file data_file : text;
            variable line_v : line;
            variable file_status : file_open_status;
            variable data_value : std_logic_vector(MEM_WIDTH-1 downto 0);
            variable i : natural := 0;
        begin
            file_open(file_status, data_file, TEST_DATA_FILE, read_mode);
            
            if file_status /= open_ok then
                report "Could not open file " & TEST_DATA_FILE severity error;
                return 0;
            end if;
            
            -- Skip to the line corresponding to the address
            while i < addr_index and not endfile(data_file) loop
                readline(data_file, line_v);
                i := i + 1;
            end loop;
            
            -- Read the value at the specified address
            if not endfile(data_file) then
                readline(data_file, line_v);
                read(line_v, data_value);
                file_close(data_file);
                return to_integer(unsigned(data_value));
            else
                file_close(data_file);
                -- If we're past the end of the file, return 0
                return 0;
            end if;
    end function;

    procedure test_done(test_idx : natural) is
    begin
        if test_passed then
            report "Test " & integer'image(test_idx) & ": PASSED";
        else 
            report "Test " & integer'image(test_idx) & ": FAILED" severity error;
        end if;
        
    end procedure;

    begin

        report "Starting testbench";
        wait_cycles(5);
        
        report "Test 1: Verification of memory initialization from file";
        for i in 0 to MEM_DEPTH-1 loop
            read_mem(i, read_expected_value(i));
        end loop;

        test_done(1);

        report "Test 2: Write and read back test";
        for i in 0 to MEM_DEPTH-1 loop
            write_mem(i, 255-i);
            wait_cycles(1);
            read_mem(i, 255-i);
        end loop;

        test_done(2);

        report "All tests complete";
        sim_done <= true;
        wait for 100 ns;
        finish;
        
        wait;
    end process;
    
end architecture testbench; 