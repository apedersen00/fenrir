library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library vunit_lib;
context vunit_lib.vunit_context;

use work.conv_pool_pkg.all;

entity tb_conv_pool is
    generic(runner_cfg : string);
end entity tb_conv_pool;

architecture testbench of tb_conv_pool is
    
    constant CLK_PERIOD : time := 10 ns;

    -- control signals
    signal clk : std_logic := '1';
    signal rst_o, enable_o, timestep_o : std_logic := '0';
    
    -- extra debug signals
    signal uut_main_state, uut_next_state, uut_last_state : main_state_et;
    signal uut_main_state_vec, uut_next_state_vec, uut_last_state_vec : std_logic_vector(2 downto 0);

    procedure waitf(n : in integer) is
    begin
        wait for n * CLK_PERIOD;
    end procedure waitf;

begin

    clk <= not clk after 10 ns;

    uut: entity work.conv_pool
    port map (
        clk => clk,
        rst_i => rst_o,
        enable_i => enable_o,
        timestep_i => timestep_o,

        debug_main_state => uut_main_state,
        debug_next_state => uut_next_state,
        debug_last_state => uut_last_state,
        debug_main_state_vec => uut_main_state_vec,
        debug_next_state_vec => uut_next_state_vec,
        debug_last_state_vec => uut_last_state_vec
    );

    main : process
    begin

        test_runner_setup(runner, runner_cfg);

        waitf(10); -- Initial wait to ensure everything is stable

        -- test reset
        if run("test_reset") then
            -- First, make sure we're in a known state
            rst_o <= '0';
            enable_o <= '1';  -- Enable the module
            waitf(2);         -- Wait for 2 clock cycles to stabilize
            
            -- Now assert reset and check the state
            rst_o <= '1';
            waitf(2);         -- Wait for 2 clock cycles to ensure reset propagates
            assert uut_main_state = RESET
                report "uut_main_state should be RESET after reset signal" 
            severity failure;

            -- Now deassert reset and check the state
            rst_o <= '0';
            waitf(2);         -- Wait for 2 clock cycles to ensure state transition
            assert uut_main_state = IDLE
                report "uut_main_state should be IDLE after reset signal is deasserted" 
            severity failure;
            
        end if;


        test_runner_cleanup(runner);
        wait for 100 ns; -- <-- Add this line!
        wait;

    end process main;

end testbench;