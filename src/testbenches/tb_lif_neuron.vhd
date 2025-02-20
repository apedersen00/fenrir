/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: tb.vhd
    Description: Testbench for the LIF neuron model (lif_neuron.vhd).

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity lif_neuron_tb is
end lif_neuron_tb;

architecture Behavioral of lif_neuron_tb is
    -- Constants
    constant CLK_PERIOD : time := 10 ns;

    -- Signals
    signal clk              : std_logic := '0';
    signal reset            : std_logic := '1';
    signal param_leak_str   : std_logic_vector(6 downto 0)  := "0000001";
    signal param_thr        : std_logic_vector(11 downto 0) := "000000000100";
    signal state_core       : std_logic_vector(11 downto 0) := (others => '0');
    signal state_core_next  : std_logic_vector(11 downto 0);
    signal syn_weight       : std_logic_vector(3 downto 0)  := "0011";
    signal syn_event        : std_logic := '0';
    signal time_ref         : std_logic := '0';
    signal spike_out        : std_logic;

    -- Instantiate the LIF neuron
    component lif_neuron
        port (
            param_leak_str  : in std_logic_vector(6 downto 0);
            param_thr       : in std_logic_vector(11 downto 0);
            state_core      : in std_logic_vector(11 downto 0);
            state_core_next : out std_logic_vector(11 downto 0);

            syn_weight      : in std_logic_vector(3 downto 0);
            syn_event       : in std_logic;
            time_ref        : in std_logic;

            spike_out       : out std_logic
        );
    end component;

begin
    -- Clock generation
    clk <= not clk after CLK_PERIOD / 2;

    -- Instantiate the LIF neuron
    uut: lif_neuron
        port map (
            param_leak_str  => param_leak_str,
            param_thr       => param_thr,
            state_core      => state_core,
            state_core_next => state_core_next,
            syn_weight      => syn_weight,
            syn_event       => syn_event,
            time_ref        => time_ref,
            spike_out       => spike_out
        );

    -- Test process
    process
    begin
        -- Reset
        reset <= '1';
        wait for CLK_PERIOD * 2;
        reset <= '0';

        -- Positive weight = +1
        -- Initial state
        syn_weight  <= "0001";
        state_core  <= "000000000000";
        syn_event   <= '0';
        time_ref    <= '0';
        wait for CLK_PERIOD;

        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;

        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;

        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;

        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;
    
        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;

        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;

        -- Leakage event
        syn_event   <= '1';
        time_ref    <= '1';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        time_ref    <= '0';
        wait for CLK_PERIOD;

        -- Leakage event
        syn_event   <= '1';
        time_ref    <= '1';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        time_ref    <= '0';
        wait for CLK_PERIOD;

        -- Leakage event
        syn_event   <= '1';
        time_ref    <= '1';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        time_ref    <= '0';
        wait for CLK_PERIOD;

        -- Negative weight = -1
        -- Initial state
        syn_weight  <= "1111";
        state_core  <= "000000000000";
        syn_event   <= '0';
        time_ref    <= '0';
        wait for CLK_PERIOD;

        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;

        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;

        -- Synaptic event
        syn_event   <= '1';
        time_ref    <= '0';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        wait for CLK_PERIOD;

        -- Leakage event
        syn_event   <= '1';
        time_ref    <= '1';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        time_ref    <= '0';
        wait for CLK_PERIOD;

        -- Leakage event
        syn_event   <= '1';
        time_ref    <= '1';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        time_ref    <= '0';
        wait for CLK_PERIOD;

        -- Leakage event
        syn_event   <= '1';
        time_ref    <= '1';
        wait for CLK_PERIOD;
        state_core  <= state_core_next;
        syn_event   <= '0';
        time_ref    <= '0';
        wait for CLK_PERIOD;

        -- End of simulation
        wait;
    end process;
end Behavioral;
