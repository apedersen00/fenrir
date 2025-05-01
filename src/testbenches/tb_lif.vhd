library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use std.env.finish;

entity lif_tb is
end lif_tb;

architecture behavior of lif_tb is

    constant clk_period : time := 10 ns;

    signal clk          : std_logic := '0';

    signal cfg_en       : std_logic;
    signal cfg_addr     : std_logic_vector(3 downto 0);
    signal cfg_val      : std_logic_vector(31 downto 0);
    
    signal syn_valid    : std_logic;
    signal nrn_valid    : std_logic;
    signal syn_weight   : std_logic_vector(7 downto 0);
    signal nrn_state    : std_logic_vector(11 downto 0);
    signal nrn_index    : std_logic_vector(15 downto 0);
    signal timestep     : std_logic;
    signal nrn_state_next   : std_logic_vector(11 downto 0);
    signal event_fifo_data  : std_logic_vector(15 downto 0);
    signal event_fifo_we    : std_logic;
    signal rst          : std_logic;

begin

    LIF_NEURON : entity work.LIF_NEURON
    port map (
        i_cfg_en            => cfg_en,
        i_cfg_addr          => cfg_addr,
        i_cfg_val           => cfg_val,
        i_syn_valid         => syn_valid,
        i_nrn_valid         => nrn_valid,
        i_syn_weight        => syn_weight,
        i_nrn_state         => nrn_state,
        i_nrn_index         => nrn_index,
        i_timestep          => timestep,
        o_nrn_state_next    => nrn_state_next,
        o_event_fifo_out    => event_fifo_data,
        o_event_fifo_we     => event_fifo_we,
        i_clk               => clk,
        i_rst               => rst
    );

    clk <= not clk after clk_period / 2;

    PROC_SEQUENCER : process
    begin

        -- Initialize signals
        cfg_en      <= '0';
        cfg_addr    <= (others => '0');
        cfg_val     <= (others => '0');
        syn_valid   <= '0';
        nrn_valid   <= '0';
        syn_weight  <= (others => '0');
        nrn_state   <= (others => '0');
        nrn_index   <= (others => '0');
        timestep    <= '0';
        rst         <= '1';
        wait until rising_edge(clk);

        -- configure synapse loader
        rst      <= '0';
        cfg_en   <= '1';
        cfg_addr <= "0000";
        cfg_val  <=
            "00000000"                              &   -- zero padding
            std_logic_vector(to_unsigned(1, 12))    &   -- beta
            std_logic_vector(to_unsigned(10, 12));      -- threshold
        wait until rising_edge(clk);
        cfg_en  <= '0';

        -- Neuron state = 1, weight = 0
        wait until rising_edge(clk);
        syn_weight  <= std_logic_vector(to_unsigned(0, 8));
        nrn_state   <= std_logic_vector(to_unsigned(1, 12));
        nrn_index   <= std_logic_vector(to_unsigned(0, 16));
        syn_valid   <= '1';
        nrn_valid   <= '1';
        timestep    <= '0';
        wait until rising_edge(clk);

        -- Neuron state = 1, weight = 1
        wait until rising_edge(clk);
        syn_weight  <= std_logic_vector(to_unsigned(1, 8));
        nrn_state   <= std_logic_vector(to_unsigned(1, 12));
        nrn_index   <= std_logic_vector(to_unsigned(1, 16));
        syn_valid   <= '1';
        nrn_valid   <= '1';
        timestep    <= '0';
        wait until rising_edge(clk);

        -- Neuron state = 8, weight = 3
        wait until rising_edge(clk);
        syn_weight  <= std_logic_vector(to_unsigned(3, 8));
        nrn_state   <= std_logic_vector(to_unsigned(8, 12));
        nrn_index   <= std_logic_vector(to_unsigned(2, 16));
        syn_valid   <= '1';
        nrn_valid   <= '1';
        timestep    <= '0';
        wait until rising_edge(clk);

        -- Neuron state = 8, weight = 1
        wait until rising_edge(clk);
        syn_weight  <= std_logic_vector(to_unsigned(2, 8));
        nrn_state   <= std_logic_vector(to_unsigned(8, 12));
        nrn_index   <= std_logic_vector(to_unsigned(2, 16));
        syn_valid   <= '1';
        nrn_valid   <= '1';
        timestep    <= '1';
        wait until rising_edge(clk);

        for i in 0 to 100 loop
            wait until rising_edge(clk);
        end loop;

        finish;
    end process;

end behavior;
