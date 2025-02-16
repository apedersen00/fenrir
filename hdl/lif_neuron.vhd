/*
Aarhus University (AU, Denmark)

-----------------------------------------------------------------------------

lif_neuron.vhd
    VHDL adaptation of tinyODIN's LIF neuron model

Author(s):
    A. Pedersen, Aarhus University
    A. Cherencq, Aarhus University

Citations:
    C. Frenkel, M. Lefebvre, J.-D. Legat and D. Bol, "A 0.086-mm² 12.7-pJ/SOP 64k-Synapse 256-Neuron Online-Learning
    Digital Spiking Neuromorphic Processor in 28-nm CMOS," IEEE Transactions on Biomedical Circuits and Systems,
    vol. 13, no. 1, pp. 145-158, 2019.

-----------------------------------------------------------------------------

Description:
    Implementation of Leaky Integrate-and-Fire (LIF) Neuron Model.
    - Updates neuron state (state_core_next) based on synaptic events.
    - Two event types:
        - Leakage (event_leak): moves neuron state towards resting potential.
        - Synaptic (event_syn): moves neuron state towards firing potential.
    - Fires a spike (spike_out) when the state reaches the threshold (param_thr).
    - Neuron state is reset upon firing.

-----------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity lif_neuron is
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
end lif_neuron;

architecture Behavioral of lif_neuron is
    signal state_core_next_i  : std_logic_vector(11 downto 0);
    signal state_leakp_ovfl   : std_logic_vector(11 downto 0);
    signal state_leakn_ovfl   : std_logic_vector(11 downto 0);
    signal state_syn_ovfl     : std_logic_vector(11 downto 0);
    signal state_leakp        : std_logic_vector(11 downto 0);
    signal state_leakn        : std_logic_vector(11 downto 0);
    signal state_syn          : std_logic_vector(11 downto 0);
    signal syn_weight_ext     : std_logic_vector(11 downto 0);
    signal event_leak         : std_logic;
    signal event_syn          : std_logic;
begin
    event_leak <= syn_event and time_ref;
    event_syn  <= syn_event and (not time_ref);

    spike_out  <= '0' when state_core_next_i(11) = '1' else
                  '1' when state_core_next_i >= param_thr else '0';

    spike_core_next <= (others => '0') when spike_out = '1' else state_core_next_i;

    syn_weight_ext <= ("111111110000" & syn_weight) when syn_weight(3) = '1' else
                      ("000000000000" & syn_weight);

    process (state_core, event_leak, event_syn, state_leakp, state_leakn, state_syn)
    begin
        if event_leak = '1' then
            if state_core(11) = '1' then
                state_core_next_i <= state_leakp;
            else
                state_core_next_i <= state_leakn;
            end if;
        elsif event_syn = '1' then
            state_core_next_i <= state_syn;
        else
            state_core_next_i <= state_core;
        end if;
    end process;

    state_leakn_ovfl <= state_core - ("00000" * param_leak_str);
    state_leakn      <= "000000000000" when state_leakn_ovfl(11) = '1' else state_leakn_ovfl;
    state_leakp_ovfl <= state_core + ("00000" * param_leak_str);
    state_leakp      <= "000000000000" when state_leakp_ovfl(11) = '0' else state_leakp_ovfl;
    state_syn_ovfl   <= state_core + syn_weight_ext;

    state_syn <= "100000000000" when (state_syn_ovfl(11) = '0' and state_core(11) = '1' and syn_weight_ext(11) = '1') else
                 "011111111111" when (state_syn_ovfl(11) = '1' and state_core(11) = '0' and syn_weight_ext(11) = '0') else
                 state_syn_ovfl;

end Behavioral;
