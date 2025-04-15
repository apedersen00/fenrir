/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: lif.vhd
    Description:

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity lif_block is
    port (
        i_syn_weight    : in std_logic_vector(7 downto 0);
        i_nrn_state     : in std_logic_vector(11 downto 0);

        i_timestep      : in std_logic;
        o_spike         : out std_logic;

        o_fault         : out std_logic
    );
end lif_block;

architecture Behavioral of lif_neuron is
    signal shadow_syn_weight_reg    : std_logic_vector(7 downto 0);
    signal shadow_nrn_state_reg     : std_logic_vector(11 downto 0);

    signal syn_weight_reg           : std_logic_vector(7 downto 0);
    signal nrn_state_reg            : std_logic_vector(11 downto 0);
begin
    event_leak <= syn_event and time_ref;
    event_syn  <= syn_event and (not time_ref);

    spike_out  <= '0' when state_core_next_i(11) = '1' else
                  '1' when state_core_next_i >= param_thr else '0';

    state_core_next <= (others => '0') when spike_out = '1' else state_core_next_i;

    syn_weight_ext <= "11111111" & syn_weight when syn_weight(3) = '1' else
                      "00000000" & syn_weight;

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

    state_leakn_ovfl <= state_core - ("00000" & param_leak_str);
    state_leakn      <= (others => '0') when state_leakn_ovfl(11) = '1' else state_leakn_ovfl;
    state_leakp_ovfl <= state_core + ("00000" & param_leak_str);
    state_leakp      <= (others => '0') when state_leakp_ovfl(11) = '0' else state_leakp_ovfl;
    state_syn_ovfl   <= state_core + syn_weight_ext;

    state_syn <= "100000000000" when (state_syn_ovfl(11) = '0' and state_core(11) = '1' and syn_weight_ext(11) = '1') else
                 "011111111111" when (state_syn_ovfl(11) = '1' and state_core(11) = '0' and syn_weight_ext(11) = '0') else
                 state_syn_ovfl;

end Behavioral;
