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

    Configuration Registers:
        - (reg_cfg_0):
            - <12b> [11:0]  threshold       : common spike threshold
            - <12b> [23:12] beta (leakage)  : common neuron leakage per timestep

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

--  Instantiation Template:
--  INST_NAME : entity work.LIF_NEURON
--  port map (
--      i_cfg_en            =>
--      i_cfg_addr          =>
--      i_cfg_val           =>
--      i_syn_valid         =>
--      i_nrn_valid         =>
--      i_syn_weight        =>
--      i_nrn_state         =>
--      i_nrn_index         =>
--      i_timestep          =>
--      o_nrn_state_next    =>
--      o_event_fifo_out    =>
--      o_event_fifo_we     =>
--      i_clk               =>
--      i_rst               =>
--  );

entity LIF_NEURON is
    port (
        -- configuration interface
        i_cfg_en        : in std_logic;                             -- enable configuration
        i_cfg_addr      : in std_logic_vector(3 downto 0);          -- register to configure
        i_cfg_val       : in std_logic_vector(31 downto 0);         -- value to configure

        -- inputs
        i_syn_valid     : in std_logic;                             -- synapse weight valid
        i_nrn_valid     : in std_logic;                             -- neuron state valid
        i_syn_weight    : in std_logic_vector(7 downto 0);          -- synapse weight
        i_nrn_state     : in std_logic_vector(11 downto 0);         -- neuron state
        i_nrn_index     : in std_logic_vector(15 downto 0);         -- address of neuron
        i_timestep      : in std_logic;                             -- timestep enable

        -- outputs
        o_nrn_state_next    : out std_logic_vector(11 downto 0);    -- next neuron state
        o_event_fifo_out    : out std_logic_vector(15 downto 0);    -- spike out event
        o_event_fifo_we     : out std_logic;                        -- enable write to output fifo

        -- misc
        i_clk           : in std_logic;
        i_rst           : in std_logic;
    );
end LIF_NEURON;

architecture Behavioral of LIF_NEURON is
    -- registers
    signal reg_cfg_0        : std_logic_vector(31 downto 0);    -- configuration register 0

    -- configuration
    signal cfg_threshold    : std_logic_vector(11 downto 0);
    signal cfg_beta         : std_logic_vector(11 downto 0);

begin

    -- configuration decoding
    cfg_threshold   <= reg_cfg_0(11 downto 0);
    cfg_beta        <= reg_cfg_0(23 downto 12);

    -- configuration interface
    config : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                reg_cfg_0   <= (others => '0');
            elsif i_cfg_en = '1' then
                case i_cfg_addr is
                    when "0000" => reg_cfg_0 <= i_cfg_val;
                    when others => null;
                end case;
            end if;
        end if;
    end process;

    nxt_state : process(i_clk)
        variable v_next_state : integer range 0 to 65536;
    begin
        if rising_edge(i_clk) then
            if i_syn_valid = '1' and i_nrn_valid = '1' then
                v_next_state := to_integer(unsigned(i_syn_weight)) + to_integer(unsigned(i_nrn_state));

                if i_timestep = '1' then
                    v_next_state := v_next_state - to_integer(unsigned(cfg_beta));
                end if;

                if v_next_state >= to_integer(unsigned(cfg_threshold)) then
                    o_nrn_state_next <= (others => '0');
                    o_event_fifo_out <= i_nrn_index;
                    o_event_fifo_we  <= '1';
                else
                    o_nrn_state_next <= std_logic_vector(to_unsigned(v_next_state, o_nrn_state_next'length));
                    o_event_fifo_out <= (others => '0');
                    o_event_fifo_we  <= '0';
                end if;

            end if;
        end if;
    end process;

end Behavioral;
