---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: lif.vhd
--  Description: Leaky-Integrate-and-Fire module for FENRIR. Also locksteps the loaders.
--  VHDL Version: VHDL-2008
--
--  Author(s):
--      - A. Pedersen, Aarhus University
--      - A. Cherencq, Aarhus University
--
---------------------------------------------------------------------------------------------------
--
--  Configuration Registers:
--      - (reg_cfg_0):
--          - <12b> [11:0]  threshold       : common spike threshold
--          - <12b> [23:12] beta (leakage)  : common neuron leakage per timestep
--          - <8b>  [31:24] weight scalar   : common weight scalar
--
---------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

--  Instantiation Template:
--  INST_NAME : entity work.FC_LIF_NEURON
--  port map (
--      i_cfg_en            =>
--      i_cfg_addr          =>
--      i_cfg_val           =>
--      i_nrn_valid         =>
--      i_nrn_valid_next    =>
--      i_nrn_valid_last    =>
--      i_nrn_state         =>
--      i_syn_valid         =>
--      i_syn_valid_next    =>
--      i_syn_valid_last    =>
--      i_syn_weight        =>
--      i_nrn_index         =>
--      i_timestep          =>
--      o_nrn_state_next    =>
--      o_event_fifo_out    =>
--      o_event_fifo_we     =>
--      o_continue          =>
--      o_goto_idle         =>
--      o_output_valid      =>
--      i_clk               =>
--      i_rst               =>
--  );

entity FC_LIF_NEURON is
    port (
        -- configuration interface
        i_reg_cfg_0         : in std_logic_vector(31 downto 0);

        -- neuron interface
        i_nrn_valid         : in std_logic;                             -- neuron state valid
        i_nrn_valid_next    : in std_logic;
        i_nrn_valid_last    : in std_logic;
        i_nrn_state         : in std_logic_vector(11 downto 0);         -- neuron state

        -- synapse interface
        i_syn_valid         : in std_logic;                             -- synapse weight valid
        i_syn_valid_next    : in std_logic;
        i_syn_valid_last    : in std_logic;
        i_syn_weight        : in std_logic_vector(7 downto 0);          -- synapse weight
        i_nrn_index         : in std_logic_vector(11 downto 0);         -- address of neuron

        -- control
        i_timestep          : in std_logic;                             -- timestep enable

        -- outputs
        o_nrn_state_next    : out std_logic_vector(11 downto 0);    -- next neuron state
        o_event_fifo_out    : out std_logic_vector(11 downto 0);    -- spike out event
        o_event_fifo_we     : out std_logic;                        -- enable write to output fifo
        o_continue          : out std_logic;                        -- continue iteration
        o_goto_idle         : out std_logic;
        o_output_valid      : out std_logic;

        -- misc
        i_clk           : in std_logic;
        i_rst           : in std_logic
    );
end FC_LIF_NEURON;

architecture Behavioral of FC_LIF_NEURON is

    attribute MARK_DEBUG : string;

    -- configuration
    signal cfg_threshold    : std_logic_vector(11 downto 0);
    signal cfg_beta         : std_logic_vector(11 downto 0);
    signal cfg_weight_scale : std_logic_vector(7 downto 0);
    signal cfg_bits_per_syn : integer;

    signal syn_reg          : std_logic_vector(7 downto 0);
    signal nrn_reg          : std_logic_vector(11 downto 0);
    signal idx_reg          : std_logic_vector(11 downto 0);
    signal reg_valid        : std_logic;

    signal dbg_continue     : std_logic;
    signal dbg_goto_idle    : std_logic;
    attribute MARK_DEBUG of syn_reg: signal is "TRUE";
    attribute MARK_DEBUG of nrn_reg: signal is "TRUE";
    attribute MARK_DEBUG of idx_reg: signal is "TRUE";
    attribute MARK_DEBUG of reg_valid: signal is "TRUE";
    attribute MARK_DEBUG of dbg_continue: signal is "TRUE";
    attribute MARK_DEBUG of dbg_goto_idle: signal is "TRUE";

begin

    o_continue <= dbg_continue;
    o_goto_idle <= dbg_goto_idle;

    -- configuration decoding
    cfg_threshold       <= i_reg_cfg_0(11 downto 0);
    cfg_beta            <= i_reg_cfg_0(23 downto 12);
    cfg_weight_scale    <= i_reg_cfg_0(31 downto 24);
    cfg_bits_per_syn    <= 4;

    -- lockstep the neuron and synapse loader
    dbg_continue  <= i_nrn_valid_next and (i_syn_valid_next or i_timestep);
    dbg_goto_idle <= i_nrn_valid_last and (i_syn_valid_last or i_timestep);

    out_val : process(i_clk)
    begin
        if rising_edge(i_clk) then
            o_output_valid <= reg_valid;
        end if;
    end process;

    input_reg : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                syn_reg     <= (others => '0');
                nrn_reg     <= (others => '0');
                idx_reg     <= (others => '0');
                reg_valid   <= '0';
            else
                if i_nrn_valid and (i_syn_valid or i_timestep) then
                    syn_reg     <= i_syn_weight;
                    nrn_reg     <= i_nrn_state;
                    idx_reg     <= i_nrn_index;
                    reg_valid   <= '1';
                else
                    syn_reg     <= (others => '0');
                    nrn_reg     <= (others => '0');
                    idx_reg     <= (others => '0');
                    reg_valid   <= '0';
                end if;
            end if;
        end if;
    end process;

    nxt_state : process(i_clk)
        variable v_syn_weight   : integer range -2047 to 2047;
        variable v_cur_state    : integer range -2047 to 2047;
        variable v_next_state   : integer range -65535 to 65535;
        variable v_beta         : integer range 0 to 4095;
        variable v_weight_scale : integer range 0 to 255;
    begin
        if rising_edge(i_clk) then
            if reg_valid = '1' then
                v_weight_scale  := to_integer(unsigned(cfg_weight_scale));
                v_beta          := to_integer(unsigned(cfg_beta));
                v_cur_state     := to_integer(signed(nrn_reg));
                v_syn_weight    := to_integer(signed(syn_reg(cfg_bits_per_syn - 1 downto 0)));

                if v_weight_scale /= 0 then
                    v_syn_weight := v_syn_weight * v_weight_scale;
                end if;

                if (i_timestep = '1') and (v_cur_state <= to_integer(unsigned(cfg_threshold))) then
                    if v_cur_state > v_beta then
                        v_next_state := v_cur_state - v_beta;
                    elsif v_cur_state < -v_beta then
                        v_next_state := v_cur_state + v_beta;
                    else
                        v_next_state := 0;
                    end if;
                elsif i_timestep = '0' then
                    v_next_state := v_cur_state + v_syn_weight;
                    if (v_next_state > 2047) then
                        v_next_state := 2047;
                    elsif (v_next_state < -2048) then
                        v_next_state := -2048;
                    end if;
                end if;

                if (i_timestep = '1') and (v_cur_state >= to_integer(unsigned(cfg_threshold))) then
                    o_nrn_state_next <= (others => '0');
                    o_event_fifo_out <= std_logic_vector(to_unsigned(0, 12)) when unsigned(idx_reg) = 0 else
                                        std_logic_vector(to_unsigned(to_integer(unsigned(idx_reg)), 12));
                    o_event_fifo_we  <= '1';
                else
                    o_nrn_state_next <= std_logic_vector(to_signed(v_next_state, o_nrn_state_next'length));
                    o_event_fifo_out <= (others => '0');
                    o_event_fifo_we  <= '0';
                end if;

            else
                o_event_fifo_we     <= '0';
                o_nrn_state_next    <= (others => '0');
                o_event_fifo_out    <= (others => '0');
                o_event_fifo_we     <= '0';
            end if;
        end if;
    end process;

end Behavioral;
