/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: fenrir.vhd
    Description: Top module of FENRIR.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fenrir is
    generic (
        IN_SIZE : integer := 1024;
        NUM_NRN : integer := 10
    );
    port (
        -- control
        clk             : in std_logic;
        nRst            : in std_logic;
        busy            : out std_logic;

        -- data in
        data_rdy        : in std_logic;
        ibf_addr        : out std_logic_vector(15 downto 0);
        ibf_in          : in std_logic_vector(31 downto 0);
    );
end fenrir;

architecture Structural of fenrir is
    -- controller signals
    signal out_addr        : std_logic_vector(15 downto 0);
    signal out_in          : std_logic_vector(31 downto 0);

    signal syn_addr        : std_logic_vector(15 downto 0);
    signal syn_in          : std_logic_vector(31 downto 0);

    signal nrn_addr        : std_logic_vector(15 downto 0);
    signal nrn_in          : std_logic_vector(31 downto 0);

    -- memory signals
    signal out_we          : std_logic := '0';                 -- Write enable for memory
    signal out_din         : std_logic_vector(31 downto 0);    -- Data input to memory
    signal out_dout        : std_logic_vector(31 downto 0);    -- Data output from memory

    signal syn_we          : std_logic := '0';                 -- Write enable for memory
    signal syn_din         : std_logic_vector(31 downto 0);    -- Data input to memory
    signal syn_dout        : std_logic_vector(31 downto 0);    -- Data output from memory

    signal nrn_we          : std_logic;                        -- Write enable for memory
    signal nrn_din         : std_logic_vector(31 downto 0);    -- Data input to memory
    signal nrn_dout        : std_logic_vector(31 downto 0);    -- Data output from memory

    -- lif neuron
    signal param_leak_str  : std_logic_vector(6 downto 0);     -- leakage stength parameter
    signal param_thr       : std_logic_vector(11 downto 0);    -- neuron firing threshold parameter

    signal state_core      : std_logic_vector(11 downto 0);    -- core neuron state from SRAM
    signal state_core_next : std_logic_vector(11 downto 0);    -- next core neuron state to SRAM

    signal syn_weight      : std_logic_vector(3 downto 0);     -- synaptic weight
    signal syn_event       : std_logic;                        -- synaptic event trigger
    signal time_ref        : std_logic;                        -- time reference event trigger

    signal spike_out       : std_logic;                        -- neuron spike event output

begin

    -- instantiate controller
    controller: entity work.controller
        generic map (
            IN_SIZE         => IN_SIZE,
            NUM_NRN         => NUM_NRN
        )
        port map (
            clk             => clk,
            nRst            => nRst,
            busy            => busy,
            data_rdy        => data_rdy,

            ibf_addr        => ibf_addr,
            ibf_in          => ibf_in,

            out_addr        => out_addr,
            out_in          => out_dout,
            out_out         => out_din,
            out_we          => out_we,

            syn_addr        => syn_addr,
            syn_in          => syn_dout,

            nrn_addr        => nrn_addr,
            nrn_in          => nrn_dout,
            nrn_out         => nrn_din,
            nrn_we          => nrn_we,

            param_leak_str  => param_leak_str,
            param_thr       => param_thr,
            state_core      => state_core,
            state_core_next => state_core_next,
            syn_weight      => syn_weight,
            syn_event       => syn_event,
            time_ref        => time_ref,
            spike_out       => spike_out
        );

    lif_neuron: entity work.lif_neuron
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

    -- instantiate memory modules with unique names
    out_mem: entity work.bram_mem
        generic map (
            G_DEBUG                 => false,
            G_DEBUG_COUNTER_INIT    => 0,
            DEPTH                   => NUM_NRN / 32 + 1,
            WIDTH                   => 32,
            WIDTH_ADDR              => 16,
            FILENAME                => "data/out_init.data"
        )
        port map (
            clk         => clk,
            we          => out_we,
            addr        => out_addr,
            din         => out_din,
            dout        => out_dout
        );

    syn_mem: entity work.bram_mem
        generic map (
            G_DEBUG                 => false,
            G_DEBUG_COUNTER_INIT    => 0,
            DEPTH                   => IN_SIZE * IN_SIZE,
            WIDTH                   => 32,
            WIDTH_ADDR              => 16,
            FILENAME                => "data/4bit_syn.data"
        )
        port map (
            clk         => clk,
            we          => syn_we,
            addr        => syn_addr,
            din         => syn_din,
            dout        => syn_dout
        );

    nrn_mem: entity work.bram_mem
        generic map (
            G_DEBUG                 => false,
            G_DEBUG_COUNTER_INIT    => 0,
            DEPTH                   => NUM_NRN,
            WIDTH                   => 32,
            WIDTH_ADDR              => 16,
            FILENAME                => "data/nrn_init.data"
        )
        port map (
            clk         => clk,
            we          => nrn_we,
            addr        => nrn_addr,
            din         => nrn_din,
            dout        => nrn_dout
        );

end Structural;
