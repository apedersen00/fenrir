---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: fc_layer.vhd
--  Description: A single-layer fully-connected spiking neural network. Contains input FIFO.
--
--  Author(s):
--      - A. Pedersen, Aarhus University
--      - A. Cherencq, Aarhus University
--
---------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use std.textio.all;

--  Instantiation Template:
--  INST_NAME : entity work.FC_LAYER
--  generic map (
--      IN_SIZE         =>
--      OUT_SIZE        =>
--      SYN_MEM_WIDTH   =>
--      BITS_PER_SYN    =>
--      SYN_INIT_FILE   =>
--      NRN_INIT_FILE   =>
--  )
--  port map (
--      -- config
--      i_cfg_en            =>
--      i_cfg_addr          =>
--      i_cfg_val           =>
--      -- input
--      i_in_fifo_we        =>
--      i_in_fifo_wdata     =>
--      -- output
--      o_out_fifo_we       =>
--      o_out_fifo_wdata    =>
--      -- status
--      o_in_fifo_empty     =>
--      o_in_fifo_full      =>
--      i_out_fifo_full     =>
--      o_busy              =>
--      -- control
--      i_enable            =>
--      i_rst               =>
--      i_clk               =>
--      -- debug
--      o_sched_tstep       =>
--      o_nrnmem_we         =>
--      o_nrnmem_waddr      =>
--      o_nrnmem_wdata      =>
--  );

entity FC_LAYER is
    generic (
        IN_SIZE         : integer;
        OUT_SIZE        : integer;
        IS_LAST         : integer;
        SYN_MEM_WIDTH   : integer;
        BITS_PER_SYN    : integer;
        SYN_INIT_FILE   : string := "";
        NRN_INIT_FILE   : string := ""
    );
    port (
        i_cfg_en            : in std_logic;
        i_cfg_addr          : in std_logic_vector(7 downto 0);
        i_cfg_val           : in std_logic_vector(31 downto 0);
        i_enable            : in std_logic;
        i_in_fifo_we        : in std_logic;
        i_in_fifo_wdata     : in std_logic_vector(12 downto 0);
        o_in_fifo_empty     : out std_logic;
        o_in_fifo_full      : out std_logic;
        i_out_fifo_full     : in std_logic;
        o_out_fifo_we       : out std_logic;
        o_out_fifo_wdata    : out std_logic_vector(12 downto 0);
        i_rst               : in std_logic;
        i_clk               : in std_logic;
        o_busy              : out std_logic;
        o_sched_tstep       : out std_logic;                                                                                    -- for tb only
        o_nrnmem_we         : out std_logic;                                                                                    -- for tb only
        o_nrnmem_waddr      : out std_logic_vector(integer(ceil(log2(real(integer(ceil(real(OUT_SIZE) / 3.0))))))-1 downto 0);  -- for tb only
        o_nrnmem_wdata      : out std_logic_vector(35 downto 0)                                                                 -- for tb only
    );
end FC_LAYER;

architecture behavior of FC_LAYER is

    constant SYN_MEM_DEPTH  : integer := IN_SIZE * OUT_SIZE / (SYN_MEM_WIDTH / BITS_PER_SYN);
    constant NRN_MEM_DEPTH  : integer := (OUT_SIZE + 3 - 1) / 3;

    -- synapse loader config interface
    signal synldr_cfg_en        : std_logic;
    signal synldr_cfg_addr      : std_logic_vector(3 downto 0);
    signal synldr_cfg_val       : std_logic_vector(31 downto 0);

    -- neuron loader config interface
    signal nrnldr_cfg_en        : std_logic;
    signal nrnldr_cfg_addr      : std_logic_vector(3 downto 0);
    signal nrnldr_cfg_val       : std_logic_vector(31 downto 0);

    -- lif config interface
    signal lif_cfg_en           : std_logic;
    signal lif_cfg_addr         : std_logic_vector(3 downto 0);
    signal lif_cfg_val          : std_logic_vector(31 downto 0);

    -- neuron writer config interface
    signal nrnwrt_cfg_en        : std_logic;
    signal nrnwrt_cfg_addr      : std_logic_vector(3 downto 0);
    signal nrnwrt_cfg_val       : std_logic_vector(31 downto 0);

    -- input event fifo
    signal in_fifo_re          : std_logic;
    signal in_fifo_rvalid      : std_logic;
    signal in_fifo_rdata       : std_logic_vector(12 downto 0);
    signal in_fifo_empty       : std_logic;

    -- synapse loader
    signal synldr_weight        : std_logic_vector(7 downto 0);
    signal synldr_valid         : std_logic;
    signal synldr_valid_next    : std_logic;
    signal synldr_valid_last    : std_logic;
    signal synldr_start         : std_logic;
    signal synldr_busy          : std_logic;
    signal synldr_fifo_rdata    : std_logic_vector(11 downto 0);
    signal synldr_fifo_re       : std_logic;

    -- neuron loader
    signal nrnldr_state         : std_logic_vector(11 downto 0);
    signal nrnldr_nrn_index     : std_logic_vector(11 downto 0);
    signal nrnldr_valid         : std_logic;
    signal nrnldr_valid_next    : std_logic;
    signal nnrldr_valid_last    : std_logic;
    signal nrnldr_start         : std_logic;
    signal nrnldr_busy          : std_logic;

    -- lif
    signal lif_nrn_state_next   : std_logic_vector(11 downto 0);
    signal lif_continue         : std_logic;
    signal lif_goto_idle        : std_logic;
    signal lif_out_valid        : std_logic;
    signal lif_fifo_we          : std_logic;
    signal lif_fifo_wdata       : std_logic_vector(11 downto 0);

    -- synapse memory
    signal synmem_we            : std_logic;
    signal synmem_waddr         : std_logic_vector(integer(ceil(log2(real(SYN_MEM_DEPTH))))-1 downto 0);
    signal synmem_wdata         : std_logic_vector(SYN_MEM_WIDTH - 1 downto 0);
    signal synmem_re            : std_logic;
    signal synmem_raddr         : std_logic_vector(integer(ceil(log2(real(SYN_MEM_DEPTH))))-1 downto 0);
    signal synmem_rdata         : std_logic_vector(SYN_MEM_WIDTH - 1 downto 0);

    -- neuron memory
    signal nrnmem_we            : std_logic;
    signal nrnmem_waddr         : std_logic_vector(integer(ceil(log2(real(NRN_MEM_DEPTH))))-1 downto 0);
    signal nrnmem_wdata         : std_logic_vector(35 downto 0);
    signal nrnmem_re            : std_logic;
    signal nrnmem_raddr         : std_logic_vector(integer(ceil(log2(real(NRN_MEM_DEPTH))))-1 downto 0);
    signal nrnmem_rdata         : std_logic_vector(35 downto 0);

    signal timestep             : std_logic;
    signal write_timestep       : std_logic;

begin

    -- unused signals
    synmem_we       <= '0';
    synmem_waddr    <= (others => '0');
    synmem_wdata    <= (others => '0');

    -- for testbench only
    o_nrnmem_we     <= nrnmem_we;
    o_nrnmem_waddr  <= nrnmem_waddr;
    o_nrnmem_wdata  <= nrnmem_wdata;
    o_sched_tstep   <= timestep;

    o_out_fifo_wdata    <= '0' & lif_fifo_wdata when write_timestep  = '0' else "1000000000000";
    o_out_fifo_we       <= lif_fifo_we when (write_timestep = '0' or IS_LAST = 1) else '1';

    o_in_fifo_empty     <= in_fifo_empty;

    config : process(i_clk)
    begin
        if rising_edge(i_clk) then

            synldr_cfg_en   <= '0';
            nrnldr_cfg_en   <= '0';
            lif_cfg_en      <= '0';
            nrnwrt_cfg_en   <= '0';

            if (i_cfg_en = '1') and (i_rst = '0') then
                case i_cfg_addr(7 downto 4) is
                    when "0000" =>
                        synldr_cfg_en   <= '1';
                        synldr_cfg_addr <= i_cfg_addr(3 downto 0);
                        synldr_cfg_val  <= i_cfg_val;
                    when "0001" =>
                        nrnldr_cfg_en   <= '1';
                        nrnldr_cfg_addr <= i_cfg_addr(3 downto 0);
                        nrnldr_cfg_val  <= i_cfg_val;
                    when "0010" =>
                        lif_cfg_en      <= '1';
                        lif_cfg_addr    <= i_cfg_addr(3 downto 0);
                        lif_cfg_val     <= i_cfg_val;
                    when "0011" =>
                        nrnwrt_cfg_en   <= '1';
                        nrnwrt_cfg_addr <= i_cfg_addr(3 downto 0);
                        nrnwrt_cfg_val  <= i_cfg_val;
                    when others =>
                        null;
                end case;
            end if;
        end if;
    end process;

    SCHEDULER : entity work.SCHEDULER
    port map (
        i_enable            => i_enable,
        i_synldr_busy       => synldr_busy,
        i_nrnldr_busy       => nrnldr_busy,
        o_synldr_start      => synldr_start,
        o_nrnldr_start      => nrnldr_start,
        o_timestep          => timestep,
        o_write_timestep    => write_timestep,
        i_fifo_in_empty     => in_fifo_empty,
        o_fifo_re           => in_fifo_re,
        i_fifo_rdata        => in_fifo_rdata,
        i_re                => synldr_fifo_re,
        o_rdata             => synldr_fifo_rdata,
        i_fifo_out_full     => i_out_fifo_full,
        o_busy              => o_busy,
        i_clk               => i_clk,
        i_rst               => i_rst
    );

    INPUT_FIFO : entity work.BRAM_FIFO
    generic map (
        DEPTH => 256,
        WIDTH => 13     -- 1b timestep and 12b neuron index
    )
    port map (
        i_we                => i_in_fifo_we,
        i_wdata             => i_in_fifo_wdata,
        i_re                => in_fifo_re,
        o_rvalid            => in_fifo_rvalid,
        o_rdata             => in_fifo_rdata,
        o_empty             => in_fifo_empty,
        o_empty_next        => open,
        o_full              => o_in_fifo_full,
        o_full_next         => open,
        o_fill_count        => open,
        i_clk               => i_clk,
        i_rst               => i_rst
    );

    SYN_MEMORY : entity work.DUAL_PORT_BRAM
    generic map (
        DEPTH       => SYN_MEM_DEPTH,
        WIDTH       => SYN_MEM_WIDTH,
        FILENAME    => SYN_INIT_FILE
    )
    port map (
        i_we        => synmem_we,
        i_waddr     => synmem_waddr,
        i_wdata     => synmem_wdata,
        i_re        => synmem_re,
        i_raddr     => synmem_raddr,
        o_rdata     => synmem_rdata,
        i_clk       => i_clk
    );

    NRN_MEMORY : entity work.DUAL_PORT_BRAM
    generic map (
        DEPTH       => NRN_MEM_DEPTH,
        WIDTH       => 36,
        FILENAME    => NRN_INIT_FILE
    )
    port map (
        i_we        => nrnmem_we,
        i_waddr     => nrnmem_waddr,
        i_wdata     => nrnmem_wdata,
        i_re        => nrnmem_re,
        i_raddr     => nrnmem_raddr,
        o_rdata     => nrnmem_rdata,
        i_clk       => i_clk
    );

    SYN_LOADER : entity work.SYNAPSE_LOADER
    generic map (
        SYN_MEM_DEPTH   => SYN_MEM_DEPTH,
        SYN_MEM_WIDTH   => SYN_MEM_WIDTH
    )
    port map (
        i_cfg_en            => synldr_cfg_en,
        i_cfg_addr          => synldr_cfg_addr,
        i_cfg_val           => synldr_cfg_val,
        o_fifo_re           => synldr_fifo_re,
        i_fifo_rvalid       => '1',                 -- unused
        i_fifo_rdata        => synldr_fifo_rdata,
        o_syn_weight        => synldr_weight,
        o_syn_valid         => synldr_valid,
        o_syn_valid_next    => synldr_valid_next,
        o_syn_valid_last    => synldr_valid_last,
        o_synmem_re         => synmem_re,
        o_synmem_raddr      => synmem_raddr,
        i_synmem_rdata      => synmem_rdata,
        i_start             => synldr_start,
        i_continue          => lif_continue,
        o_busy              => synldr_busy,
        i_goto_idle         => lif_goto_idle,
        i_clk               => i_clk,
        i_rst               => i_rst
    );

    NRN_LOADER : entity work.NEURON_LOADER
    generic map (
        NRN_MEM_DEPTH   => NRN_MEM_DEPTH
    )
    port map (
        i_cfg_en            => nrnldr_cfg_en,
        i_cfg_addr          => nrnldr_cfg_addr,
        i_cfg_val           => nrnldr_cfg_val,
        o_nrn_re            => nrnmem_re,
        o_nrn_addr          => nrnmem_raddr,
        i_nrn_data          => nrnmem_rdata,
        o_nrn_state         => nrnldr_state,
        o_nrn_index         => nrnldr_nrn_index,
        o_nrn_valid         => nrnldr_valid,
        o_nrn_valid_next    => nrnldr_valid_next,
        o_nrn_valid_last    => nnrldr_valid_last,
        i_start             => nrnldr_start,
        i_continue          => lif_continue,
        o_busy              => nrnldr_busy,
        i_goto_idle         => lif_goto_idle,
        i_clk               => i_clk,
        i_rst               => i_rst
    );

    LIF : entity work.LIF_NEURON
    port map (
        i_cfg_en            => lif_cfg_en,
        i_cfg_addr          => lif_cfg_addr,
        i_cfg_val           => lif_cfg_val,
        i_nrn_valid         => nrnldr_valid,
        i_nrn_valid_next    => nrnldr_valid_next,
        i_nrn_valid_last    => nnrldr_valid_last,
        i_nrn_state         => nrnldr_state,
        i_syn_valid         => synldr_valid,
        i_syn_valid_next    => synldr_valid_next,
        i_syn_valid_last    => synldr_valid_last,
        i_syn_weight        => synldr_weight,
        i_nrn_index         => nrnldr_nrn_index,
        i_timestep          => timestep,
        o_nrn_state_next    => lif_nrn_state_next,
        o_event_fifo_out    => lif_fifo_wdata,
        o_event_fifo_we     => lif_fifo_we,
        o_output_valid      => lif_out_valid,
        o_continue          => lif_continue,
        o_goto_idle         => lif_goto_idle,
        i_clk               => i_clk,
        i_rst               => i_rst
    );

    NRN_WRITER : entity work.NEURON_WRITER
    generic map (
        NRN_MEM_DEPTH   => NRN_MEM_DEPTH
    )
    port map (
        i_cfg_en    => nrnwrt_cfg_en,
        i_cfg_addr  => nrnwrt_cfg_addr,
        i_cfg_val   => nrnwrt_cfg_val,
        o_nrn_we    => nrnmem_we,
        o_nrn_addr  => nrnmem_waddr,
        o_nrn_data  => nrnmem_wdata,
        i_nrn_state => lif_nrn_state_next,
        i_valid     => lif_out_valid,
        i_nrn_data  => (others => '0'),
        i_clk       => i_clk,
        i_rst       => i_rst
    );

end behavior;
