---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: bram_multiplexer.vhd
--  Description: Multiplexer for PS/PL access to BRAM.
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

--  Instantiation Template:
--  INST_NAME : entity work.BRAM_MUX
--  generic map (
--      RAM_WIDTH   =>
--      RAM_DEPTH   =>
--      PS_DEPTH    =>
--  )
--  port map (
--      i_clk               =>
--      i_sel               =>
--      -- BRAM interface
--      o_we                =>
--      o_waddr             =>
--      o_wdata             =>
--      o_re                =>
--      o_raddr             =>
--      i_rdata             =>
--      -- BRAM port A interface PS
--      i_ps_bram_addr_a    =>
--      i_ps_bram_clk_a     =>
--      i_ps_bram_wrdata_a  =>
--      o_ps_bram_rddata_a  =>
--      i_ps_bram_en_a      =>
--      i_ps_bram_rst_a     =>
--      i_ps_bram_we_a      =>
--      -- BRAM interface PL
--      i_pl_we             =>
--      i_pl_waddr          =>
--      i_pl_wdata          =>
--      i_pl_re             =>
--      i_pl_raddr          =>
--      o_pl_rdata          =>
--  );

entity BRAM_MUX is
    generic (
        RAM_WIDTH   : integer;
        RAM_DEPTH   : integer;
        PS_DEPTH    : integer
    );
    port (
        i_clk               : in std_logic;
        i_sel               : in std_logic;

        -- BRAM interface
        o_we                : out std_logic;
        o_waddr             : out std_logic_vector(integer(ceil(log2(real(RAM_DEPTH)))) - 1 downto 0);
        o_wdata             : out std_logic_vector(RAM_WIDTH - 1 downto 0);
        o_re                : out std_logic;
        o_raddr             : out std_logic_vector(integer(ceil(log2(real(RAM_DEPTH)))) - 1 downto 0);
        i_rdata             : in std_logic_vector(RAM_WIDTH - 1 downto 0);
        o_clk               : out std_logic;

        -- BRAM port A interface PS
        i_ps_bram_addr_a    : in std_logic_vector(integer(ceil(log2(real(PS_DEPTH)))) - 1 downto 0);
        i_ps_bram_clk_a     : in std_logic;
        i_ps_bram_wrdata_a  : in std_logic_vector(RAM_WIDTH - 1 downto 0);
        o_ps_bram_rddata_a  : out std_logic_vector(RAM_WIDTH - 1 downto 0);
        i_ps_bram_en_a      : in std_logic;
        i_ps_bram_rst_a     : in std_logic;
        i_ps_bram_we_a      : in std_logic_vector((RAM_WIDTH / 8) - 1 downto 0);

        -- BRAM interface PL
        i_pl_we             : in std_logic;
        i_pl_waddr          : in std_logic_vector(integer(ceil(log2(real(RAM_DEPTH)))) - 1 downto 0);
        i_pl_wdata          : in std_logic_vector(RAM_WIDTH - 1 downto 0);
        i_pl_re             : in std_logic;
        i_pl_raddr          : in std_logic_vector(integer(ceil(log2(real(RAM_DEPTH)))) - 1 downto 0);
        o_pl_rdata          : out std_logic_vector(RAM_WIDTH - 1 downto 0);
        i_pl_clk            : in std_logic
    );
end BRAM_MUX;

architecture behavior of BRAM_MUX is

    constant BRAM_ADDR_WIDTH : integer := integer(ceil(log2(real(RAM_DEPTH))));

begin

    mux : process(i_clk)
    begin
        if rising_edge(i_clk) then

            case i_sel is
                when '0' =>
                    o_we        <= (i_ps_bram_we_a(0) or i_ps_bram_we_a(1) or i_ps_bram_we_a(2) or i_ps_bram_we_a(3));
                    o_waddr     <= i_ps_bram_addr_a(BRAM_ADDR_WIDTH - 1 downto 0);
                    o_wdata     <= i_ps_bram_wrdata_a(RAM_WIDTH - 1 downto 0);
                    o_re        <= '0';
                    o_raddr     <= (others => '0');
                    o_clk       <= i_ps_bram_clk_a;
                when '1' =>
                    o_we        <= i_pl_we;
                    o_waddr     <= i_pl_waddr;
                    o_wdata     <= i_pl_wdata;
                    o_re        <= i_pl_re;
                    o_raddr     <= i_pl_raddr;
                    o_pl_rdata  <= i_rdata;
                    o_clk       <= i_pl_clk;
            end case;

        end if;
    end process;

end behavior;
