---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: neuron_writer.vhd
--  Description: Module for writing next neuron state from LIF logic to memory.
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
--          - <11b> [10:0]  layer_size          : number of neurons in the layer
--          - <11b> [21:11] layer_offset        : neuron address layer offset
--
---------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

--  Instantiation Template:
--  INST_NAME : entity work.FC_NEURON_WRITER
--  generic map (
--      NRN_MEM_DEPTH   =>
--  )
--  port map (
--      i_reg_cfg_0 =>
--      -- neuron memory interface
--      o_nrn_we    =>
--      o_nrn_addr  =>
--      o_nrn_data  =>
--      -- lif interface
--      i_nrn_state =>
--      i_valid     =>
--      -- neuron loader interface
--      i_nrn_data  =>
--      -- control signals
--      i_clk       =>
--      i_rst       =>
--  );

entity FC_NEURON_WRITER is
    generic (
        NRN_MEM_DEPTH   : integer
    );
    port (
        -- configuration interface
        i_reg_cfg_0 : in std_logic_vector(31 downto 0);

        -- neuron memory interface
        o_nrn_we    : out std_logic;
        o_nrn_addr  : out std_logic_vector(integer(ceil(log2(real(NRN_MEM_DEPTH))))-1 downto 0);
        o_nrn_data  : out std_logic_vector(35 downto 0);     -- neuron memory data out (3x12b)

        -- lif interface
        i_nrn_state : in std_logic_vector(11 downto 0);
        i_valid     : in std_logic;

        -- neuron loader interface
        i_nrn_data  : in std_logic_vector(35 downto 0);

        -- control signals
        i_clk       : in std_logic;
        i_rst       : in std_logic
    );
end FC_NEURON_WRITER;

architecture Behavioral of FC_NEURON_WRITER is

    -- configuration
    signal cfg_layer_size       : std_logic_vector(10 downto 0);    -- number of neurons in the layer

    signal reg_nrn_valid_0      : std_logic;
    signal reg_nrn_valid_1      : std_logic;
    signal reg_nrn_valid_2      : std_logic;

    signal reg_nrn_state_0      : std_logic_vector(11 downto 0);
    signal reg_nrn_state_1      : std_logic_vector(11 downto 0);
    signal reg_nrn_state_2      : std_logic_vector(11 downto 0);

    signal nrn_we               : std_logic;
    signal nrn_data             : std_logic_vector(35 downto 0);    -- packed vector of neuron data for output

    signal nrn_index            : integer range -1 to 2047;
    signal nrn_addr_cntr        : integer range 0 to 1024;

begin

    -- configuration decoding
    cfg_layer_size      <= i_reg_cfg_0(10 downto 0);

    o_nrn_we            <= nrn_we;
    o_nrn_addr          <= std_logic_vector(to_unsigned(nrn_addr_cntr, o_nrn_addr'length));
    o_nrn_data          <= nrn_data;

    addr_incr : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if (i_rst = '1') then
                nrn_addr_cntr <= 0;
            else
                if (nrn_we = '1') then
                    if (nrn_addr_cntr >= NRN_MEM_DEPTH - 1) then
                        nrn_addr_cntr <= 0;
                    else
                        nrn_addr_cntr <= nrn_addr_cntr + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    -- put valid next neuron states into registers and pack into 36-bit vector
    write_reg : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then

                nrn_index       <= -1;
                reg_nrn_state_0 <= (others => '0');
                reg_nrn_state_1 <= (others => '0');
                reg_nrn_state_2 <= (others => '0');
                reg_nrn_valid_0 <= '0';
                reg_nrn_valid_1 <= '0';
                reg_nrn_valid_2 <= '0';

            elsif i_valid = '1' then

                nrn_index <= nrn_index + 1;

                if reg_nrn_valid_0 = '0' then
                    reg_nrn_state_0 <= i_nrn_state;
                    reg_nrn_valid_0 <= '1';
                elsif reg_nrn_valid_1 = '0' then
                    reg_nrn_state_1 <= i_nrn_state;
                    reg_nrn_valid_1 <= '1';
                elsif reg_nrn_valid_2 = '0' then
                    reg_nrn_state_2 <= i_nrn_state;
                    reg_nrn_valid_2 <= '1';
                end if;

            elsif (nrn_index + 1 >= unsigned(cfg_layer_size)) and (unsigned(cfg_layer_size) /= 0) then

                nrn_index <= -1;

                if reg_nrn_valid_0 = '0' then
                    reg_nrn_state_0 <= i_nrn_data(35 downto 24);
                    reg_nrn_valid_0 <= '1';
                end if;
                if reg_nrn_valid_1 = '0' then
                    reg_nrn_state_1 <= i_nrn_data(23 downto 12);
                    reg_nrn_valid_1 <= '1';
                end if;
                if reg_nrn_valid_2 = '0' then
                    reg_nrn_state_2 <= i_nrn_data(11 downto 0);
                    reg_nrn_valid_2 <= '1';
                end if;

            end if;

            if reg_nrn_valid_0 = '1' and
               reg_nrn_valid_1 = '1' and
               reg_nrn_valid_2 = '1' then

                nrn_we <= '1';
                nrn_data <= reg_nrn_state_2 &
                            reg_nrn_state_1 &
                            reg_nrn_state_0;

                reg_nrn_valid_0 <= '0';
                reg_nrn_valid_1 <= '0';
                reg_nrn_valid_2 <= '0';
            else
                nrn_we <= '0';
                nrn_data <= (others => '0');
            end if;

        end if;
    end process;

end Behavioral;
