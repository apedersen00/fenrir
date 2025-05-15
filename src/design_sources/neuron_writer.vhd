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
--  INST_NAME : entity work.NEURON_WRITER
--  generic map (
--      NRN_MEM_DEPTH   =>
--  )
--  port map (
--      i_cfg_en    =>
--      i_cfg_addr  =>
--      i_cfg_val   =>
--      o_nrn_we    =>
--      o_nrn_addr  =>
--      o_nrn_data  =>
--      i_nrn_state =>
--      i_valid     =>
--      i_last      =>
--      i_nrn_data  =>
--      i_clk       =>
--      i_rst       =>
--      o_fault     =>
--  );

entity NEURON_WRITER is
    generic (
        NRN_MEM_DEPTH   : integer                           -- depth of the neuron memory
    );
    port (
        -- configuration interface
        i_cfg_en    : in std_logic;                         -- enable configuration
        i_cfg_addr  : in std_logic_vector(3 downto 0);      -- register to configure
        i_cfg_val   : in std_logic_vector(31 downto 0);     -- value to configure

        -- neuron memory interface
        o_nrn_we    : out std_logic;                        -- neuron memory read enable
        o_nrn_addr  : out std_logic_vector(integer(ceil(log2(real(NRN_MEM_DEPTH))))-1 downto 0);
        o_nrn_data  : out std_logic_vector(35 downto 0);     -- neuron memory data out (3x12b)

        -- lif interface
        i_nrn_state : in std_logic_vector(11 downto 0);
        i_valid     : in std_logic;

        -- neuron loader interface
        i_nrn_data  : in std_logic_vector(35 downto 0);

        -- control signals
        i_clk       : in std_logic;
        i_rst       : in std_logic;
        o_fault     : out std_logic
    );
end NEURON_WRITER;

architecture Behavioral of NEURON_WRITER is

    -- registers
    signal reg_cfg_0            : std_logic_vector(31 downto 0);    -- configuration register 0

    -- configuration
    signal cfg_layer_size       : std_logic_vector(10 downto 0);    -- number of neurons in the layer
    signal cfg_layer_offset     : std_logic_vector(10 downto 0);    -- neuron address layer offset

    signal reg_nrn_valid_0      : std_logic;
    signal reg_nrn_valid_1      : std_logic;
    signal reg_nrn_valid_2      : std_logic;

    signal reg_nrn_state_0      : std_logic_vector(11 downto 0);
    signal reg_nrn_state_1      : std_logic_vector(11 downto 0);
    signal reg_nrn_state_2      : std_logic_vector(11 downto 0);

    signal nrn_we               : std_logic;
    signal nrn_data             : std_logic_vector(35 downto 0);    -- packed vector of neuron data for output

    signal nrn_index            : integer range -1 to 1024;
    signal nrn_addr_cntr        : integer range 0 to 512;

begin

    -- configuration decoding
    cfg_layer_size      <= reg_cfg_0(10 downto 0);
    cfg_layer_offset    <= reg_cfg_0(21 downto 11);

    o_nrn_we            <= nrn_we;
    o_nrn_addr          <= std_logic_vector(to_unsigned(nrn_addr_cntr, o_nrn_addr'length));
    o_nrn_data          <= nrn_data;

    addr_incr : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if (nrn_we = '1') then
                nrn_addr_cntr <= nrn_addr_cntr + 1;
            end if;
        end if;
    end process;

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
                else
                    -- overflow :(
                    o_fault <= '1';
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
