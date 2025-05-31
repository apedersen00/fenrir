library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity FENRIR_WRAPPER is
    port (
        ext_clk                 : in  std_logic;
        sw                      : in  std_logic_vector(3 downto 0);
        led                     : out std_logic_vector(3 downto 0)
    );
end FENRIR_WRAPPER;

architecture rtl of FENRIR_WRAPPER is

    signal fc1_synldr_reg_cfg_0 : std_logic_vector(31 downto 0);
    signal fc1_nrnldr_reg_cfg_0 : std_logic_vector(31 downto 0);
    signal fc1_lif_reg_cfg_0    : std_logic_vector(31 downto 0);
    signal fc1_nrnwrt_reg_cfg_0 : std_logic_vector(31 downto 0);

    signal class_count_0        : std_logic_vector(31 downto 0);
    signal class_count_1        : std_logic_vector(31 downto 0);
    signal class_count_2        : std_logic_vector(31 downto 0);
    signal class_count_3        : std_logic_vector(31 downto 0);
    signal class_count_4        : std_logic_vector(31 downto 0);
    signal class_count_5        : std_logic_vector(31 downto 0);
    signal class_count_6        : std_logic_vector(31 downto 0);
    signal class_count_7        : std_logic_vector(31 downto 0);
    signal class_count_8        : std_logic_vector(31 downto 0);
    signal class_count_9        : std_logic_vector(31 downto 0);

begin

    fc1_synldr_reg_cfg_0 <=
        "00000000"                              &   -- zero padding
        std_logic_vector(to_unsigned(1, 2))     &   -- bits per weight
        std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
        std_logic_vector(to_unsigned(10, 11));      -- neurons per layer

    --
    fc1_nrnldr_reg_cfg_0 <=
        "0000000000"                            &   -- zero padding
        std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
        std_logic_vector(to_unsigned(10, 11));      -- neurons per 

    fc1_lif_reg_cfg_0 <=
        std_logic_vector(to_unsigned(10, 8))    &   -- weight scalar
        std_logic_vector(to_unsigned(230, 12))  &   -- beta
        std_logic_vector(to_unsigned(67, 12));      -- thresholdlayer

    fc1_nrnwrt_reg_cfg_0 <=
        "0000000000"                            &   -- zero padding
        std_logic_vector(to_unsigned(0, 11))    &   -- layer offset
        std_logic_vector(to_unsigned(10, 11));      -- neurons per layer

    FENRIR_inst : entity work.FENRIR_TOP
        port map (
            sysclk                 => ext_clk,
            ctrl                   => sw,
            ps_fifo                => (others => '0'),
            led                    => led,

            i_fc1_synldr_reg_cfg_0 => fc1_synldr_reg_cfg_0,
            i_fc1_nrnldr_reg_cfg_0 => fc1_nrnldr_reg_cfg_0,
            i_fc1_lif_reg_cfg_0    => fc1_lif_reg_cfg_0,
            i_fc1_nrnwrt_reg_cfg_0 => fc1_nrnwrt_reg_cfg_0,

            o_class_count_0        => class_count_0,
            o_class_count_1        => class_count_1,
            o_class_count_2        => class_count_2,
            o_class_count_3        => class_count_3,
            o_class_count_4        => class_count_4,
            o_class_count_5        => class_count_5,
            o_class_count_6        => class_count_6,
            o_class_count_7        => class_count_7,
            o_class_count_8        => class_count_8,
            o_class_count_9        => class_count_9
        );

end rtl;
