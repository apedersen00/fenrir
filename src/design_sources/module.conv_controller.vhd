library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.conv_types.all;

entity conv_controller is
    generic(
        IMG_WIDTH : integer := 32;
        IMG_HEIGHT : integer := 32;
        FEATURE_MAPS : integer := 1
    );
    port(
        clk : in std_logic;
        rst : in std_logic;

        -- for the 8xConv block
        windows : out window_array_8_t;
        kernel : out kernel_t;

        -- image data
        img_row_data_in : in std_logic_vector(63 downto 0);
        img_row_address : out std_logic_vector(7 downto 0);
        img_write_en : out std_logic;

        -- kernel data
        kernel_data_in : in std_logic_vector(63 downto 0);
        kernel_address : out std_logic_vector(7 downto 0);
        kernel_write_en : out std_logic;

    );
end entity conv_controller;

architecture Behavioral of conv_controller is

    type states is (
        IDLE,
        READ_KERNEL,
        READ_IMG_ROW,
        COMPUTE_CONV
    );

    signal STATE : states := IDLE;

begin

process (clk, rst)


end architecture Behavioral; 