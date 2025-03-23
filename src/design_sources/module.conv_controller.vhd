library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

use work.conv_types.all;

entity conv_controller is
    generic(
        IMG_WIDTH : integer := 32;
        IMG_HEIGHT : integer := 32;
        FEATURE_MAPS : integer := 1
    );
    port(
        clk : in std_logic;
        --rst : in std_logic;
        data_ready : in std_logic;
        
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
        KERNEL_READ_REQ,
        KERNEL_READ_DATA,
        IMG_ROW_REQ_ROW,
        IMG_ROW_REQ_AND_READ,
        COMPUTE_CONV
    );

    signal STATE : states := IDLE;
    
begin

process (clk)
    variable kernel_counter : integer := 0;
    variable request_kernel : boolean := true;
    variable 
begin
    if rising_edge(clk) then
        case STATE is

            WHEN IDLE =>
                if data_ready = '1' then
                    STATE <= KERNEL_READ_REQ;
                end if;

            WHEN KERNEL_READ_REQ =>

                kernel_address <= std_logic_vector(to_unsigned(kernel_counter, 8));
                STATE <= KERNEL_READ_DATA;

            WHEN KERNEL_READ_DATA =>

                kernel <= kernel_ram_to_kernel_t(kernel_data_in);
                kernel_counter := kernel_counter + 1;
                STATE <= IMG_REQ_ROW;

            WHEN IMG_ROW_REQ_ROW =>

                -- do math on img width later but for now its fine


            WHEN IMG_ROW_REQ_AND_READ =>
            WHEN COMPUTE_CONV =>
            WHEN OTHERS =>

        end case;
        
    end if;
    -- 
end process;

end architecture Behavioral; 