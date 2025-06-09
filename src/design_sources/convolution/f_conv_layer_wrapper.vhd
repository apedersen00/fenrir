library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fast_conv_controller_wrapper is
    generic (
        COORD_BITS                  : integer := 10;
        IMG_WIDTH                   : integer := 640;
        IMG_HEIGHT                  : integer := 480;
        CHANNELS                    : integer := 8;
        BITS_PER_CHANNEL            : integer := 8;
        INPUT_FIFO_EVENT_CAPACITY   : integer := 1024
    );
    port (
        clk                 : in std_logic;
        rst_n               : in std_logic;
        sys_enable          : in std_logic;
        sys_reset           : in std_logic;
        timestep            : in std_logic;

        system_active       : out std_logic;
        fifo_empty          : out std_logic;
        fifo_full           : out std_logic;

        spike_event         : in std_logic_vector(2*COORD_BITS - 1 downto 0);
        write_enable        : in std_logic;

        output_fifo_full    : in std_logic;
        output_write_enable : out std_logic;
        output_fifo_data    : out std_logic_vector((COORD_BITS - 1)*2 + CHANNELS downto 0)
    );
end entity;

architecture rtl of fast_conv_controller_wrapper is

    component fast_conv_controller
        generic (
            COORD_BITS                  : integer;
            IMG_WIDTH                   : integer;
            IMG_HEIGHT                  : integer;
            CHANNELS                    : integer;
            BITS_PER_CHANNEL            : integer;
            BRAM_DATA_WIDTH             : integer;
            BRAM_ADDR_WIDTH             : integer;
            FIFO_DATA_WIDTH             : integer;
            INPUT_FIFO_EVENT_CAPACITY   : integer;
            INPUT_FIFO_ADDR_WIDTH       : integer;
            OUTPUT_FIFO_DATA_WIDTH      : integer
        );
        port (
            clk                 : in std_logic;
            rst_n               : in std_logic;
            sys_enable          : in std_logic;
            sys_reset           : in std_logic;
            timestep            : in std_logic;

            system_active       : out std_logic;
            fifo_empty          : out std_logic;
            fifo_full           : out std_logic;

            spike_event         : in std_logic_vector(2*COORD_BITS - 1 downto 0);
            write_enable        : in std_logic;

            output_fifo_full    : in std_logic;
            output_write_enable : out std_logic;
            output_fifo_data    : out std_logic_vector((COORD_BITS - 1)*2 + CHANNELS downto 0)
        );
    end component;

begin

    u_fast_conv_controller : fast_conv_controller
        generic map (
            COORD_BITS                => COORD_BITS,
            IMG_WIDTH                 => IMG_WIDTH,
            IMG_HEIGHT                => IMG_HEIGHT,
            CHANNELS                  => CHANNELS,
            BITS_PER_CHANNEL          => BITS_PER_CHANNEL,
            BRAM_DATA_WIDTH           => CHANNELS * BITS_PER_CHANNEL,
            BRAM_ADDR_WIDTH           => integer(ceil(log2(real(IMG_WIDTH * IMG_HEIGHT)))),
            FIFO_DATA_WIDTH           => 2 * COORD_BITS,
            INPUT_FIFO_EVENT_CAPACITY => INPUT_FIFO_EVENT_CAPACITY,
            INPUT_FIFO_ADDR_WIDTH     => integer(ceil(log2(real(INPUT_FIFO_EVENT_CAPACITY)))),
            OUTPUT_FIFO_DATA_WIDTH    => (COORD_BITS - 1) * 2 + CHANNELS + 1
        )
        port map (
            clk                 => clk,
            rst_n               => rst_n,
            sys_enable          => sys_enable,
            sys_reset           => sys_reset,
            timestep            => timestep,

            system_active       => system_active,
            fifo_empty          => fifo_empty,
            fifo_full           => fifo_full,

            spike_event         => spike_event,
            write_enable        => write_enable,

            output_fifo_full    => output_fifo_full,
            output_write_enable => output_write_enable,
            output_fifo_data    => output_fifo_data
        );

end architecture;
