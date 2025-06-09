library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity f_conv_layer_wrapper is
    generic(
        COORD_BITS: integer := 8; -- number of bits for coordinates. mainly used for the input event fifo
        IMG_WIDTH: integer :=  40; -- 1/16th of DVexplorer image width
        IMG_HEIGHT: integer := 30; -- 1/16th of DVexplorer image height
        CHANNELS: integer := 6; -- number of output channels  
        BITS_PER_CHANNEL: integer := 6 -- number of bits per channel
    ); -- only has 1 input channel
    port(
        clk: in std_logic;
        rst_n: in std_logic; -- resets states
        sys_enable: in std_logic;
        sys_reset: in std_logic; -- resets everything
        timestep: in std_logic;

        system_active: out std_logic;
        fifo_empty: out std_logic;
        fifo_full: out std_logic;

        spike_event : in std_logic_vector(2 * COORD_BITS - 1 downto 0);
        write_enable: in std_logic;

        output_fifo_full: in std_logic;
        output_write_enable: out std_logic;
        output_fifo_data: out std_logic_vector((COORD_BITS - 1) * 2 + CHANNELS downto 0) -- +1 for the timestep propagation
    );
end f_conv_layer_wrapper;

architecture wrapper of f_conv_layer_wrapper is

    component fast_conv_controller
    generic(
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
    port(
        clk                         : in std_logic;
        rst_n                       : in std_logic;
        sys_enable                  : in std_logic;
        sys_reset                   : in std_logic;
        timestep                    : in std_logic;
        system_active               : out std_logic;
        fifo_empty                  : out std_logic;
        fifo_full                   : out std_logic;
        spike_event                 : in std_logic_vector(2 * COORD_BITS - 1 downto 0);
        write_enable                : in std_logic;
        output_fifo_full            : in std_logic;
        output_write_enable    : out std_logic;
        output_fifo_data            : out std_logic_vector((COORD_BITS - 1) * 2 + CHANNELS downto 0) -- +1 for the timestep propagation
    );
    end component fast_conv_controller;

begin

    u_fast_conv_controller: fast_conv_controller
        generic map(
            COORD_BITS                  => COORD_BITS,
            IMG_WIDTH                   => IMG_WIDTH,
            IMG_HEIGHT                  => IMG_HEIGHT,
            CHANNELS                    => CHANNELS,
            BITS_PER_CHANNEL            => BITS_PER_CHANNEL,
            BRAM_DATA_WIDTH             => CHANNELS * BITS_PER_CHANNEL, -- 32 bits for the BRAM data width
            BRAM_ADDR_WIDTH             => integer(ceil(log2(real(IMG_HEIGHT * IMG_WIDTH)))),
            FIFO_DATA_WIDTH             => 2 * COORD_BITS, -- 2 coordinates for the input event fifo
            INPUT_FIFO_EVENT_CAPACITY   => 1024,
            INPUT_FIFO_ADDR_WIDTH       => integer(ceil(log2(real(1024)))), -- 1024 is the maximum
            OUTPUT_FIFO_DATA_WIDTH      => (COORD_BITS - 1) * 2 + CHANNELS + 1 -- +1 for the timestep propagation
        )
        port map(
            clk                         => clk,
            rst_n                       => rst_n,
            sys_enable                  => sys_enable,
            sys_reset                   => sys_reset,
            timestep                    => timestep,
            system_active               => system_active,
            fifo_empty                  => fifo_empty,
            fifo_full                   => fifo_full,
            spike_event                 => spike_event,
            write_enable                => write_enable,
            output_fifo_full            => output_fifo_full,
            output_write_enable         => output_write_enable,
            output_fifo_data            => output_fifo_data
        );

end architecture wrapper;