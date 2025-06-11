library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity conv2d_wrapper is
    generic(
        KERNEL_SIZE: integer := 3;
        IN_CHANNELS: integer := 2;
        OUT_CHANNELS: integer := 2;
        IMG_HEIGHT: integer := 8;
        IMG_WIDTH: integer := 8;

        BITS_PER_KERNEL_WEIGHT: integer := 6;
        BITS_PER_NEURON: integer := 9;
        INPUT_FIFO_EVENT_CAPACITY: integer := 1024;
        BITS_PER_COORDINATE: integer := 4
    );
    port(
        -- control signals
        clk: in std_logic;
        rst_n: in std_logic;
        enable: in std_logic;
        -- input fifo signals
        input_write_enable: in std_logic;
        input_fifo_full_next: out std_logic;
        input_fifo_data: in std_logic_vector(2 * BITS_PER_COORDINATE + IN_CHANNELS downto 0);
        -- output fifo signals
        output_fifo_data: out std_logic_vector((BITS_PER_COORDINATE -1) * 2 + OUT_CHANNELS downto 0);
        output_fifo_full_next: in std_logic;
        output_fifo_write_enable: out std_logic

    );
end conv2d_wrapper;

architecture wrapper of conv2d_wrapper is

begin

    u_conv2d: entity work.CONV2D
        generic map(
            KERNEL_SIZE => KERNEL_SIZE,
            IN_CHANNELS => IN_CHANNELS,
            OUT_CHANNELS => OUT_CHANNELS,
            IMG_HEIGHT => IMG_HEIGHT,
            IMG_WIDTH => IMG_WIDTH,

            BITS_PER_KERNEL_WEIGHT => BITS_PER_KERNEL_WEIGHT,
            BITS_PER_NEURON => BITS_PER_NEURON,
            INPUT_FIFO_EVENT_CAPACITY => INPUT_FIFO_EVENT_CAPACITY,
            BITS_PER_COORDINATE => BITS_PER_COORDINATE
        )
        port map(
            clk => clk,
            rst_n => rst_n,
            enable => enable,

            input_write_enable => input_write_enable,
            input_fifo_full_next => input_fifo_full_next,
            input_fifo_data => input_fifo_data,

            output_fifo_data => output_fifo_data,
            output_fifo_full_next => output_fifo_full_next,
            output_fifo_write_enable => output_fifo_write_enable
        );

end architecture wrapper;