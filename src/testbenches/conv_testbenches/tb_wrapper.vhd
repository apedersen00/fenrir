library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity fast_conv_controller_tb is
end fast_conv_controller_tb;

architecture tb of fast_conv_controller_tb is

    constant COORD_BITS    : integer := 10;
    constant IMG_WIDTH     : integer := 640;
    constant IMG_HEIGHT    : integer := 480;
    constant CHANNELS      : integer := 8;
    constant BITS_PER_CHANNEL : integer := 8;

    signal clk                : std_logic := '0';
    signal rst_n              : std_logic := '0';
    signal sys_enable         : std_logic := '0';
    signal sys_reset          : std_logic := '0';
    signal timestep           : std_logic := '0';

    signal system_active      : std_logic;
    signal fifo_empty         : std_logic;
    signal fifo_full          : std_logic;

    signal spike_event        : std_logic_vector(2*COORD_BITS - 1 downto 0);
    signal write_enable       : std_logic := '0';

    signal output_fifo_full   : std_logic := '0';
    signal output_write_enable: std_logic;
    signal output_fifo_data   : std_logic_vector((COORD_BITS - 1)*2 + CHANNELS downto 0);

    constant clk_period : time := 10 ns;

begin

    uut: entity work.f_conv_layer_wrapper
    generic map (
        COORD_BITS                  => COORD_BITS,
        IMG_WIDTH                   => IMG_WIDTH,
        IMG_HEIGHT                  => IMG_HEIGHT,
        CHANNELS                    => CHANNELS,
        BITS_PER_CHANNEL            => BITS_PER_CHANNEL
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

    clk_process :process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;

    stim_proc: process
    begin
        rst_n <= '0';
        sys_reset <= '1';
        wait for 50 ns;

        rst_n <= '1';
        sys_reset <= '0';
        sys_enable <= '1';
        wait for clk_period*2;

        --timestep <= '1';
        wait for clk_period;
        --timestep <= '0';

        spike_event <= (others => '1');
        write_enable <= '1';
        wait for clk_period;
        write_enable <= '0';

        wait for 200 ns;

        timestep <= '1';
        wait for clk_period;
        timestep <= '0';

        --sys_enable <= '0';
        wait;
    end process;

end architecture;
