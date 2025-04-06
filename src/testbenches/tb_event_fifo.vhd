library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.conv_system.all;

entity tb_event_fifo is
end entity tb_event_fifo;

architecture testbench of tb_event_fifo is

    CONSTANT CLK_PERIOD : time := 10 ns;

    signal clk                      : std_logic := '1';
    signal reset                    : std_logic := '0';
    signal write_enable             : std_logic := '0';
    signal read_enable              : std_logic := '0';
    signal data_in                  : EVENT_R; 
    signal data_out                 : EVENT_R;
    signal flag_full                : std_logic;
    signal flag_empty               : std_logic;

    function make_event_r_from_integers(
        x : integer;
        y : integer;
        polarity : integer;
        timestamp : integer
    ) return EVENT_R is
        variable event_r : EVENT_R;
    begin
        event_r.x := std_logic_vector(to_unsigned(x, pixel_address_out_width));
        event_r.y := std_logic_vector(to_unsigned(y, pixel_address_out_width));
        event_r.polarity := signed(to_signed(polarity, 2));
        event_r.timestamp := std_logic_vector(to_unsigned(timestamp, AER_EVENT_WIDTH - (2*pixel_address_out_width)));
        return event_r;
    end function make_event_r_from_integers;

begin

    clk <= not clk after CLK_PERIOD / 2;

    dut : entity work.aer_fifo
        port map(
            clk             => clk,
            reset           => reset,
            write_enable    => write_enable,
            read_enable     => read_enable,
            data_in         => data_in,
            data_out        => data_out,
            flag_full       => flag_full,
            flag_empty      => flag_empty
        );

    stimulus : process
    begin 

        reset <= '1';
        wait for CLK_PERIOD;
        reset <= '0';

        -- Lets try to write three samples to the FIFO And them read them
        write_enable <= '1';
        read_enable <= '0';

        data_in <= make_event_r_from_integers(1, 2, 1, 0);
        wait for CLK_PERIOD;
        data_in <= make_event_r_from_integers(3, 4, 1, 1);
        wait for CLK_PERIOD;
        data_in <= make_event_r_from_integers(5, 6, 1, 2);
        wait for CLK_PERIOD;
        -- Flag full should be high now 
        assert flag_full = '1' report "FIFO should be full" severity error;
        write_enable <= '0';
        read_enable <= '1';
        wait for CLK_PERIOD;
        -- Read first sample
        assert data_out = make_event_r_from_integers(1, 2, 1, 0) report "First sample should be (1, 2)" severity error;



        wait;

    end process stimulus;
end architecture testbench;


