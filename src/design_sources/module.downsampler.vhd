library ieee;
use ieee.std_logic_1164.all;

use work.conv_system.all;

entity downsampler is 
    port(
        clk : in std_logic;
        timeref_clock : in timeref_clock;
        AER_In : in DUMMY_AER_IN;
        AER_Out : out EVENT_R;
        ready : out std_logic;
        data_ready: in std_logic;

        fifo_out_write_enable : out std_logic;
        fifo_full : in std_logic
    );
end entity downsampler;

architecture behavioral of downsampler is

    type STATES is (
        IDLE,
        PROCESSING,
        OUTPUT
    );

    signal current_state : STATES := IDLE;

begin

    process (clk)
    begin

        if rising_edge(clk) then

            CASE current_state is
            WHEN IDLE => 
                if data_ready = '1' then
                    ready <= '1';
                    current_state <= PROCESSING;
                end if;

                fifo_out_write_enable <= '0';

            WHEN PROCESSING =>

                AER_OUT <= downsample_input(AER_In, timeref_clock);
                ready <= '0';
                fifo_out_write_enable <= '1';
                current_state <= OUTPUT;

            WHEN OUTPUT =>
                if fifo_full = '0' then
                    current_state <= IDLE;
                end if;

            END case;

        end if;

    end process;

end architecture behavioral;