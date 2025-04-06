library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
use ieee.std_logic_textio.all;

entity fenrir_tb is
end fenrir_tb;

architecture behavior of fenrir_tb is
    -- controller signals
    signal clk             : std_logic := '0';
    signal nRst            : std_logic := '0';
    signal busy            : std_logic;
    signal data_rdy        : std_logic := '0';

    signal ibf_addr        : std_logic_vector(15 downto 0);
    signal ibf_dout        : std_logic_vector(31 downto 0);
    signal ibf_din         : std_logic_vector(31 downto 0);
    signal ibf_we          : std_logic := '0';

    -- Clock period
    constant clk_period : time := 10 ns;

procedure load_spike_data (
    signal clk      : in std_logic;
    signal ibf_addr : out std_logic_vector(15 downto 0);
    signal ibf_dout : out std_logic_vector(31 downto 0);
    signal ibf_din  : out std_logic_vector(31 downto 0);
    signal ibf_we   : out std_logic;
    file spike_file : text;
    start_line      : in integer;
) is
    variable line_buffer : line;
    variable data_word   : std_logic_vector(31 downto 0);
begin
    -- Skip to the desired starting line
    for skip in 0 to start_line - 1 loop
        exit when endfile(f);
        readline(f, line_buf); -- discard
    end loop;

    -- Load 64 lines
    for i in 0 to 63 loop
        exit when endfile(f);
        readline(f, line_buf);
        hread(line_buf, word);

        ibf_addr <= std_logic_vector(to_unsigned(i, ibf_addr'length));
        ibf_din  <= word;
        ibf_we   <= '1';
        wait for clk_period;

        ibf_we <= '0';
        wait for clk_period;
    end loop;
end procedure;

begin
    -- instantiate controller
    uut: entity work.fenrir
        generic map (
            IN_SIZE         => 1024,
            NUM_NRN         => 10
        )
        port map (
            clk             => clk,
            nRst            => nRst,
            busy            => busy,
    
            data_rdy        => data_rdy,
            ibf_addr        => ibf_addr,
            ibf_in          => ibf_dout
        );

    ibf_mem: entity work.bram_mem
        generic map (
            G_DEBUG                 => false,
            G_DEBUG_COUNTER_INIT    => 0,
            DEPTH                   => 1024 / 16 + 1,
            WIDTH                   => 32,
            WIDTH_ADDR              => 16,
            FILENAME                => "data/ibf_init.data"
        )
        port map (
            clk         => clk,
            we          => ibf_we,
            addr        => ibf_addr,
            din         => ibf_din,
            dout        => ibf_dout
        );

    -- Generate clock signal
    clk_process : process
    begin
        clk <= '0';
        wait for clk_period / 2;
        clk <= '1';
        wait for clk_period / 2;
    end process;

    -- Stimulus process
    stimulus: process

    file spike_file         : text open read_mode is "data/spike_data.txt";
    variable line_buffer    : line;
    variable data_word      : std_logic_vector(31 downto 0);

    begin
        -- Reset the DUT
        nRst <= '0';
        wait for clk_period * 2;
        nRst <= '1';
        wait for clk_period;

        for block in 0 to 99 loop
            load_spike_data(clk, ibf_addr, ibf_dout, ibf_din, ibf_we, spike_file, block * 64);
            wait for clk_period * 10;

            data_rdy <= '1';
            wait for clk_period;

            loop
                exit when busy = '0';
                data_rdy <= '0';
                wait for clk_period;
            end loop;

            wait for clk_period * 10;
            
        end loop;

        wait;
    end process;

end behavior;
