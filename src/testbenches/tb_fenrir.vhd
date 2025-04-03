library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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
    begin
        -- Reset the DUT
        nRst <= '0';
        wait for clk_period * 2;
        nRst <= '1';
        wait for clk_period;

        -- Test 1: When data is ready, controller should go into READ state
        data_rdy <= '1';
        wait for clk_period;

        -- Simulate reading and incrementing address
        for i in 0 to 1800 loop
            data_rdy <= '0';
            wait for clk_period;
        end loop;

        data_rdy <= '1';
        wait for clk_period;

        -- Simulate reading and incrementing address
        for i in 0 to 255 loop
            data_rdy <= '0';
            wait for clk_period;
        end loop;

        wait;
    end process;

end behavior;
