library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fifo_tb is
end fifo_tb;

architecture behavior of fifo_tb is

    constant clk_period : time := 10 ns;
    constant DEPTH      : integer := 128;
    constant WIDTH      : integer := 32;

    signal we           : std_logic;
    signal wdata        : std_logic_vector(WIDTH - 1 downto 0);
    signal re           : std_logic;
    signal rvalid       : std_logic;
    signal rdata        : std_logic_vector(WIDTH - 1 downto 0);
    signal empty        : std_logic;
    signal empty_next   : std_logic;
    signal full         : std_logic;
    signal full_next    : std_logic;
    signal fill_count   : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal clk          : std_logic;
    signal rst          : std_logic;
    signal fault        : std_logic;

begin

    DUT : entity work.fifo
        generic map (
            DEPTH => DEPTH,
            WIDTH => WIDTH
        )
        port map (
            i_we            => we,
            i_wdata         => wdata,
            i_re            => re,
            o_rvalid        => rvalid,
            o_rdata         => rdata,
            o_empty         => empty,
            o_empty_next    => empty_next,
            o_full          => full,
            o_full_next     => full_next,
            o_fill_count    => fill_count,
            i_clk           => clk,
            i_rst           => rst,
            o_fault         => fault
        );

    clk <= not clk after clk_period / 2;

    rst     <= '1';
    we      <= '0';
    wdata   <= (others => '0');
    re      <= '0';

    PROC_SEQUENCER : process
    begin

        wait for 10 * clk_period;
        rst <= '0';
        wait until rising_edge(clk);

        -- start writing
        we <= '1';

        -- fill the FIFO
        while full_next = '0' loop
            wdata <= std_logic_vector(unsigned(wdata) + 1);
            wait until rising_edge(clk);
        end loop;

        -- stop writing
        we <= '0';

        -- empty the FIFO
        re <= '1';
        wait until empty_next = '1';

        wait for 10 * clk_period;
        finish;
    end process;

end behavior;
