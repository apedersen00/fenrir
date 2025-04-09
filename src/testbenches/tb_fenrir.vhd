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

    -- FENRIR memory signals
    signal fnr_ibf_addr        : std_logic_vector(15 downto 0);
    signal fnr_ibf_dout        : std_logic_vector(31 downto 0);
    signal fnr_ibf_din         : std_logic_vector(31 downto 0);
    signal fnr_ibf_we          : std_logic := '0';

    -- TB stimulus signals
    signal tb_ibf_addr     : std_logic_vector(15 downto 0);
    signal tb_ibf_dout     : std_logic_vector(15 downto 0);
    signal tb_ibf_din      : std_logic_vector(31 downto 0);
    signal tb_ibf_we       : std_logic := '0';

    -- Enable TB mode, muxing the stimulus to the IBF memory
    signal tb_mode         : std_logic :=  '0';

    -- Clock period
    constant clk_period : time := 10 ns;

    type spike_array_t is array(0 to 6399) of std_logic_vector(31 downto 0);
    signal spike_data : spike_array_t;

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
            ibf_addr        => fnr_ibf_addr,
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

    -- MUX logic
    ibf_addr <= tb_ibf_addr when tb_mode = '1' else fnr_ibf_addr;
    ibf_we   <= tb_ibf_we   when tb_mode = '1' else '0';
    ibf_din  <= tb_ibf_din;

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
    
        procedure load_all_spike_data(
            signal memory : out spike_array_t
        ) is
            file spike_file     : text open read_mode is "C:/home/university/8-semester/fenrir/src/design_sources/data/spike_data.txt";
            variable line_buf   : line;
            variable word       : std_logic_vector(31 downto 0);
        begin
            for i in 0 to 6399 loop
                exit when endfile(spike_file);
                readline(spike_file, line_buf);
                read(line_buf, word);
                memory(i) <= word;
            end loop;
        end procedure;
        
        procedure load_frame(
            signal clk          : in std_logic;
            signal ibf_addr     : out std_logic_vector(15 downto 0);
            signal ibf_din      : out std_logic_vector(31 downto 0);
            signal ibf_we       : out std_logic;
            signal memory       : in spike_array_t;
            frame_idx           : in integer
        ) is
            variable base : integer := frame_idx * 64;
        begin
            for i in 0 to 63 loop
                ibf_addr <= std_logic_vector(to_unsigned(i, ibf_addr'length));
                ibf_din  <= memory(base + i);
                ibf_we   <= '1';
                wait for clk_period;

                ibf_we <= '0';
                wait for clk_period;
            end loop;
        end procedure;
    
    begin
        -- Reset the DUT
        nRst <= '0';
        wait for clk_period * 2;
        nRst <= '1';
        wait for clk_period;

        load_all_spike_data(spike_data);

        for frame in 0 to 99 loop
            tb_mode <= '1';
            load_frame(clk, tb_ibf_addr, tb_ibf_din, tb_ibf_we, spike_data, frame);
            tb_mode <= '0';
            wait for clk_period * 10;

            data_rdy <= '1';
            wait for clk_period;
            data_rdy <= '0';
            wait until busy = '0';
        end loop;

        wait;
    end process;

end behavior;
