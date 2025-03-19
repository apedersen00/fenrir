library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity conv is

    port (
        clk: in std_logic;
        image_buffer_data_in: in std_logic_vector(63 downto 0);
        image_buffer_address: out std_logic_vector(7 downto 0);
        conv_out: out std_logic_vector(31 downto 0);
        out_address: out std_logic_vector(7 downto 0);
        out_write_enable: out std_logic;
        ready : out std_logic;
        enable : in std_logic;
        -- add kernel ram ports later
        kernel_in: in std_logic_vector(31 downto 0)
    );

end entity conv;

architecture Behavioral of conv is

    type STATES is (
        IDLE,
        READ_ROWS,
        CONVOLVE
    );

    signal row1, row2, row3 : std_logic_vector(63 downto 0);
    signal STATE: STATES := IDLE;
    signal row_counter: integer := 0;
    signal read_row_counter: integer := 0;

    procedure read_rows is 
    begin
        if read_row_counter = 0 then 
            image_buffer_address <= std_logic_vector(to_unsigned(row_counter, 8));
            read_row_counter <= read_row_counter + 1;
            row_counter <= row_counter + 1;
        elsif read_row_counter = 1 then
            image_buffer_address <= std_logic_vector(to_unsigned(row_counter, 8));
            row1 <= image_buffer_data_in;
            read_row_counter <= read_row_counter + 1;
            row_counter <= row_counter + 1;
        elsif read_row_counter = 2 then
            image_buffer_address <= std_logic_vector(to_unsigned(row_counter, 8));
            row2 <= image_buffer_data_in;
            read_row_counter <= read_row_counter + 1;
            row_counter <= row_counter + 1;
        elsif read_row_counter = 3 then
            row3 <= image_buffer_data_in;
            read_row_counter <= 0;
            STATE <= CONVOLVE;
        end if;
    end read_rows;

    function getPixel(row: std_logic_vector(63 downto 0); column: integer) return std_logic_vector is
    begin
        return signed(row(column*2 downto column*2 - 1));
    end getPixel;

    procedure convolve(column_id: integer; conv_out_idx: integer) is 

        type window is array(0 to 2, 0 to 2) of signed (1 downto 0);
        variable win: window;

    begin
        -- fill window
        win(0, 0) := getPixel(row1, column_id - 1);
        win(0, 1) := getPixel(row1, column_id);
        win(0, 2) := getPixel(row1, column_id + 1);

        win(1, 0) := getPixel(row2, column_id - 1);
        win(1, 1) := getPixel(row2, column_id);
        win(1, 2) := getPixel(row2, column_id + 1);

        win(2, 0) := getPixel(row3, column_id - 1);
        win(2, 1) := getPixel(row3, column_id);
        win(2, 2) := getPixel(row3, column_id + 1);

        -- convolve

        conv_out(conv_out_idx+3 downto conv_out_idx) <= 
            win(0,0) * kernel_in(1 downto 0) +
            win(0,1) * kernel_in(3 downto 2) +
            win(0,2) * kernel_in(5 downto 4) +
            win(1,0) * kernel_in(7 downto 6) +
            win(1,1) * kernel_in(9 downto 8) +
            win(1,2) * kernel_in(11 downto 10) +
            win(2,0) * kernel_in(13 downto 12) +
            win(2,1) * kernel_in(15 downto 14) +
            win(2,2) * kernel_in(17 downto 16);

    end convolve;

begin

process (clk)
begin
    if rising_edge(clk) then
        case STATE is
            WHEN IDLE =>
                if enable = '1' then
                    STATE <= READ_ROWS;
                    ready <= '0';
                end if;
            WHEN READ_ROWS =>
                read_rows;
            WHEN CONVOLVE => 
                convolve(1, 0);
                convolve(5, 4);
                convolve(9, 8);
                convolve(13, 12);
                convolve(17, 16);
                convolve(21, 20);
                convolve(25, 24);
                convolve(29, 28);
                STATE <= IDLE;
                ready <= '1';
        end case;
    end if;
end process;

end architecture Behavioral;