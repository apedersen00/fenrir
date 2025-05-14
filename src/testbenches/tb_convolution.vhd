library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_conv is
end entity tb_conv;

architecture testbench of tb_conv is

    CONSTANT CLK_PERIOD        : time    := 10 ns;
    CONSTANT AER_BUS_WIDTH     : integer := 32;
    CONSTANT COORD_WIDTH       : integer := 6;
    CONSTANT TIME_STAMP_WIDTH  : integer := 20;
    CONSTANT IMG_HEIGHT        : integer := 64;
    CONSTANT IMG_WIDTH         : integer := 64;
    CONSTANT KERNEL_SIZE       : integer := 5;
    CONSTANT DESC_LENGTH       : integer := 80;

    signal clk                 : std_logic := '1';
    signal reset_o             : std_logic := '0';
    signal enable_o            : std_logic := '0';

    -- fifo signals
    signal aer_fifo_bus_o      : std_logic_vector(AER_BUS_WIDTH - 1 downto 0) := (others => '0');
    signal aer_empty_o         : std_logic := '1';
    signal aer_fifo_read_i     : std_logic;

    subtype description_t is string(1 to DESC_LENGTH);
    type test_case_t is record
        x           : integer range 0 to 2**COORD_WIDTH-1;
        y           : integer range 0 to 2**COORD_WIDTH-1;
        time_stamp  : integer range 0 to 2**TIME_STAMP_WIDTH-1;
        description : description_t;
    end record;

    type test_cases_array_t is array (natural range <>) of test_case_t;
    

    function pad_string(s: string) return description_t is
        variable result : description_t := (others => ' ');  -- Initialize with spaces
    begin
        if s'length <= DESC_LENGTH then
            result(1 to s'length) := s;
        else
            result := s(1 to DESC_LENGTH);  -- Truncate if too long
        end if;
        return result;
    end function;
    constant test_cases : test_cases_array_t := (
        -- Regular case, center of image
        (x => 32, y => 32, time_stamp => 100, description => pad_string("Center of image")),
        
        -- Edge cases
        (x => 0, y => 0, time_stamp => 200, description => pad_string("Top-left corner")),
        (x => IMG_WIDTH - 1, y => IMG_HEIGHT - 1, time_stamp => 300, description => pad_string("Bottom-right corner")),
        (x => 0, y => IMG_HEIGHT - 1, time_stamp => 400, description => pad_string("Top-right corner")),
        (x => IMG_WIDTH - 1, y => 0, time_stamp => 500, description => pad_string("Bottom-left corner")),
        
        -- Near-edge cases (kernel will partially go outside image)
        (x => 1, y => 1, time_stamp => 600, description => pad_string("Near top-left corner")),
        (x => IMG_WIDTH - 2, y => IMG_HEIGHT - 2, time_stamp => 700, description => pad_string("Near bottom-right corner"))
    );

begin
    
    -- UUT
    uut : entity work.Reverse_Convolution_Layer
        generic map(
            AerBusWidth => AER_BUS_WIDTH,
            CoordinateWidth => COORD_WIDTH,
            TimeStampWidth => TIME_STAMP_WIDTH,
            KernelSizeOneAxis => KERNEL_SIZE,
            ImageWidth => IMG_WIDTH,
            ImageHeight => IMG_HEIGHT,
            MemoryAddressWidth => 11
        )
        port map(
            clk                => clk,
            reset_i            => reset_o,
            enable_i           => enable_o,
            aer_fifo_bus_i     => aer_fifo_bus_o,
            aer_empty_i        => aer_empty_o,
            aer_fifo_read_o    => aer_fifo_read_i
        );

    -- Clock Generation
    clk <= not clk after CLK_PERIOD / 2;

    -- Test reset
    stimulus : process

        procedure waitf(n : in integer) is
        begin
            for i in 1 to n loop
                wait for CLK_PERIOD;
            end loop;
        end procedure;

        function set_bus(
            x: integer;
            y: integer;
            time_stamp: integer
        ) return std_logic_vector is
        begin
            return std_logic_vector(to_unsigned(x, COORD_WIDTH)) 
                    & std_logic_vector(to_unsigned(y, COORD_WIDTH)) 
                    & std_logic_vector(to_unsigned(time_stamp, TIME_STAMP_WIDTH));
        end function;

        procedure simulate_fifo_read(
            test_case   : in test_case_t;
            wait_cycles : in integer := 10
        ) is
        begin
            -- Set FIFO not empty
            aer_empty_o <= '0';
            
            -- Wait for the DUT to request a read
            wait until aer_fifo_read_i = '1';
            
            -- Wait 1 clock cycle (simulating FIFO latency)
            waitf(1);
            
            -- Put test data on the bus
            aer_fifo_bus_o <= set_bus(test_case.x, test_case.y, test_case.time_stamp);
            
            -- Set FIFO as empty again (no more data)
            aer_empty_o <= '1';
            
            -- Wait specified number of cycles for processing
            waitf(wait_cycles);
            
            -- Report test case
            report "Tested: " & test_case.description;
        end procedure;

        procedure run_all_test_cases(
            processing_time : in integer := 10
        ) is
        begin
            for i in test_cases'range loop
                simulate_fifo_read(test_cases(i), processing_time);
            end loop;
        end procedure;

    begin
        -- Reset the system
        reset_o <= '1';
        waitf(5);
        reset_o <= '0';
        waitf(2);
        -- Enable the system
        enable_o <= '1';
        waitf(2);
        -- Disable the system
        enable_o <= '0';
        aer_empty_o <= '0';
        waitf(2);
        -- Enable the system again
        aer_empty_o <= '1';
        enable_o <= '1';

        waitf(2);

        -- Run individual test case
        report "Running individual test case";
        simulate_fifo_read(test_cases(0));
        
        -- Run all test cases
        report "Running all test cases";
        run_all_test_cases(15);
        
        -- Test with different processing time
        report "Running tests with longer processing time";
        run_all_test_cases(20);
        
        -- Done with testing
        report "All tests completed";
        wait;

    end process stimulus;

end architecture testbench;