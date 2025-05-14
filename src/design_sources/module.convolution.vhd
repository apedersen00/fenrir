library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity Reverse_Convolution_Layer is 
generic(
    AerBusWidth                         : integer := 32;
    CoordinateWidth                     : integer := 6;
    TimeStampWidth                      : integer := 20;
    KernelSizeOneAxis                   : integer := 5;
    ImageWidth                          : integer := 64;
    ImageHeight                         : integer := 64;
    MemoryAddressWidth                  : integer := 11
);
port(
    -- Standard control signals
    clk                                 : in std_logic;
    reset_i                             : in std_logic;
    enable_i                            : in std_logic;

    -- AER Fifo Interface
    aer_fifo_bus_i                      : in  std_logic_vector(AerBusWidth - 1 downto 0);
    aer_empty_i                         : in  std_logic;
    aer_fifo_read_o                     : out std_logic
);
end Reverse_Convolution_Layer;

architecture behavioral of Reverse_Convolution_Layer is

    type main_states_e is (
        RESET,
        IDLE,
        PAUSE,
        READ_FIFO,
        PROCESS_EVENT
    );

    signal chip_state, chip_next_state, chip_last_state : main_states_e := IDLE;
    signal aer_time_stamp   : integer range 0 to 2**TimeStampWidth - 1;

    type coordinate_t is record
        x : integer range 0 to ImageWidth - 1;
        y : integer range 0 to ImageHeight - 1;
    end record;
    type pixels_to_update_t is array(0 to (KernelSizeOneAxis**2) - 1) of coordinate_t;
    signal pixels_to_update : pixels_to_update_t := (others => (x => 0, y => 0));
    signal addresses_to_process : integer range 0 to (KernelSizeOneAxis**2) - 1 := 0;

    function convert_coordinate_to_address(coord: coordinate_t) return std_logic_vector is
    begin
        return std_logic_vector(
            to_unsigned(coord.x + coord.y * ImageWidth, MemoryAddressWidth)
        );
    end function;

begin

    state_register_update : process(clk)
    begin 
    if rising_edge(clk) then
        if reset_i = '1' then
            chip_state <= RESET;
        else 
            chip_state <= chip_next_state;
        end if;
    end if;
    end process state_register_update;

    update_last_state : process(clk)
    begin
    if rising_edge(clk) then
        if reset_i = '1' then
            chip_last_state <= IDLE;
        elsif chip_state /= PAUSE then
            chip_last_state <= chip_state;
        end if;
    end if;
    end process update_last_state;

    state_machine_controller : process(all)
    begin
        chip_next_state <= chip_state;
        if enable_i = '0' then
            if chip_state /= RESET then
                chip_next_state <= PAUSE;
            end if;
        else
            case chip_state is
                when RESET =>
                    chip_next_state <= IDLE;               
                when IDLE => 
                    if aer_empty_i = '0' then
                        chip_next_state <= READ_FIFO;
                    end if;
                when READ_FIFO =>
                    chip_next_state <= PROCESS_EVENT;
                WHEN PROCESS_EVENT =>
                    -- temporary just move to next state idle
                    chip_next_state <= IDLE;
                when PAUSE =>
                    chip_next_state <= chip_last_state;
                when others =>
                    chip_next_state <= RESET;
            end case;
        end if;
    end process state_machine_controller;

    input_fifo_control : process(chip_state, enable_i, aer_empty_i, reset_i)
    begin
        aer_fifo_read_o <= '0';
        if reset_i = '1' or enable_i = '0' then
            aer_fifo_read_o <= '0';
        elsif chip_state = IDLE and aer_empty_i = '0' then
                aer_fifo_read_o <= '1';
        end if;
    end process input_fifo_control;

    set_read_event_register : process(clk)
        variable aer_x : integer range 0 to 2**CoordinateWidth - 1 := 0;
        variable aer_y : integer range 0 to 2**CoordinateWidth - 1 := 0;
        variable idx_counter : integer range 0 to (KernelSizeOneAxis**2) - 1 := 0;
        variable kernel_x, kernel_y : integer := 0;
    begin
    if rising_edge(clk) then

        if reset_i = '1' then
            aer_time_stamp <= 0;
            aer_x := 0;
            aer_y := 0;
            pixels_to_update <= (others => (x => 0, y => 0));
            addresses_to_process <= 0;
        elsif chip_state = READ_FIFO then
            -- Read aer bus
            aer_x := to_integer(unsigned(aer_fifo_bus_i(AerBusWidth - 1 downto AerBusWidth - CoordinateWidth)));
            aer_y := to_integer(unsigned(aer_fifo_bus_i(AerBusWidth - CoordinateWidth - 1 downto AerBusWidth - 2*CoordinateWidth)));
            aer_time_stamp <= to_integer(unsigned(aer_fifo_bus_i(AerBusWidth - 2*CoordinateWidth - 1 downto AerBusWidth - 2*CoordinateWidth - TimeStampWidth)));
            -- reset the previous values
            pixels_to_update <= (others => (x => 0, y => 0));
            addresses_to_process <= 0;

            for x in -(KernelSizeOneAxis-1)/2 to (KernelSizeOneAxis-1)/2 loop
                for y in -(KernelSizeOneAxis-1)/2 to (KernelSizeOneAxis-1)/2 loop

                    kernel_x := aer_x + x;
                    kernel_y := aer_y + y;

                    if (kernel_x >= 0 and kernel_x < ImageWidth and
                        kernel_y >= 0 and kernel_y < ImageHeight) 
                    then
                        pixels_to_update(idx_counter) <= 
                            (
                                x => kernel_x, 
                                y => kernel_y
                            );
                        idx_counter := idx_counter + 1;
                    end if;

                end loop;
            end loop;
            addresses_to_process <= idx_counter;
        end if;
    end if;
    end process set_read_event_register;
end behavioral;