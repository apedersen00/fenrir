library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity Reverse_Convolution_Layer is 
generic(
    AerBusWidth                        : integer := 32
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
    
    signal aer_fifo_read_req : std_logic := '0';
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

end behavioral;