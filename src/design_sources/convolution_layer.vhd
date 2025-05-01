library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity convolution_layer is
    generic (
        ImageWidth                   : integer := 32; 
        ImageHeight                  : integer := 32;
        InputXCoordinateWidth        : integer := 8;
        InputYCoordinateWidth        : integer := 8;
        InputTimeStampWidth          : integer := 32;
        NeuronMembranePotentialWidth : integer := 10;
        NeuronResetPotentialWidth    : integer := 10;
        NeuronThresholdPotentialWidth: integer := 10;
        NeuronTimestampWidth         : integer := 32;
        KernelWeightWidth            : integer := 8
    );
    port(
        -- Control signals from outside the module
        clk                     : in  std_logic;
        reset_i                   : in  std_logic;
        config_command_i        : in  std_logic_vector(1 downto 0);
        config_data_i           : in  std_logic_vector(31 downto 0);

        -- Input FIFO Interface
        event_data_i            : in  std_logic_vector(InputXCoordinateWidth + InputYCoordinateWidth + InputTimeStampWidth - 1 downto 0);
        event_fifo_empty_ni     : in  std_logic;
        event_fifo_read_o       : out std_logic
    );
end entity convolution_layer;

architecture behavioral of convolution_layer is

    type config_commands_e is (
        NO_COMMAND,
        SET_KERNEL_WEIGHT,
        SET_RESET_POTENTIAL,
        SET_THRESHOLD_POTENTIAL
    );

    signal config_command : config_commands_e;

    type states_e is(
        IDLE,
        CONFIG,
        PROCESS_EVENT
    );

    signal state : states_e := IDLE;

begin

    command_decoder: process(clk, reset_i)
    begin

        IF reset_i = '1' THEN
        CASE config_command_i is
            WHEN "00" => config_command <= NO_COMMAND;
            WHEN "01" => config_command <= SET_KERNEL_WEIGHT;
            WHEN "10" => config_command <= SET_RESET_POTENTIAL;
            WHEN "11" => config_command <= SET_THRESHOLD_POTENTIAL;
            WHEN OTHERS => config_command <= NO_COMMAND;
        END CASE;
        END IF;

    end process command_decoder;


end architecture behavioral;