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
        KernelWeightWidth            : integer := 8;
        AmountOfFeatureMaps          : integer := 4;
        KernelSizeOneAxis            : integer := 3
    );
    port(
        -- Control signals from outside the module
        clk                     : in  std_logic;
        reset_i                   : in  std_logic;
        config_command_i        : in  std_logic_vector(1 downto 0);
        config_data_io           : inout  std_logic_vector(31 downto 0);

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

    -- 3D array for the kernel weights: access (x,y,z) for one kernel weight
    type kernel_weights_t is array (
        0 to KernelSizeOneAxis - 1,
        0 to KernelsizeOneAxis - 1,
        0 to AmountOfFeatureMaps - 1) of std_logic_vector(KernelWeightWidth - 1 downto 0);
    -- Initialize the kernel weights to zero
    signal kernel_weights : kernel_weights_t := (others => (others => (others => (others => '0'))));

begin

    config_command_exe : process(clk, reset_i, config_command)
    begin

        IF reset_i = '1' AND rising_edge(clk) then
        CASE config_command is
            WHEN NO_COMMAND => 
            -- Do nothing, wait for a command
            WHEN SET_KERNEL_WEIGHT =>
            WHEN SET_RESET_POTENTIAL =>
            WHEN SET_THRESHOLD_POTENTIAL => 
        END CASE;
        END if;


    end process config_command_exe;

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