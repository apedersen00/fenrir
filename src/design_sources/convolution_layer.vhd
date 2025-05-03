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
        clk                     : in    std_logic;
        reset_i                 : in    std_logic;
        config_command_i        : in    std_logic_vector(1 downto 0);
        config_data_i           : in    std_logic_vector(31 downto 0);

        -- Input FIFO Interface
        event_data_i            : in    std_logic_vector(InputXCoordinateWidth + InputYCoordinateWidth + InputTimeStampWidth - 1 downto 0);
        event_fifo_empty_ni     : in    std_logic;
        event_fifo_read_o       : out   std_logic
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
        variable kernel_x : integer := 0;
        variable kernel_y : integer := 0;
        variable kernel_z : integer := 0;
        variable kernel_weight : std_logic_vector(KernelWeightWidth - 1 downto 0) := (others => '0');
    begin
        IF reset_i = '1' AND rising_edge(clk) then
        CASE config_command is
            WHEN NO_COMMAND => 
            WHEN SET_KERNEL_WEIGHT =>
                -- First 3 * 4 bits are used for the kernel position (x,y,z), rest is the kernel weight
                kernel_x := to_integer(unsigned(config_data_i(31 downto 28)));
                kernel_y := to_integer(unsigned(config_data_i(27 downto 24)));
                kernel_z := to_integer(unsigned(config_data_i(23 downto 20)));
                kernel_weight := std_logic_vector(resize(unsigned(config_data_i(19 downto 0)), KernelWeightWidth));
                kernel_weights(kernel_x, kernel_y, kernel_z) <= kernel_weight;

            WHEN SET_RESET_POTENTIAL =>
            WHEN SET_THRESHOLD_POTENTIAL => 
        END CASE;
        END if;
    end process config_command_exe;

    command_decoder: process(clk, reset_i)
    begin

        IF reset_i = '1' THEN
        -- Set the state to config
        state <= CONFIG;
        -- this should be on the clock edge
        IF rising_edge(clk) THEN
            CASE config_command_i is
            WHEN "00" => config_command <= NO_COMMAND;
            WHEN "01" => config_command <= SET_KERNEL_WEIGHT;
            WHEN "10" => config_command <= SET_RESET_POTENTIAL;
            WHEN "11" => config_command <= SET_THRESHOLD_POTENTIAL;
            WHEN OTHERS => config_command <= NO_COMMAND;
            END CASE;
        END IF;
        END IF;

    end process command_decoder;

    event_listener : process(clk, reset_i, event_fifo_empty_ni)
    begin
    IF reset_i = '0' AND rising_edge(clk) then

        IF event_fifo_empty_ni = '0' THEN event_fifo_read_o <= '1';
        ELSE event_fifo_read_o <= '0';
        END IF;

    END IF;
    end process event_listener; 

end architecture behavioral;