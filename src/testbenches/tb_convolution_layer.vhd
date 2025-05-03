library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_convolution_layer is
end entity tb_convolution_layer;

architecture testbench of tb_convolution_layer is

    CONSTANT CLK_PERIOD          : time    := 10 ns;

    CONSTANT X_COORDINATE_WIDTH               : integer := 6;
    CONSTANT Y_COORDINATE_WIDTH               : integer := 6;
    CONSTANT TIME_STAMP_WIDTH                 : integer := 24;
    CONSTANT IMAGE_WIDTH                      : integer := 40;
    CONSTANT IMAGE_HEIGHT                     : integer := 30;
    CONSTANT KERNEL_WEIGHT_WIDTH              : integer := 4;
    CONSTANT AMOUNT_OF_FEATURE_MAPS           : integer := 4;
    CONSTANT KERNEL_SIZE_ONE_AXIS             : integer := 3;
    CONSTANT NEURON_MEMBRANE_POTENTIAL_WIDTH  : integer := 10;
    CONSTANT NEURON_RESET_POTENTIAL_WIDTH     : integer := 10;
    CONSTANT NEURON_THRESHOLD_POTENTIAL_WIDTH : integer := 10;
    
    signal clk                   : std_logic := '1';
    signal reset_o               : std_logic := '0';
    signal config_command_o      : std_logic_vector(1 downto 0) := (others => '0');
    signal config_data_io        : std_logic_vector(31 downto 0) := (others => '0');
    signal event_data_o          : std_logic_vector(XCoordinateWidth + YCoordinateWidth + TimeStampWidth - 1 downto 0) := (others => '0');
    signal event_fifo_empty_no   : std_logic := '1';
    signal event_fifo_read_i     : std_logic;

    procedure waitf(n : in integer) is
    begin
        wait for n * CLK_PERIOD;
    end procedure waitf;

    procedure test_control_signals(
        signal reset_o : inout std_logic;
        signal config_command_o : inout std_logic_vector(1 downto 0);
        signal config_data_io : inout std_logic_vector(31 downto 0)
    ) is
    begin
        -- Test all the control signals
        reset_o <= '1';
        -- NO_COMMAND
        config_command_o <= "00"; 
        config_data_io <= (others => '0');
        waitf(1);
        -- SET_KERNEL_WEIGHT
        config_command_o <= "01";
        config_data_io <= (others => '1');
        waitf(1);
        -- SET_RESET_POTENTIAL
        config_command_o <= "10";
        config_data_io <= (others => '0');
        waitf(1);
        -- SET_THRESHOLD_POTENTIAL
        config_command_o <= "11";
        config_data_io <= (others => '1');
        waitf(1);
        reset_o <= '0';
        waitf(1);
    end procedure test_control_signals;

begin

    conv_unit: entity work.convolution_layer
        generic map (
            ImageWidth                   => IMAGE_WIDTH, 
            ImageHeight                  => IMAGE_HEIGHT,
            InputXCoordinateWidth        => X_COORDINATE_WIDTH,
            InputYCoordinateWidth        => Y_COORDINATE_WIDTH,
            InputTimeStampWidth          => TIME_STAMP_WIDTH,
            KernelWeightWidth            => KERNEL_WEIGHT_WIDTH,
            AmountOfFeatureMaps          => AMOUNT_OF_FEATURE_MAPS,
            KernelSizeOneAxis            => KERNEL_SIZE_ONE_AXIS,
            NeuronMembranePotentialWidth => 10,
            NeuronResetPotentialWidth    => 10,
            NeuronThresholdPotentialWidth=> 10,
            NeuronTimestampWidth         => TIME_STAMP_WIDTH
        )
        port map (
            clk                     => clk,
            reset_i                 => reset_o,
            config_command_i        => config_command_o,
            config_data_io          => config_data_io,
            event_data_i           => event_data_o,
            event_fifo_empty_ni     => event_fifo_empty_no,
            event_fifo_read_o       => event_fifo_read_i
        );

    -- Clock generation process
    clk <= not clk after CLK_PERIOD / 2;

    test_control_main : process
    begin
        -- Test number 1: test control signals
        test_control_signals(
            reset_o => reset_o,
            config_command_o => config_command_o,
            config_data_io => config_data_io
        );

        -- Add more tests here as needed

        wait;
    end process test_control_main;

end architecture testbench;