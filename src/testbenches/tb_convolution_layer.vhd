library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_convolution_layer is
end entity tb_convolution_layer;

architecture testbench of tb_convolution_layer is

    CONSTANT CLK_PERIOD          : time    := 10 ns;

    CONSTANT XCoordinateWidth    : integer := 8;
    CONSTANT YCoordinateWidth    : integer := 8;
    CONSTANT TimeStampWidth      : integer := 32;

    signal clk                   : std_logic := '1';
    signal reset_o               : std_logic := '0';
    signal config_command_o      : std_logic_vector(1 downto 0) := (others => '0');
    signal config_data_io        : std_logic_vector(31 downto 0) := (others => '0');
    signal event_data_o          : std_logic_vector(XCoordinateWidth + YCoordinateWidth + TimeStampWidth - 1 downto 0) := (others => '0');
    signal event_fifo_empty_no   : std_logic := '1';
    signal event_fifo_read_i     : std_logic;

    signal test_number          : integer := 0;

    procedure waitf(n : in integer) is
    begin
        wait for n * CLK_PERIOD;
    end procedure waitf;

begin

    conv_unit: entity work.convolution_layer
        generic map (
            ImageWidth                   => 32, 
            ImageHeight                  => 32,
            InputXCoordinateWidth        => XCoordinateWidth,
            InputYCoordinateWidth        => YCoordinateWidth,
            InputTimeStampWidth          => TimeStampWidth,
            NeuronMembranePotentialWidth => 10,
            NeuronResetPotentialWidth    => 10,
            NeuronThresholdPotentialWidth=> 10,
            NeuronTimestampWidth         => 32,
            KernelWeightWidth            => 8,
            AmountOfFeatureMaps          => 4,
            KernelSizeOneAxis            => 3
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

    -- test all control signals
    command_encoder : process
    begin
    IF test_number = 0 THEN

        reset_o <= '1';
        waitf(2);
        reset_o <= '0';
        waitf(2);

    END IF;
    end process command_encoder;

end architecture testbench;