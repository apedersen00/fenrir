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
        clk                  : in  std_logic;
        reset                : in  std_logic;

        -- Input FIFO Interface
        event_data_i         : in  std_logic_vector(InputXCoordinateWidth + InputYCoordinateWidth + InputTimeStampWidth - 1 downto 0);
        event_fifo_empty_ni  : in  std_logic;
        event_fifo_read_o    : out std_logic
    );
end entity convolution_layer;

architecture behavioral of convolution_layer is

    type states_e is(
        IDLE,
        CONFIG,
        PROCESS_EVENT
    );

    signal state : states_e := IDLE;

begin
end architecture behavioral;