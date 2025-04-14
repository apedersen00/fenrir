library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity conv_unit is
    generic(
        BITS_PER_NEURON : integer := 10;
        FEATURE_MAPS : integer := 10;
        TIMESTAMP_WIDTH: integer := 12;
        NEURON_RESET_WIDTH : integer := 4;
        NEURON_THRESHOLD_WIDTH : integer := 4;
        KERNEL_BIT_WIDTH : integer := 4;
        LEAKAGE_PARAM_WIDTH : integer := 4;
    );
    port(
        enable : in std_logic;
        input_data : in std_logic_vector(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto 0);
        output_data : out std_logic_vector(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto 0);
        kernels : in std_logic_vector(FEATURE_MAPS * KERNEL_BIT_WIDTH - 1 downto 0);
        neuron_reset_value : in std_logic_vector(NEURON_RESET_WIDTH - 1 downto 0);
        neuron_threshold_value : in std_logic_vector(NEURON_THRESHOLD_WIDTH - 1 downto 0);
        leakage_param : in std_logic_vector(LEAKAGE_PARAM_WIDTH - 1 downto 0);
        timestamp_event: in std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0);
        spike_events: out std_logic_vector(FEATURE_MAPS - 1 downto 0);
        event_happened_flag: out std_logic
    );
end entity conv_unit;

architecture behavioral of conv_unit is

    signal timestamp : std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0);

    function get_vector_slice(
        data_vector: std_logic_vector;
        index: integer;
        width: integer
    ) return std_logic_vector is
        variable slice: std_logic_vector(width - 1 downto 0);
        begin

            slice := data_vector((index + 1) * width - 1 downto index * width);

            return slice;

        end function;

    function calculate_leakage(
        neuron_value: std_logic_vector(BITS_PER_NEURON - 1 downto 0);
        leakage_param: std_logic_vector(LEAKAGE_PARAM_WIDTH - 1 downto 0);
        timestamp: std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0);
        timestamp_new: std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0)
    ) return std_logic_vector(BITS_PER_NEURON - 1 DOWNto 0) is
    begin

    end function;

    procedure set_out_vector_slice(
        data_vector: inout std_logic_vector;
        index: integer;
        width: integer;
        value: std_logic_vector
    ) is
        begin

            data_vector((index + 1) * width - 1 downto index * width) := value;

        end procedure;

begin

process(enable)

    variable timestamp : std_logic_vector (TIMESTAMP_WIDTH - 1 downto 0);
    variable current_neuron_membrane_potential : std_logic_vector(BITS_PER_NEURON - 1 downto 0);
    variable kernel_value : std_logic_vector(KERNEL_BIT_WIDTH - 1 downto 0);
    variable event_happened : std_logic := '0';

begin
    if rising_edge(enable) then

        -- update timestamp
        timestamp := input_data(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto FEATURE_MAPS * BITS_PER_NEURON);

        for i in 0 to FEATURE_MAPS - 1 loop

            -- apply leakage
            current_neuron_membrane_potential := get_vector_slice(input_data, i, BITS_PER_NEURON);
            current_neuron_membrane_potential := calculate_leakage(
                current_neuron_membrane_potential,
                leakage_param,
                timestamp,
                timestamp_event
            );

            -- get kernel value
            kernel_value := get_vector_slice(kernels, i, KERNEL_BIT_WIDTH);

            -- add kernel value to neuron membrane potential
            current_neuron_membrane_potential := std_logic_vector(unsigned(current_neuron_membrane_potential) + unsigned(kernel_value));
            
            -- check if spike event happened
            if unsigned(current_neuron_membrane_potential) >= unsigned(neuron_threshold_value) then
                spike_events(i) := '1';
                -- reset neuron membrane potential
                current_neuron_membrane_potential := neuron_reset_value;
                event_happened := '1';
            else
                spike_events(i) := '0';
            end if;
            -- if spike event happened, set spike_events(i) to '1'
            -- else set spike_events(i) to '0'
            -- set the output_data to the result of the multiplication

            set_out_vector_slice(
                output_data,
                i,
                BITS_PER_NEURON,
                current_neuron_membrane_potential
            );

        end loop;

        -- set the timestamp in the output data and OR the spike events to set event_happened_flag if needed
        output_data(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto FEATURE_MAPS * BITS_PER_NEURON) <= timestamp_event;

        event_happened_flag <= event_happened;

    end if;
end process;

end architecture behavioral;