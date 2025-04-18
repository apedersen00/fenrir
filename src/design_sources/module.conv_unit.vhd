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
        TIME_SCALING_FACTOR : integer := 200
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
        neuron_value    : std_logic_vector;
        leakage_param   : std_logic_vector;
        timestamp_old   : std_logic_vector;
        timestamp_new   : std_logic_vector
    ) return std_logic_vector is
        variable mem      : unsigned(neuron_value'range)    := unsigned(neuron_value);
        variable leak     : unsigned(leakage_param'range)   := unsigned(leakage_param);
        variable t_old    : unsigned(timestamp_old'range)   := unsigned(timestamp_old);
        variable t_new    : unsigned(timestamp_new'range)   := unsigned(timestamp_new);
        variable delta_t  : unsigned(timestamp_old'range);
        variable scaled_leak : unsigned(mem'range);
        variable leak_product : unsigned(mem'range);
    begin
        -- Calculate time delta
        if t_new >= t_old then
            delta_t := t_new - t_old;
        else
            -- for wraparound
            delta_t := (others => '0');
        end if;

        -- Multiply leakage × delta_t
        leak_product := resize(leak * delta_t, mem'length);

        -- Apply scaling divisor
        scaled_leak := leak_product / TIME_SCALING_FACTOR;

        -- Apply leakage with saturation
        if mem > scaled_leak then
            mem := mem - scaled_leak;
        else
            mem := (others => '0');
        end if;

        return std_logic_vector(mem);
    end function;

    procedure set_out_vector_slice(
        variable data_vector : inout std_logic_vector;
        index  : integer;
        width  : integer;
        value  : std_logic_vector
    ) is
    begin
        data_vector((index + 1) * width - 1 downto index * width) := value;
    end procedure;

begin
    process(all)  -- Make the process sensitive to all signals
        variable timestamp_var : std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0);
        variable current_neuron_membrane_potential : std_logic_vector(BITS_PER_NEURON - 1 downto 0);
        variable kernel_value : std_logic_vector(KERNEL_BIT_WIDTH - 1 downto 0);
        variable event_happened : std_logic;
        variable output_data_var : std_logic_vector(output_data'range);
        variable spike_events_var : std_logic_vector(FEATURE_MAPS - 1 downto 0);
    begin
        if enable = '1' then
            -- Initialize variables
            event_happened := '0';
            output_data_var := (others => '0');
            spike_events_var := (others => '0');
            
            -- Get timestamp from input data
            timestamp_var := input_data(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto FEATURE_MAPS * BITS_PER_NEURON);
            
            -- Copy timestamp to output data right away
            output_data_var(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto FEATURE_MAPS * BITS_PER_NEURON) := timestamp_event;
            
            -- Process each neuron
            for i in 0 to FEATURE_MAPS - 1 loop
                -- Get current membrane potential
                current_neuron_membrane_potential := get_vector_slice(input_data, i, BITS_PER_NEURON);
                
                report "Neuron " & integer'image(i) &
                ": Mem(before) = " & to_hstring(current_neuron_membrane_potential)
                severity note;
                
                -- Apply leakage
                current_neuron_membrane_potential := calculate_leakage(
                    current_neuron_membrane_potential,
                    leakage_param,
                    timestamp_var,
                    timestamp_event
                );
                report "After leakage: " & to_hstring(current_neuron_membrane_potential) severity note;
                
                -- Get kernel value
                kernel_value := get_vector_slice(kernels, i, KERNEL_BIT_WIDTH);
                report "Kernel value: " & to_hstring(kernel_value) severity note;
                report "Kernel(" & integer'image(i) & ") = " & to_hstring(kernel_value) severity note;
                
                -- Add kernel value to neuron membrane potential
                current_neuron_membrane_potential := std_logic_vector(unsigned(current_neuron_membrane_potential) + unsigned(kernel_value));
                report "After adding kernel value: " & to_hstring(current_neuron_membrane_potential) severity note;
                
                -- Check if spike event happened
                if unsigned(current_neuron_membrane_potential) >= unsigned(neuron_threshold_value) then
                    spike_events_var(i) := '1';
                    -- Reset neuron membrane potential
                    current_neuron_membrane_potential := std_logic_vector(resize(unsigned(neuron_reset_value), BITS_PER_NEURON));
                    event_happened := '1';
                else
                    spike_events_var(i) := '0';
                end if;
                
                -- Update output data for this neuron
                set_out_vector_slice(
                    output_data_var,
                    i,
                    BITS_PER_NEURON,
                    current_neuron_membrane_potential
                );
            end loop;
            
            -- Update outputs all at once
            output_data <= output_data_var;
            spike_events <= spike_events_var;
            event_happened_flag <= event_happened;
        end if;
    end process;
end architecture behavioral;