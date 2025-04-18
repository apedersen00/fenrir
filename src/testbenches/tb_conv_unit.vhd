library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_conv_unit is
end entity tb_conv_unit;

architecture testbench of tb_conv_unit is
    CONSTANT CLK_PERIOD : time := 10 ns;
    
    -- Constants matching the DUT
    CONSTANT BITS_PER_NEURON : integer := 10;
    CONSTANT FEATURE_MAPS : integer := 10;
    CONSTANT TIMESTAMP_WIDTH : integer := 12;
    
    signal clk : std_logic := '1';
    signal enable : std_logic := '0';
    signal input_data : std_logic_vector(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto 0) := (others => '0');
    signal output_data : std_logic_vector(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto 0);
    signal kernels : std_logic_vector(FEATURE_MAPS * 4 - 1 downto 0) := (others => '0');
    signal neuron_reset_value : std_logic_vector(3 downto 0) := "0001";  -- Reset to 1
    signal neuron_threshold_value : std_logic_vector(3 downto 0) := "1111";  -- Threshold 15
    signal leakage_param : std_logic_vector(3 downto 0) := "0001";  -- Small leakage
    signal timestamp_event : std_logic_vector(11 downto 0) := x"008";
    signal spike_events : std_logic_vector(FEATURE_MAPS - 1 downto 0);
    signal event_happened_flag : std_logic;
    
    -- Helper procedure to set neuron value in input_data
    procedure set_neuron_value(
        signal data : inout std_logic_vector;
        index : integer;
        value : std_logic_vector
    ) is
    begin
        data(BITS_PER_NEURON * (index + 1) - 1 downto BITS_PER_NEURON * index) <= value;
    end procedure;
    
    -- Helper procedure to set kernel value
    procedure set_kernel_value(
        signal k_data : inout std_logic_vector;
        index : integer;
        value : std_logic_vector(3 downto 0)
    ) is
    begin
        k_data(4 * (index + 1) - 1 downto 4 * index) <= value;
    end procedure;
    
    -- Helper function to extract neuron value from output_data
    function get_neuron_value(
        data : std_logic_vector;
        index : integer
    ) return std_logic_vector is
    begin
        return data(BITS_PER_NEURON * (index + 1) - 1 downto BITS_PER_NEURON * index);
    end function;
    
begin
    -- Clock generation
    clk <= not clk after CLK_PERIOD / 2;
    
    -- Device under test instantiation
    dut : entity work.conv_unit
        port map(
            enable => enable,
            input_data => input_data,
            output_data => output_data,
            kernels => kernels,
            neuron_reset_value => neuron_reset_value,
            neuron_threshold_value => neuron_threshold_value,
            leakage_param => leakage_param,
            timestamp_event => timestamp_event,
            spike_events => spike_events,
            event_happened_flag => event_happened_flag
        );
    
    -- Stimulus process
    stimulus : process
    begin
        -- Initialize all inputs
        input_data <= (others => '0');
        kernels <= (others => '0');
        wait for CLK_PERIOD * 5;
        
        -- Set timestamp in input_data
        input_data(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto BITS_PER_NEURON * FEATURE_MAPS) <= x"001";
        
        -- Set initial neuron values (10 neurons with values 1 to 10)
        for i in 0 to FEATURE_MAPS - 1 loop
            set_neuron_value(input_data, i, std_logic_vector(to_unsigned(i + 1, BITS_PER_NEURON)));
        end loop;
        
        -- Set kernel values (all 4-bit values set to 2)
        for i in 0 to FEATURE_MAPS - 1 loop
            set_kernel_value(kernels, i, "0010");  -- Value 2
        end loop;
        
        -- Apply stimulus
        wait for CLK_PERIOD;
        enable <= '1';  -- Enable the module
        wait for CLK_PERIOD * 2;  -- Wait a bit to see the results
        
        -- Print out the results
        for i in 0 to FEATURE_MAPS - 1 loop
            report "Output Neuron " & integer'image(i) & " = " & 
                   integer'image(to_integer(unsigned(get_neuron_value(output_data, i))));
        end loop;
        
        report "Spike events = " & to_string(spike_events);
        report "Event happened flag = " & std_logic'image(event_happened_flag);
        
        -- Second test: Try to generate spikes
        wait for CLK_PERIOD * 5;
        enable <= '0';  -- Disable temporarily
        
        -- Set timestamp in input_data
        input_data(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto BITS_PER_NEURON * FEATURE_MAPS) <= x"002";
        
        -- Set high neuron values to trigger spikes
        for i in 0 to FEATURE_MAPS - 1 loop
            set_neuron_value(input_data, i, std_logic_vector(to_unsigned(13 + i, BITS_PER_NEURON)));  -- Close to threshold
        end loop;
        
        -- Set kernel values high enough to push some neurons over threshold
        for i in 0 to FEATURE_MAPS - 1 loop
            set_kernel_value(kernels, i, "0011");  -- Value 3
        end loop;
        
        -- Apply stimulus
        timestamp_event <= x"010";  -- Different timestamp
        wait for CLK_PERIOD;
        enable <= '1';  -- Enable the module
        wait for CLK_PERIOD * 2;  -- Wait a bit to see the results
        
        -- Print out the results
        for i in 0 to FEATURE_MAPS - 1 loop
            report "Output Neuron " & integer'image(i) & " = " & 
                   integer'image(to_integer(unsigned(get_neuron_value(output_data, i))));
        end loop;
        
        report "Spike events = " & to_string(spike_events);
        report "Event happened flag = " & std_logic'image(event_happened_flag);
        
        -- End simulation
        wait for CLK_PERIOD * 10;
        report "Simulation finished" severity note;
        wait;
    end process;
    
    -- Monitoring process
    monitor: process
    begin
        wait for CLK_PERIOD;
        while true loop
            if enable = '1' then
                report "====== At time " & time'image(now) & " ======";
                report "Enable = " & std_logic'image(enable);
                report "Event happened = " & std_logic'image(event_happened_flag);
                report "Timestamp = " & to_string(timestamp_event);
            end if;
            wait for CLK_PERIOD;
        end loop;
    end process;
end architecture testbench;