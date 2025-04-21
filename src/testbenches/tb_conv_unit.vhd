library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_conv_unit is
end entity tb_conv_unit;

architecture testbench of tb_conv_unit is
    CONSTANT CLK_PERIOD : time := 10 ns;
    
    -- Constants matching the DUT
    CONSTANT BITS_PER_NEURON : integer := 4;
    CONSTANT FEATURE_MAPS : integer := 2;
    CONSTANT TIMESTAMP_WIDTH : integer := 4;
    
    -- Number of pixels in kernel window
    CONSTANT KERNEL_SIZE : integer := 9;  -- Simulating a 3x3 kernel window
    
    -- Signals for the DUT
    signal clk : std_logic := '1';
    signal enable : std_logic := '0';
    signal input_data : std_logic_vector(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto 0) := (others => '0');
    signal output_data : std_logic_vector(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto 0);
    signal kernels : std_logic_vector(FEATURE_MAPS * 4 - 1 downto 0) := (others => '0');
    signal neuron_reset_value : std_logic_vector(3 downto 0) := "0001"; 
    signal neuron_threshold_value : std_logic_vector(3 downto 0) := "1100";  
    signal leakage_param : std_logic_vector(3 downto 0) := "0000";  -- Small leakage
    signal timestamp_event : std_logic_vector(TIMESTAMP_WIDTH - 1 downto 0) := (others => '0');
    signal spike_events : std_logic_vector(FEATURE_MAPS - 1 downto 0);
    signal event_happened_flag : std_logic;
    
    
    -- Helper procedure to set neuron value in a vector
    procedure set_neuron_value(
        variable data_vec : inout std_logic_vector;
        index : integer;
        value : std_logic_vector
    ) is
    begin
        data_vec(BITS_PER_NEURON * (index + 1) - 1 downto BITS_PER_NEURON * index) := value;
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
    
    -- Helper function to extract neuron value from a vector
    function get_neuron_value(
        data : std_logic_vector;
        index : integer
    ) return std_logic_vector is
    begin
        return data(BITS_PER_NEURON * (index + 1) - 1 downto BITS_PER_NEURON * index);
    end function;
    
    function get_timestamp(
        data : std_logic_vector
    ) return std_logic_vector is
    begin
        return data(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto FEATURE_MAPS * BITS_PER_NEURON);
    end function;
    
    function address_calculation(
        row : integer;
        col : integer;
        width : integer
    ) return std_logic_vector is
        variable addr : std_logic_vector(6 downto 0);
        begin
        addr := std_logic_vector(to_unsigned(col + row * width, addr'length));
        return addr;
    end function;


    type input_array is array (0 to 8) of std_logic_vector(BITS_PER_NEURON * FEATURE_MAPS + TIMESTAMP_WIDTH - 1 downto 0);
    signal i_data : input_array := (
        x"a20",
        x"a23",
        x"a22",
        x"a21",
        x"a20",
        x"a23",
        x"a22",
        x"a21",
        x"a20"
    );
    type kernel_array is array (0 to 8) of std_logic_vector(7 downto 0);
    signal k_data : kernel_array := (
        x"00",
        x"01",
        x"02",
        x"03",
        x"04",
        x"05",
        x"06",
        x"07",
        x"08"
    );
begin
    -- Clock generation
    clk <= not clk after CLK_PERIOD / 2;
    
    -- bram instantiation from BMG
    

    -- Device under test instantiation
    dut : entity work.conv_unit
        generic map(
            BITS_PER_NEURON => BITS_PER_NEURON,
            FEATURE_MAPS => FEATURE_MAPS,
            TIMESTAMP_WIDTH => TIMESTAMP_WIDTH
        )
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
    
    stimulus : process
    begin
        neuron_reset_value <= "0000";
        neuron_threshold_value <= "1100";
        leakage_param <= "0000"; -- Small leakage
        timestamp_event <= "0011"; -- Initial timestamp
        enable <= '1';

        for i in 0 to 8 loop
            input_data <= i_data(i);
            kernels <= k_data(i);
            wait for CLK_PERIOD;
        end loop;

        wait;

    end process;

end architecture testbench;