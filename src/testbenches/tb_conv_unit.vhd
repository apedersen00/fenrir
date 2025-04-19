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
    

    -- Signals for the BRAM
    signal addra : std_logic_vector(6 downto 0) := (others => '0');
    signal addrb : std_logic_vector(6 downto 0) := (others => '0');
    signal dina : std_logic_vector(11 downto 0) := (others => '0');
    signal dinb : std_logic_vector(11 downto 0) := (others => '0');
    signal douta : std_logic_vector(11 downto 0) := (others => '0');
    signal doutb : std_logic_vector(11 downto 0) := (others => '0');
    signal wea : std_logic := '0';
    signal web : std_logic := '0';
    
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


begin
    -- Clock generation
    clk <= not clk after CLK_PERIOD / 2;
    
    -- bram instantiation from BMG
    bram : entity work.mem_neuron_potentials
        port map(
            clka => clk,
            clkb => clk,
            addra => addra,
            addrb => addrb,
            dina => dina,
            dinb => output_data, 
            wea => wea,
            web => web,
            douta => input_data,
            doutb => doutb
        );

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
        addrb <= "WWWWWWW";
        wait for CLK_PERIOD * 10;
        dina <= x"112";
        wea <= '1';
        for i in 0 to 99 loop
            addra <= std_logic_vector(to_unsigned(i, addra'length));
            wait for CLK_PERIOD * 1;
        end loop;
        wea <= '0';
        timestamp_event <= "0010";

        enable <= '1';

        -- first event arrives at pixel (3,3) in an 10x10 image, output is 10x10x2
        -- start by setting address to the pixel by using formula: addr = col + row * width
        -- address a is for reading, address b is for writing

        for dy in -1 to 1 loop
            for dx in -1 to 1 loop

                addra <= address_calculation(3 + dy, 3 + dx, 10);
                wait for CLK_PERIOD * 1;

            end loop;
        end loop;

        wait;

    end process;

end architecture testbench;