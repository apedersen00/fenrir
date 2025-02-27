library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_controller is
-- Testbench doesn't have ports
end tb_controller;

architecture behavior of tb_controller is
    -- Component declaration for the Unit Under Test (UUT)
    component controller
        port (
            clk             : in std_logic;
            ready           : out std_logic;
            
            input_vector    : in std_logic_vector(15 downto 0);
            input_select    : out std_logic_vector(3 downto 0);
            data_rdy        : in std_logic;
            
            neuron_address  : out std_logic_vector(7 downto 0);
            neuron_input    : in std_logic_vector(31 downto 0);
            neuron_we       : out std_logic;
            neuron_update   : out std_logic_vector(31 downto 0);
            
            synapse_address : out std_logic_vector(7 downto 0);
            synapse_in      : in std_logic_vector(31 downto 0);
            
            param_leak_str  : out std_logic_vector(6 downto 0);
            param_thr       : out std_logic_vector(11 downto 0);
            state_core      : out std_logic_vector(11 downto 0);
            syn_weight      : out std_logic_vector(1 downto 0);
            syn_event       : out std_logic;
            
            state_core_next : in std_logic_vector(11 downto 0);
            spike_out       : in std_logic
        );
    end component;
    
    -- Clock period definition
    constant clk_period : time := 10 ns;
    
    -- Inputs
    signal clk            : std_logic := '0';
    signal input_vector   : std_logic_vector(15 downto 0) := (others => '0');
    signal data_rdy       : std_logic := '0';
    signal neuron_input   : std_logic_vector(31 downto 0) := (others => '0');
    signal synapse_in     : std_logic_vector(31 downto 0) := (others => '0');
    signal state_core_next: std_logic_vector(11 downto 0) := (others => '0');
    signal spike_out      : std_logic := '0';
    
    -- Outputs
    signal ready          : std_logic;
    signal input_select   : std_logic_vector(3 downto 0);
    signal neuron_address : std_logic_vector(7 downto 0);
    signal neuron_we      : std_logic;
    signal neuron_update  : std_logic_vector(31 downto 0);
    signal synapse_address: std_logic_vector(7 downto 0);
    signal param_leak_str : std_logic_vector(6 downto 0);
    signal param_thr      : std_logic_vector(11 downto 0);
    signal state_core     : std_logic_vector(11 downto 0);
    signal syn_weight     : std_logic_vector(1 downto 0);
    signal syn_event      : std_logic;
    
    -- Memory arrays to simulate BRAM behavior
    type neuron_mem_type is array (0 to 255) of std_logic_vector(31 downto 0);
    type synapse_mem_type is array (0 to 255) of std_logic_vector(31 downto 0);
    
    signal neuron_mem : neuron_mem_type := (others => (others => '0'));
    signal synapse_mem : synapse_mem_type := (others => (others => '0'));
    
begin
    -- Instantiate the Unit Under Test (UUT)
    uut: controller port map (
        clk => clk,
        ready => ready,
        
        input_vector => input_vector,
        input_select => input_select,
        data_rdy => data_rdy,
        
        neuron_address => neuron_address,
        neuron_input => neuron_input,
        neuron_we => neuron_we,
        neuron_update => neuron_update,
        
        synapse_address => synapse_address,
        synapse_in => synapse_in,
        
        param_leak_str => param_leak_str,
        param_thr => param_thr,
        state_core => state_core,
        syn_weight => syn_weight,
        syn_event => syn_event,
        
        state_core_next => state_core_next,
        spike_out => spike_out
    );
    
    -- Clock process definition
    clk_process: process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;
    
    -- Simulate memory access for neuron and synapse
    memory_process: process(clk)
    begin
        if rising_edge(clk) then
            -- Read from neuron memory
            neuron_input <= neuron_mem(to_integer(unsigned(neuron_address)));
            
            -- Read from synapse memory
            synapse_in <= synapse_mem(to_integer(unsigned(synapse_address)));
            
            -- Write to neuron memory when write enable is active
            if neuron_we = '1' then
                neuron_mem(to_integer(unsigned(neuron_address))) <= neuron_update;
            end if;
        end if;
    end process;
    
    -- Stimulus process
    stimulus_proc: process
    begin
        -- Initialize memories with test data
        -- Neuron memory: Each entry has bits for leak_str(30-24), threshold(23-12), and initial state_core(11-0)
        neuron_mem(0) <= "0000001" & "000011110000" & "000000000001"; -- Neuron 0
        neuron_mem(1) <= "0000010" & "000011110000" & "000000000010"; -- Neuron 1
        neuron_mem(2) <= "0000011" & "000011110000" & "000000000011"; -- Neuron 2
        
        -- Synapse memory: Each 32-bit word contains 16 2-bit weights
        -- We'll set up some test weights: "00" = no connection, "01" = weak, "10" = medium, "11" = strong
        synapse_mem(0) <= "01" & "10" & "11" & "01" & "00" & "01" & "10" & "11" & "01" & "10" & "00" & "01" & "10" & "11" & "01" & "10"; -- First 16 weights for neuron 0
        synapse_mem(1) <= "11" & "01" & "10" & "00" & "01" & "10" & "11" & "01" & "10" & "11" & "01" & "10" & "00" & "01" & "10" & "11"; -- Next 16 weights for neuron 0
        synapse_mem(2) <= "10" & "11" & "01" & "10" & "00" & "01" & "10" & "11" & "01" & "00" & "11" & "01" & "10" & "00" & "01" & "10"; -- Last 16 weights for neuron 0
        synapse_mem(3) <= "01" & "00" & "11" & "01" & "10" & "00" & "01" & "10" & "11" & "01" & "00" & "11" & "01" & "10" & "00" & "01"; -- First 16 weights for neuron 1
        
        -- Hold reset for a few clock cycles
        wait for clk_period * 5;
        
        -- Test case 1: Process one neuron with a set of inputs
        -- Set input vector (16 bits representing input spikes)
        input_vector <= "1010010110100101"; -- Some test pattern
        
        -- Simulate state_core_next values
        -- These would be calculated by some external logic in the actual system
        -- but for our testbench, we're just providing sample values
        wait for clk_period;
        
        -- Set data ready signal
        data_rdy <= '1';
        wait for clk_period;
        data_rdy <= '0';
        
        -- The controller should now start processing
        -- Wait for the ready signal to become active again
        wait until ready = '1';
        wait for clk_period * 2;
        
        -- Test case 2: Process with a different input pattern and simulate a spike
        input_vector <= "1111000011110000";
        
        -- This time we'll simulate a spike during processing
        wait for clk_period;
        data_rdy <= '1';
        wait for clk_period;
        data_rdy <= '0';
        
        -- Wait for some processing to occur
        wait for clk_period * 10;
        
        -- Simulate a spike
        spike_out <= '1';
        wait for clk_period;
        spike_out <= '0';
        
        -- Wait for the controller to finish processing
        wait until ready = '1';
        wait for clk_period * 2;
        
        -- End the simulation
        wait for clk_period * 10;
        assert false report "Simulation ended" severity failure;
        
        wait;
    end process;
    
    -- Process to update state_core_next based on current simulation state
    state_update_proc: process(clk)
    begin
        if rising_edge(clk) then
            -- In a real implementation, this would be calculated based on current state,
            -- leak parameter, input, etc. For this testbench, we'll use a simple model:
            -- If syn_event is active, we'll increase the state by the weight value
            -- and apply some leakage
            
            if syn_event = '1' then
                -- Convert syn_weight to an integer value
                case syn_weight is
                    when "00" => 
                        state_core_next <= state_core; -- No change
                    when "01" => 
                        state_core_next <= std_logic_vector(unsigned(state_core) + 1); -- Small increase
                    when "10" => 
                        state_core_next <= std_logic_vector(unsigned(state_core) + 2); -- Medium increase
                    when "11" => 
                        state_core_next <= std_logic_vector(unsigned(state_core) + 4); -- Large increase
                    when others =>
                        state_core_next <= state_core;
                end case;
            else
                -- Apply some leakage (decrease the state slightly)
                if unsigned(state_core) > 0 then
                    state_core_next <= std_logic_vector(unsigned(state_core) - 1);
                else
                    state_core_next <= state_core;
                end if;
            end if;
            
            -- Check if we need to trigger a spike (when state exceeds threshold)
            if param_thr /= "000000000000" and unsigned(state_core) >= unsigned(param_thr) then
                -- This will be picked up in the next clock cycle by the spike_out signal
                -- We'll handle this in the main stimulus process
            end if;
        end if;
    end process;
    
    -- Monitor process to observe important signals
    monitor_proc: process
    begin
        wait for clk_period;
        if ready = '0' then
            report "Controller is processing";
        end if;
        
        if neuron_we = '1' then
            report "Neuron memory write: Address=" & integer'image(to_integer(unsigned(neuron_address))) & 
                   ", Value=" & integer'image(to_integer(unsigned(neuron_update)));
        end if;
        
        if syn_event = '1' then
            report "Synapse event detected with weight: " & integer'image(to_integer(unsigned(syn_weight)));
        end if;
        
        if spike_out = '1' then
            report "Neuron spike detected!";
        end if;
    end process;
    
end behavior;