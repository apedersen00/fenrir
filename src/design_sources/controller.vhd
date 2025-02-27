library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity controller is
    port (

        clk             : in std_logic;
        ready           : out std_logic; -- ready signal for data

        -- Data
        input_vector    : in std_logic_vector(15 downto 0); -- 16-bit input
        input_select    : out std_logic_vector(3 downto 0); -- 4-bit selector for input
        data_rdy        : in std_logic; -- data ready signal

        -- neuron and synapse 
        neuron_address  : out std_logic_vector(7 downto 0); -- 8-bit address for neuron
        neuron_input    : in std_logic_vector(31 downto 0); -- 32-bit input for neuron

        synapse_address : out std_logic_vector(7 downto 0); -- 8-bit address for synapse
        synapse_in      : in std_logic_vector(31 downto 0); -- 32-bit input for synapse

        -- signals for active neuron
        param_leak_str  : out std_logic_vector(6 downto 0); -- leakage stength parameter
        param_thr       : out std_logic_vector(11 downto 0); -- neuron firing threshold parameter
        state_core      : out std_logic_vector(11 downto 0); -- core neuron state from SRAM
        syn_weight      : out std_logic_vector(1 downto 0); -- synaptic weight
        syn_event       : out std_logic; -- synaptic event trigger
        
        state_core_next : in std_logic_vector(11 downto 0); -- next core neuron state to SRAM
        spike_out       : in std_logic -- neuron spike event output

    );
end controller;

architecture Behavioral of controller is
    signal synapse_counter          : integer := 0;
    signal neuron_counter           : integer := 0;
    signal input_row_counter        : integer := 0;

    type main_states is (
        IDLE,
        READ_REQUEST,
        WRITE_REQUEST,
        COMPUTING
    );
    type read_request_substates is (
        READ_NEURON,
        READ_SYNAPSE,
        READ_BOTH
    );

    signal MAIN_STATE               : main_states := IDLE;
    signal REQUEST_STATE            : read_request_substates := READ_BOTH;
    
begin

    process (clk)

        
    begin
        if rising_edge(clk) then
            case MAIN_STATE is
                when IDLE =>
                    -- waiting for data
                    if data_rdy = '1' then
                        -- data is ready, flip the ready signal and start getting neuron and synapse
                        ready <= '1';
                        MAIN_STATE <= READ_REQUEST;
                    end if;

                when READ_REQUEST => 
                    -- for reading neuron and synapses
                    -- when last synapse is reached we want to read neuron and synapse. Otherwise we want to read only synapse
                    case REQUEST_STATE is
                        when READ_NEURON =>
                            -- read neuron
                            -- convert neuron counter to 8-bit address
                            neuron_address <= std_logic_vector(to_unsigned(neuron_counter, 8));

                        when READ_SYNAPSE =>
                            -- read synapse
                            if (synapse_counter > 15 and synapse_counter < 32) then
                                -- multiply neuron counter by 3 and add 1
                                synapse_address <= std_logic_vector(to_unsigned(neuron_counter * 3 + 1, 8));
                            else
                                -- multiply neuron counter by 3 and add 2
                                synapse_address <= std_logic_vector(to_unsigned(neuron_counter * 3 + 2, 8));
                            end if;

                        when READ_BOTH =>
                            -- read both
                            -- multiply neuron counter by 3
                            synapse_address <= std_logic_vector(to_unsigned(neuron_counter * 3, 8));
                            -- same address for neuron
                            neuron_address <= std_logic_vector(to_unsigned(neuron_counter, 8));

                    end case;

                    MAIN_STATE <= COMPUTING;

                when WRITE_REQUEST =>
                    -- for updating neuron

                when COMPUTING => 
                    -- main loop
                    -- start updating the neuron signals from the 32 bit word from neuron bram (neuron_input)

                    param_leak_str <= neuron_input(30 downto 24);
                    param_thr <= neuron_input(23 downto 12);
                    state_core <= neuron_input(11 downto 0);

                    -- we must update the synapse weight. Since we are reading synapses as 32 bit words, we need to extract the weight from the word
                    -- each synapse_in word contains 16 weights. We use the counter signal to index the 2 bit weight
                    syn_weight <= synapse_in(2 * synapse_counter + 1 downto 2 * synapse_counter);

                    -- pass the input bit from the input vector. Indexed by the synapse counter


                when others => 
                    MAIN_STATE <= IDLE;

            end case;
        end if;
    end process;

end Behavioral;
