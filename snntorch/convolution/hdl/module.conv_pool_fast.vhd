library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.conv_pool_pkg.all;

entity conv_pool_fast is
    generic(
        CHANNELS_OUT : integer := 12;
        BITS_PER_NEURON : integer := 6;
        BITS_PER_WEIGHT : integer := 6;
        IMG_WIDTH : integer := 32;
        IMG_HEIGHT : integer := 32;
        BITS_PER_COORD : integer := 8;
        KERNEL_SIZE : integer := 3
    );
    port(
        -- Standard ports needed for digital components
        clk : in std_logic;
        rst_i : in std_logic;
        enable_i : in std_logic;

        -- Control signals
        timestep_i : in std_logic;

        -- Event signals
        event_fifo_empty_i : in std_logic;
        event_fifo_bus_i : in std_logic_vector(2 * BITS_PER_COORD downto 0);
        event_fifo_read_o : out std_logic;

        -- pragma translate_off
        debug_main_state : out main_state_et;
        debug_next_state : out main_state_et;
        debug_last_state : out main_state_et;
        
        debug_timestep_pending : out std_logic;
        debug_current_event : out event_tensor_t;
        debug_event_valid : out std_logic;
        debug_read_cycle : out integer;
        -- lets spoof the bram signals 
        debug_mem_neuron_wea, debug_mem_neuron_web : out std_logic;
        debug_mem_neuron_ena, debug_mem_neuron_enb : out std_logic;
        debug_mem_neuron_addra, debug_mem_neuron_addrb : out std_logic_vector(9 downto 0);  -- FIXED: 10-bit to match internal
        debug_mem_neuron_dia, debug_mem_neuron_dib : out std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0);  -- FIXED: Use CHANNELS_OUT
        debug_mem_neuron_doa, debug_mem_neuron_dob : out std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0);  -- FIXED: Use CHANNELS_OUT

        debug_total_coords_to_update : out integer;
        debug_main_state_vec, debug_next_state_vec, debug_last_state_vec : out std_logic_vector(2 downto 0)
        -- pragma translate_on
    );
end entity conv_pool_fast;

architecture rtl of conv_pool_fast is

    signal main_state, main_next_state, main_last_state : main_state_et := IDLE;
    signal timestep_pending : std_logic := '0';
    
    signal current_event : event_tensor_t := (
        x_coord => 0,
        y_coord => 0,
        channel => 0
    );
    signal event_valid : std_logic := '0';
    signal read_cycle_counter : integer range 1 to 2 := 1;

    type coords_to_update_t is array (0 to (KERNEL_SIZE ** 2) - 1) of vector2_t;
    signal coords_to_update : coords_to_update_t := (others => (x => 0, y => 0));
    signal total_coords_to_update : integer range 0 to (KERNEL_SIZE ** 2) := 0;
    type kernel_weights_t is array (0 to (KERNEL_SIZE ** 2) -1) of std_logic_vector((CHANNELS_OUT * BITS_PER_WEIGHT) - 1 downto 0);
    signal kernel_weights : kernel_weights_t := (others => (others => '0'));
    
    -- Neuron Memory
    signal mem_neuron_wea_o, mem_neuron_web_o, mem_neuron_ena_o, mem_neuron_enb_o : std_logic := '0';
    signal mem_neuron_addra_o, mem_neuron_addrb_o : std_logic_vector(9 downto 0) := (others => '0');
    signal mem_neuron_dia_o, mem_neuron_dib_o : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');
    signal mem_neuron_doa_i, mem_neuron_dob_i : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');

begin

    -- UPDATED: Read request logic for 2-cycle protocol
    event_fifo_read_o <= '1' when (main_state = READ_REQUEST and read_cycle_counter = 1) else '0';
    
    state_register_update : process (clk, rst_i)
    begin
        if rising_edge(clk) then
            if rst_i = '1' then
                main_state <= RESET;
            else
                main_state <= main_next_state;
            end if;
        end if;
    end process state_register_update;

    update_last_state : process (clk, rst_i)
    begin
        if rising_edge(clk) then
            if rst_i = '1' then
                main_last_state <= IDLE;
            elsif main_state /= PAUSE then
                main_last_state <= main_state;
            end if;
        end if;
    end process update_last_state;

    -- UPDATED: State machine with 2-cycle READ_REQUEST handling
    state_machine_control : process (all)
    begin
        main_next_state <= main_state; -- Default: stay in current state

        if enable_i = '0' then
            main_next_state <= PAUSE;
        else
            case main_state is
            when IDLE =>
                -- Priority 1: Check timestep
                if timestep_pending = '1' then
                    main_next_state <= POOL;
                -- Priority 2: Check FIFO 
                elsif event_fifo_empty_i = '0' then
                    main_next_state <= READ_REQUEST;
                end if;
                
            when READ_REQUEST =>
                -- Stay in READ_REQUEST for 2 cycles, then go to EVENT_CONV
                if read_cycle_counter = 2 then
                    main_next_state <= EVENT_CONV;
                end if;
                
            when EVENT_CONV => 
                -- Process event, then back to IDLE
                main_next_state <= IDLE;
                
            when PAUSE => 
                main_next_state <= main_last_state;
                
            when POOL =>
                -- Pool processing, then back to IDLE
                main_next_state <= IDLE;
                
            when CONFIG =>
                main_next_state <= IDLE;
                
            when RESET => 
                main_next_state <= IDLE;
            end case;
        end if;
    end process state_machine_control;

    timestep_buffer : process (clk, rst_i)
    begin
        if rising_edge(clk) then
            if rst_i = '1' then 
                timestep_pending <= '0';
            else
                if timestep_i = '1' then
                    timestep_pending <= '1';
                elsif main_state = POOL then
                    timestep_pending <= '0';
                end if;
            end if;
        end if;
    end process timestep_buffer;

    -- NEW: Read cycle counter management
    read_cycle_management : process (clk, rst_i)
    begin
        if rising_edge(clk) then
            if rst_i = '1' then
                read_cycle_counter <= 1;
            else
                if main_state = READ_REQUEST then
                    if read_cycle_counter < 2 then
                        read_cycle_counter <= read_cycle_counter + 1;
                    end if;
                else
                    read_cycle_counter <= 1; -- Reset counter when leaving READ_REQUEST
                end if;
            end if;
        end if;
    end process read_cycle_management;

    -- UPDATED: Event capture process with proper timing
    event_capture : process (clk, rst_i)

        variable temp_event : event_tensor_t;
        variable coords_index : integer range 0 to (KERNEL_SIZE ** 2) := 0;
        variable kernel_x, kernel_y : integer := 0;
    begin
        if rising_edge(clk) then
            if rst_i = '1' then 
                current_event <= (x_coord => 0, y_coord => 0, channel => 0);
                event_valid <= '0';
                coords_to_update <= (others => (x => 0, y => 0));
            else
                -- Capture event on cycle 2 of READ_REQUEST (when FIFO data is valid)
                if main_state = READ_REQUEST and read_cycle_counter = 2 then
                    temp_event := bus_to_event_tensor(
                        event_fifo_bus_i,
                        BITS_PER_COORD,
                        1
                    );
                    event_valid <= '1';
                    current_event <= temp_event;
                    -- reset coords to update
                    coords_index := 0;
                    coords_to_update <= (others => (x => 0, y => 0));
                    total_coords_to_update <= 0;

                    for x in -(KERNEL_SIZE - 1) / 2 to (KERNEL_SIZE - 1) / 2 LOOP
                    for y in -(KERNEL_SIZE - 1) / 2 to (KERNEL_SIZE - 1) / 2 LOOP

                        kernel_x := temp_event.x_coord + x;
                        kernel_y := temp_event.y_coord + y;
                        if  kernel_x >= 0 and kernel_x < IMG_WIDTH and
                            kernel_y >= 0 and kernel_y < IMG_HEIGHT and 
                            coords_index < (KERNEL_SIZE ** 2) then

                            coords_to_update(coords_index) <= (x => kernel_x, y => kernel_y);
                            coords_index := coords_index + 1;

                        end if;
                    end loop; end loop;
                    total_coords_to_update <= coords_index; -- Update total coords to update
                end if;
                
                -- Clear when returning to IDLE
                if main_state = EVENT_CONV and main_next_state = IDLE then
                    event_valid <= '0';
                end if;
            end if;
        end if;
    end process event_capture;

    convolution_control : process (clk, rst_i)
    begin
    if rising_edge(clk) then
    if rst_i = '1' then

    else
        case main_state is
        WHEN EVENT_CONV =>

        WHEN OTHERS =>
        end case;    
    end if;
    end if;
    end process convolution_control;

    -- ram instances
    
    MEM_NEURON : entity work.TRUE_DUAL_PORT_READ_FIRST
    generic map(
        RAM_DEPTH => IMG_WIDTH * IMG_HEIGHT,
        DATA_WIDTH => CHANNELS_OUT * BITS_PER_NEURON,
        ADDR_WIDTH => integer(ceil(log2(real(IMG_WIDTH * IMG_HEIGHT))))
    )
    port map(
        clka => clk,
        clkb => clk,
        ena => mem_neuron_ena_o, -- Enable for neuron memory
        enb => mem_neuron_enb_o, -- Enable for neuron memory
        wea => mem_neuron_wea_o, -- Enable for neuron memory -- Write enable for neuron memory
        web => mem_neuron_web_o, -- Enable for neuron memory -- Write enable for neuron memory
        addra => mem_neuron_addra_o, -- Address for neuron memory
        addrb => mem_neuron_addrb_o, -- Address for neuron memory
        dia => mem_neuron_dia_o, -- No write data for neuron memory
        dib => mem_neuron_dib_o, -- No write data for neuron memory
        doa => mem_neuron_doa_i, -- Output not used
        dob => mem_neuron_dob_i  -- Output not used
    );
    
    -- pragma translate_off
    debug_main_state <= main_state;
    debug_next_state <= main_next_state;
    debug_last_state <= main_last_state;
    debug_timestep_pending <= timestep_pending;
    debug_current_event <= current_event;
    debug_event_valid <= event_valid;
    debug_read_cycle <= read_cycle_counter;
    debug_total_coords_to_update <= total_coords_to_update;
    --ram spoof signals
    debug_mem_neuron_wea <= mem_neuron_wea_o;
    debug_mem_neuron_web <= mem_neuron_web_o;
    debug_mem_neuron_ena <= mem_neuron_ena_o;
    debug_mem_neuron_enb <= mem_neuron_enb_o;
    debug_mem_neuron_addra <= mem_neuron_addra_o;
    debug_mem_neuron_addrb <= mem_neuron_addrb_o;
    debug_mem_neuron_dia <= mem_neuron_dia_o;
    debug_mem_neuron_dib <= mem_neuron_dib_o;
    debug_mem_neuron_doa <= mem_neuron_doa_i;
    debug_mem_neuron_dob <= mem_neuron_dob_i;

    debug_main_state_vec <= state_to_slv(main_state);
    debug_next_state_vec <= state_to_slv(main_next_state);
    debug_last_state_vec <= state_to_slv(main_last_state);
    -- pragma translate_on

end architecture rtl;