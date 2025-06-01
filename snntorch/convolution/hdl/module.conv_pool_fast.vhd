library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.conv_pool_pkg.all;

entity conv_pool_fast is
    generic(
        CHANNELS_OUT : integer := 12;
        BITS_PER_NEURON : integer := 9;
        BITS_PER_WEIGHT : integer := 9;
        IMG_WIDTH : integer := 32;
        IMG_HEIGHT : integer := 32;
        BITS_PER_COORD : integer := 8;
        KERNEL_SIZE : integer := 3
        --RAM_FILE : string
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
        -- expose thread signals
        
        debug_convolution_in_progress : out std_logic;
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
    signal total_coords_to_update : integer range 0 to (KERNEL_SIZE ** 2) := 0; -- 0 indexed 
    type kernel_weights_t is array (0 to (KERNEL_SIZE ** 2) -1) of std_logic_vector((CHANNELS_OUT * BITS_PER_WEIGHT) - 1 downto 0);
    signal kernel_weights : kernel_weights_t := (others => (others => '0'));
    signal tresholds_weights : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0'); -- Tresholds for each channel
    signal decay_weights : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0'); -- Decay weights for each channel

    -- Convolution signals
    signal convolution_in_progress : std_logic := '0';
    signal conv_counter : integer range 0 to (KERNEL_SIZE ** 2) := 0; -- Counter for convolution operations
    
    -- pooling signals
    signal pooling_in_progress : std_logic := '0';
    signal pooling_counter : integer := 0;
    signal pooling_window_counter : integer range 1 to 4 := 1;
    signal temp_pooling_window : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');
    signal pooling_anchor : vector2_t := (x => 0, y => 0);
    signal update_neurons : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0'); -- Updated neurons after pooling
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
                if convolution_in_progress = '0' then
                    main_next_state <= IDLE;
                end if;
                
            when PAUSE => 
                main_next_state <= main_last_state;
                
            when POOL =>
                -- Pool processing, then back to IDLE
                if pooling_in_progress = '0' then
                    main_next_state <= IDLE;
                end if;
                
            when CONFIG =>
                main_next_state <= IDLE;
                
            when RESET => 
                main_next_state <= IDLE;
            end case;
        end if;
    end process state_machine_control;

    pooling_control : process (clk, rst_i)
        -- temp pooling sum is always the pooling window
        variable temp_pooling_sum : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := temp_pooling_window;
        variable next_read_address : std_logic_vector(9 downto 0) := (others => '0');
        variable pooling_anchor_temp : vector2_t := pooling_anchor;
    begin
    if rising_edge(clk) then
    if rst_i = '1' then
    
    
    else
        case main_state is
        WHEN POOL =>
            -- reset pooling if the last pixel was processed
            if pooling_counter = IMG_HEIGHT * IMG_WIDTH then
                pooling_in_progress <= '0'; -- Reset pooling in progress
                pooling_counter <= 0; -- Reset pooling counter
                pooling_window_counter <= 1; -- Reset pooling window counter
            else 
            -- pooling in progress
            pooling_counter <= pooling_counter + 1;
            end if;

            -- add the current bus reading to the pooling window
            temp_pooling_sum := add_multichannel_vectors(mem_neuron_doa_i, temp_pooling_sum, BITS_PER_NEURON, CHANNELS_OUT);

                case pooling_window_counter is
                when 1 =>
                    -- no need to update pooling anchor, just read the first pixel
                when 2 =>
                    pooling_anchor_temp.x := pooling_anchor.x + 1;
                when 3 => 
                    pooling_anchor_temp.y := pooling_anchor.y + 1;
                when 4 =>
                    pooling_anchor_temp.x := pooling_anchor.x + 1;
                    pooling_anchor_temp.y := pooling_anchor.y + 1;


                -- update pooling anchor
                if pooling_anchor.x < IMG_WIDTH - 1 then
                    pooling_anchor.x <= pooling_anchor.x + 2; --stride by 2
                else
                    pooling_anchor.x <= 0;
                    if pooling_anchor.y < IMG_HEIGHT - 1 then
                        pooling_anchor.y <= pooling_anchor.y + 2;
                    else
                        pooling_anchor.y <= 0; -- Reset to top-left corner
                    end if;
                end if;
                end case;

            -- update ppooling window counter
            if pooling_counter = 4 then 
            pooling_counter <= 1; 
            else
            pooling_counter <= pooling_counter + 1; 
            end if;
            
            -- read next address
            next_read_address := fast_calc_address(pooling_anchor_temp, IMG_WIDTH);

        WHEN OTHERS => 
        if timestep_pending = '1' then 
        pooling_in_progress <= '1'; 
        pooling_counter <= 0;
        pooling_window_counter <= 1;
        -- enable neuron memory read
        mem_neuron_ena_o <= '1';
        mem_neuron_addra_o <= (others => '0'); -- first address
        
        end if;
        end case;

    end if;
    end if;
    end process pooling_control;

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

        variable previous_address : std_logic_vector(9 downto 0) := (others => '0');
        variable current_address : std_logic_vector(9 downto 0) := (others => '0');

    begin
    if rising_edge(clk) then
    if rst_i = '1' then

    else
        case main_state is
        WHEN READ_REQUEST => 
            if read_cycle_counter = 2 then
                convolution_in_progress <= '1';
                conv_counter <= 0;
            end if;
        WHEN EVENT_CONV =>
            -- write on b, read on a
            if conv_counter = 0 then
                mem_neuron_ena_o <= '1'; 
                mem_neuron_enb_o <= '1'; 

                -- set read address 
                mem_neuron_addra_o <= fast_calc_address(
                    coords_to_update(conv_counter),
                    IMG_WIDTH
                );

                
            elsif conv_counter < total_coords_to_update then

                mem_neuron_web_o <= '1'; -- Enable write on b port
                mem_neuron_addrb_o <= fast_calc_address(
                    coords_to_update(conv_counter - 1),
                    IMG_WIDTH
                );
                mem_neuron_addra_o <= fast_calc_address(
                    coords_to_update(conv_counter),
                    IMG_WIDTH
                );
                mem_neuron_dib_o <= convolution_1d(
                    kernel_weights(conv_counter - 1),
                    mem_neuron_doa_i,
                    BITS_PER_NEURON,
                    CHANNELS_OUT
                );

            elsif conv_counter = total_coords_to_update then
                --update last address
                mem_neuron_addrb_o <= fast_calc_address(
                    coords_to_update(conv_counter - 1),
                    IMG_WIDTH
                );
                mem_neuron_dib_o <= convolution_1d(
                    kernel_weights(conv_counter - 1),
                    mem_neuron_doa_i,
                    BITS_PER_NEURON,
                    CHANNELS_OUT
                );
            else 
                convolution_in_progress <= '0';
                mem_neuron_ena_o <= '0'; -- Disable neuron memory read
                mem_neuron_enb_o <= '0'; -- Disable neuron memory write
                mem_neuron_wea_o <= '0'; -- Disable write on a port
                mem_neuron_web_o <= '0'; -- Disable write on b port
                mem_neuron_addra_o <= (others => '0'); -- Reset address for a port
                mem_neuron_addrb_o <= (others => '0'); -- Reset address for b port
                mem_neuron_dia_o <= (others => '0'); -- Reset data input for a port
                mem_neuron_dib_o <= (others => '0'); -- Reset data input for b port
            end if;

            conv_counter <= conv_counter + 1;

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
        --RAM_FILE => RAM_FILE  -- Use the provided RAM initialization file
    )
    port map(
        clka => clk,
        clkb => clk,
        ena => mem_neuron_ena_o, 
        enb => mem_neuron_enb_o, 
        wea => mem_neuron_wea_o, 
        web => mem_neuron_web_o, 
        addra => mem_neuron_addra_o, 
        addrb => mem_neuron_addrb_o, 
        dia => mem_neuron_dia_o, 
        dib => mem_neuron_dib_o, 
        doa => mem_neuron_doa_i, 
        dob => mem_neuron_dob_i  
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

    debug_convolution_in_progress <= convolution_in_progress;

    debug_main_state_vec <= state_to_slv(main_state);
    debug_next_state_vec <= state_to_slv(main_next_state);
    debug_last_state_vec <= state_to_slv(main_last_state);
    -- pragma translate_on

end architecture rtl;