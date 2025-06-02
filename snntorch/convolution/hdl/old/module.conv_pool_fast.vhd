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
        debug_mem_neuron_addra, debug_mem_neuron_addrb : out std_logic_vector(9 downto 0);
        debug_mem_neuron_dia, debug_mem_neuron_dib : out std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0);
        debug_mem_neuron_doa, debug_mem_neuron_dob : out std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0);
        
        debug_convolution_in_progress : out std_logic;
        debug_pooling_in_progress : out std_logic;
        debug_total_coords_to_update : out integer;
        debug_main_state_vec, debug_next_state_vec, debug_last_state_vec : out std_logic_vector(2 downto 0)
        -- pragma translate_on
    );
end entity conv_pool_fast;

architecture rtl of conv_pool_fast is

    -- Parametric constants for any image size
    constant POOL_WINDOWS_X : integer := IMG_WIDTH / 2;
    constant POOL_WINDOWS_Y : integer := IMG_HEIGHT / 2;
    constant TOTAL_POOL_WINDOWS : integer := POOL_WINDOWS_X * POOL_WINDOWS_Y;
    constant ADDR_WIDTH : integer := integer(ceil(log2(real(IMG_WIDTH * IMG_HEIGHT))));

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
    signal tresholds_weights : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');
    signal decay_weights : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');

    -- Convolution signals
    signal convolution_in_progress : std_logic := '0';
    signal conv_counter : integer range 0 to (KERNEL_SIZE ** 2) := 0;
    
    -- Parametric pooling signals - work for any image size
    signal pooling_in_progress : std_logic := '0';
    signal pooling_counter : integer range 0 to TOTAL_POOL_WINDOWS := 0;
    signal pooling_window_counter : integer range 1 to 5 := 1;
    signal pool_window_x : integer range 0 to POOL_WINDOWS_X - 1 := 0;
    signal pool_window_y : integer range 0 to POOL_WINDOWS_Y - 1 := 0;
    signal temp_pooling_window : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');
    -- FIXED: update_neurons should be spike vector size, not full membrane size
    signal update_neurons : std_logic_vector(CHANNELS_OUT - 1 downto 0) := (others => '0');
    
    -- Memory arbiter interface signals
    -- Convolution requests
    signal conv_mem_req : std_logic := '0';
    signal conv_mem_we : std_logic := '0';
    signal conv_mem_addr_a : std_logic_vector(ADDR_WIDTH-1 downto 0) := (others => '0');
    signal conv_mem_addr_b : std_logic_vector(ADDR_WIDTH-1 downto 0) := (others => '0');
    signal conv_mem_data_in : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');
    
    -- Pooling requests  
    signal pool_mem_req : std_logic := '0';
    signal pool_mem_we : std_logic := '0';
    signal pool_mem_addr_a : std_logic_vector(ADDR_WIDTH-1 downto 0) := (others => '0');
    signal pool_mem_addr_b : std_logic_vector(ADDR_WIDTH-1 downto 0) := (others => '0');
    signal pool_mem_data_in : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');
    
    -- Actual memory signals (controlled by arbiter)
    signal mem_neuron_wea_o, mem_neuron_web_o, mem_neuron_ena_o, mem_neuron_enb_o : std_logic := '0';
    signal mem_neuron_addra_o, mem_neuron_addrb_o : std_logic_vector(ADDR_WIDTH-1 downto 0) := (others => '0');
    signal mem_neuron_dia_o, mem_neuron_dib_o : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');
    signal mem_neuron_doa_i, mem_neuron_dob_i : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0) := (others => '0');

begin

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
                if read_cycle_counter = 2 then
                    main_next_state <= EVENT_CONV;
                end if;
                
            when EVENT_CONV => 
                if convolution_in_progress = '0' then
                    main_next_state <= IDLE;
                end if;
                
            when PAUSE => 
                main_next_state <= main_last_state;
                
            when POOL =>
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
                    read_cycle_counter <= 1;
                end if;
            end if;
        end if;
    end process read_cycle_management;

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
                if main_state = READ_REQUEST and read_cycle_counter = 2 then
                    temp_event := bus_to_event_tensor(
                        event_fifo_bus_i,
                        BITS_PER_COORD,
                        1
                    );
                    event_valid <= '1';
                    current_event <= temp_event;
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
                    total_coords_to_update <= coords_index;
                end if;
                
                if main_state = EVENT_CONV and main_next_state = IDLE then
                    event_valid <= '0';
                end if;
            end if;
        end if;
    end process event_capture;

    -- CLEAN: Convolution control (only generates requests)
    convolution_control : process (clk, rst_i)
    begin
        if rising_edge(clk) then
            if rst_i = '1' then
                convolution_in_progress <= '0';
                conv_counter <= 0;
                conv_mem_req <= '0';
            else
                case main_state is
                when READ_REQUEST => 
                    if read_cycle_counter = 2 then
                        convolution_in_progress <= '1';
                        conv_counter <= 0;
                    end if;
                    
                when EVENT_CONV =>
                    if conv_counter = 0 then
                        -- Request first read
                        conv_mem_req <= '1';
                        conv_mem_we <= '0';
                        conv_mem_addr_a <= std_logic_vector(resize(unsigned(fast_calc_address(
                            coords_to_update(conv_counter), IMG_WIDTH, ADDR_WIDTH)), ADDR_WIDTH));
                            
                    elsif conv_counter < total_coords_to_update then
                        -- Request read + write
                        conv_mem_req <= '1';
                        conv_mem_we <= '1';
                        conv_mem_addr_a <= std_logic_vector(resize(unsigned(fast_calc_address(
                            coords_to_update(conv_counter), IMG_WIDTH, ADDR_WIDTH)), ADDR_WIDTH));
                        conv_mem_addr_b <= std_logic_vector(resize(unsigned(fast_calc_address(
                            coords_to_update(conv_counter - 1), IMG_WIDTH, ADDR_WIDTH)), ADDR_WIDTH));
                        conv_mem_data_in <= convolution_1d(
                            kernel_weights(conv_counter - 1),
                            mem_neuron_doa_i,
                            BITS_PER_NEURON,
                            CHANNELS_OUT
                        );
                        
                    elsif conv_counter = total_coords_to_update then
                        -- Final write
                        conv_mem_req <= '1';
                        conv_mem_we <= '1';
                        conv_mem_addr_b <= std_logic_vector(resize(unsigned(fast_calc_address(
                            coords_to_update(conv_counter - 1), IMG_WIDTH, ADDR_WIDTH)), ADDR_WIDTH));
                        conv_mem_data_in <= convolution_1d(
                            kernel_weights(conv_counter - 1),
                            mem_neuron_doa_i,
                            BITS_PER_NEURON,
                            CHANNELS_OUT
                        );
                    else 
                        convolution_in_progress <= '0';
                        conv_mem_req <= '0';
                    end if;
                    
                    if conv_counter <= total_coords_to_update then
                        conv_counter <= conv_counter + 1;
                    end if;
                    
                when others =>
                    conv_mem_req <= '0';
                end case;    
            end if;
        end if;
    end process convolution_control;

    -- CLEAN: Parametric pooling control (only generates requests)
    pooling_control : process (clk, rst_i)
        variable pooled_accumulator : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0);
        variable decayed_data : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0);
        variable updated_membrane : std_logic_vector((CHANNELS_OUT * BITS_PER_NEURON) - 1 downto 0);
        variable current_pixel_x, current_pixel_y : integer;
    begin
        if rising_edge(clk) then
            if rst_i = '1' then
                pooling_in_progress <= '0';
                pooling_counter <= 0;
                pooling_window_counter <= 1;
                pool_window_x <= 0;
                pool_window_y <= 0;
                pool_mem_req <= '0';
            else
                case main_state is
                when POOL =>
                    case pooling_window_counter is
                    when 1 =>  -- Read first pixel of 2x2 window
                        current_pixel_x := pool_window_x * 2;
                        current_pixel_y := pool_window_y * 2;
                        
                        pool_mem_req <= '1';
                        pool_mem_we <= '0';
                        pool_mem_addr_a <= std_logic_vector(to_unsigned(
                            current_pixel_y * IMG_WIDTH + current_pixel_x, ADDR_WIDTH));
                        
                        pooled_accumulator := (others => '0');
                        pooling_window_counter <= 2;
                        
                    when 2 =>  -- Process first pixel, read second
                        decayed_data := apply_decay(
                            mem_neuron_doa_i, decay_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        pooled_accumulator := decayed_data;
                        updated_membrane := check_threshold_and_reset(
                            decayed_data, tresholds_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        
                        -- Write back first pixel, read second
                        current_pixel_x := pool_window_x * 2 + 1;
                        current_pixel_y := pool_window_y * 2;
                        
                        pool_mem_req <= '1';
                        pool_mem_we <= '1';
                        pool_mem_addr_a <= std_logic_vector(to_unsigned(
                            current_pixel_y * IMG_WIDTH + current_pixel_x, ADDR_WIDTH));
                        pool_mem_addr_b <= std_logic_vector(to_unsigned(
                            (pool_window_y * 2) * IMG_WIDTH + (pool_window_x * 2), ADDR_WIDTH));
                        pool_mem_data_in <= updated_membrane;
                        
                        pooling_window_counter <= 3;
                        
                    when 3 =>  -- Process second pixel, read third
                        decayed_data := apply_decay(
                            mem_neuron_doa_i, decay_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        pooled_accumulator := add_multichannel_vectors(
                            pooled_accumulator, decayed_data, CHANNELS_OUT, BITS_PER_NEURON);
                        updated_membrane := check_threshold_and_reset(
                            decayed_data, tresholds_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        
                        -- Write back second pixel, read third
                        current_pixel_x := pool_window_x * 2;
                        current_pixel_y := pool_window_y * 2 + 1;
                        
                        pool_mem_req <= '1';
                        pool_mem_we <= '1';
                        pool_mem_addr_a <= std_logic_vector(to_unsigned(
                            current_pixel_y * IMG_WIDTH + current_pixel_x, ADDR_WIDTH));
                        pool_mem_addr_b <= std_logic_vector(to_unsigned(
                            (pool_window_y * 2) * IMG_WIDTH + (pool_window_x * 2 + 1), ADDR_WIDTH));
                        pool_mem_data_in <= updated_membrane;
                        
                        pooling_window_counter <= 4;
                        
                    when 4 =>  -- Process third pixel, read fourth
                        decayed_data := apply_decay(
                            mem_neuron_doa_i, decay_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        pooled_accumulator := add_multichannel_vectors(
                            pooled_accumulator, decayed_data, CHANNELS_OUT, BITS_PER_NEURON);
                        updated_membrane := check_threshold_and_reset(
                            decayed_data, tresholds_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        
                        -- Write back third pixel, read fourth
                        current_pixel_x := pool_window_x * 2 + 1;
                        current_pixel_y := pool_window_y * 2 + 1;
                        
                        pool_mem_req <= '1';
                        pool_mem_we <= '1';
                        pool_mem_addr_a <= std_logic_vector(to_unsigned(
                            current_pixel_y * IMG_WIDTH + current_pixel_x, ADDR_WIDTH));
                        pool_mem_addr_b <= std_logic_vector(to_unsigned(
                            (pool_window_y * 2 + 1) * IMG_WIDTH + (pool_window_x * 2), ADDR_WIDTH));
                        pool_mem_data_in <= updated_membrane;
                        
                        pooling_window_counter <= 5;
                        
                    when 5 =>  -- Process fourth pixel, move to next window
                        decayed_data := apply_decay(
                            mem_neuron_doa_i, decay_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        pooled_accumulator := add_multichannel_vectors(
                            pooled_accumulator, decayed_data, CHANNELS_OUT, BITS_PER_NEURON);
                        
                        -- Store final pooled result and check for output spikes
                        temp_pooling_window <= pooled_accumulator;
                        -- FIXED: Now correctly assigning spike vector to spike vector signal
                        update_neurons <= check_pooled_threshold(
                            pooled_accumulator, tresholds_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        
                        updated_membrane := check_threshold_and_reset(
                            decayed_data, tresholds_weights, CHANNELS_OUT, BITS_PER_NEURON);
                        
                        -- Write back fourth pixel
                        pool_mem_req <= '1';
                        pool_mem_we <= '1';
                        pool_mem_addr_b <= std_logic_vector(to_unsigned(
                            (pool_window_y * 2 + 1) * IMG_WIDTH + (pool_window_x * 2 + 1), ADDR_WIDTH));
                        pool_mem_data_in <= updated_membrane;
                        
                        -- Move to next pool window (PARAMETRIC)
                        pooling_counter <= pooling_counter + 1;
                        
                        if pool_window_x < POOL_WINDOWS_X - 1 then
                            pool_window_x <= pool_window_x + 1;
                            pooling_window_counter <= 1;
                        else
                            pool_window_x <= 0;
                            if pool_window_y < POOL_WINDOWS_Y - 1 then
                                pool_window_y <= pool_window_y + 1;
                                pooling_window_counter <= 1;
                            else
                                -- All windows completed
                                pooling_in_progress <= '0';
                                pool_mem_req <= '0';
                            end if;
                        end if;
                        
                    when others =>
                        pooling_window_counter <= 1;
                    end case;
                    
                when others => 
                    if timestep_pending = '1' and pooling_in_progress = '0' then 
                        pooling_in_progress <= '1'; 
                        pooling_counter <= 0;
                        pooling_window_counter <= 1;
                        pool_window_x <= 0;
                        pool_window_y <= 0;
                    end if;
                    pool_mem_req <= '0';
                end case;
            end if;
        end if;
    end process pooling_control;

    -- MEMORY ARBITER: Clean separation of concerns
    memory_arbiter : process (clk, rst_i)
    begin
        if rising_edge(clk) then
            if rst_i = '1' then
                mem_neuron_ena_o <= '0';
                mem_neuron_enb_o <= '0';
                mem_neuron_wea_o <= '0';
                mem_neuron_web_o <= '0';
                mem_neuron_addra_o <= (others => '0');
                mem_neuron_addrb_o <= (others => '0');
                mem_neuron_dia_o <= (others => '0');
                mem_neuron_dib_o <= (others => '0');
            else
                case main_state is
                when EVENT_CONV =>
                    if conv_mem_req = '1' then
                        mem_neuron_ena_o <= '1';
                        mem_neuron_enb_o <= conv_mem_we;
                        mem_neuron_wea_o <= '0';
                        mem_neuron_web_o <= conv_mem_we;
                        mem_neuron_addra_o <= conv_mem_addr_a;
                        mem_neuron_addrb_o <= conv_mem_addr_b;
                        mem_neuron_dib_o <= conv_mem_data_in;
                    else
                        mem_neuron_ena_o <= '0';
                        mem_neuron_enb_o <= '0';
                        mem_neuron_wea_o <= '0';
                        mem_neuron_web_o <= '0';
                    end if;
                    
                when POOL =>
                    if pool_mem_req = '1' then
                        mem_neuron_ena_o <= '1';
                        mem_neuron_enb_o <= pool_mem_we;
                        mem_neuron_wea_o <= '0';
                        mem_neuron_web_o <= pool_mem_we;
                        mem_neuron_addra_o <= pool_mem_addr_a;
                        mem_neuron_addrb_o <= pool_mem_addr_b;
                        mem_neuron_dib_o <= pool_mem_data_in;
                    else
                        mem_neuron_ena_o <= '0';
                        mem_neuron_enb_o <= '0';
                        mem_neuron_wea_o <= '0';
                        mem_neuron_web_o <= '0';
                    end if;
                    
                when others =>
                    mem_neuron_ena_o <= '0';
                    mem_neuron_enb_o <= '0';
                    mem_neuron_wea_o <= '0';
                    mem_neuron_web_o <= '0';
                end case;
            end if;
        end if;
    end process memory_arbiter;

    -- Memory instance
    MEM_NEURON : entity work.TRUE_DUAL_PORT_READ_FIRST
    generic map(
        RAM_DEPTH => IMG_WIDTH * IMG_HEIGHT,
        DATA_WIDTH => CHANNELS_OUT * BITS_PER_NEURON,
        ADDR_WIDTH => ADDR_WIDTH
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
    debug_convolution_in_progress <= convolution_in_progress;
    debug_pooling_in_progress <= pooling_in_progress;
    
    -- Memory debug signals (now from arbiter)
    debug_mem_neuron_wea <= mem_neuron_wea_o;
    debug_mem_neuron_web <= mem_neuron_web_o;
    debug_mem_neuron_ena <= mem_neuron_ena_o;
    debug_mem_neuron_enb <= mem_neuron_enb_o;
    debug_mem_neuron_addra <= std_logic_vector(resize(unsigned(mem_neuron_addra_o), 10));
    debug_mem_neuron_addrb <= std_logic_vector(resize(unsigned(mem_neuron_addrb_o), 10));
    debug_mem_neuron_dia <= mem_neuron_dia_o;
    debug_mem_neuron_dib <= mem_neuron_dib_o;
    debug_mem_neuron_doa <= mem_neuron_doa_i;
    debug_mem_neuron_dob <= mem_neuron_dob_i;

    debug_main_state_vec <= state_to_slv(main_state);
    debug_next_state_vec <= state_to_slv(main_next_state);
    debug_last_state_vec <= state_to_slv(main_last_state);
    -- pragma translate_on

end architecture rtl;