library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.conv_pool_pkg.all;

entity pooling is
    generic(
        IMG_WIDTH        : integer := 32;
        IMG_HEIGHT       : integer := 32;
        NEURON_BIT_WIDTH : integer := 9;
        POOL_SIZE        : integer := 2;     -- 2x2 pooling window
        CHANNELS         : integer := 12;
        ADDR_WIDTH       : integer := 10;
        RESET_VALUE      : integer := 0;     -- Reset value after spike
        COORD_WIDTH      : integer := 8      -- Width for coordinate encoding
    );
    port(
        -- Standard control signals
        clk              : in  std_logic;
        rst_i            : in  std_logic;
        enable_i         : in  std_logic;
        
        -- Control interface
        start_pooling_i  : in  std_logic;
        pooling_done_o   : out std_logic;
        
        -- Input memory interface (membrane potentials)
        mem_read_addr_o  : out std_logic_vector(ADDR_WIDTH-1 downto 0);
        mem_read_en_o    : out std_logic;
        mem_read_data_i  : in  std_logic_vector(CHANNELS * NEURON_BIT_WIDTH - 1 downto 0);
        mem_write_addr_o : out std_logic_vector(ADDR_WIDTH-1 downto 0);
        mem_write_en_o   : out std_logic;
        mem_write_data_o : out std_logic_vector(CHANNELS * NEURON_BIT_WIDTH - 1 downto 0);
        
        -- FIFO output interface (spike events with coordinates)
        fifo_write_en_o  : out std_logic;
        fifo_write_data_o : out std_logic_vector(2 * COORD_WIDTH + CHANNELS - 1 downto 0);  -- [x][y][spike_vector]
        fifo_full_i      : in  std_logic;
        
        -- Status signals
        busy_o           : out std_logic;
        
        -- Debug signals
        debug_state_o       : out std_logic_vector(2 downto 0);
        debug_window_x_o    : out integer range 0 to IMG_WIDTH/POOL_SIZE;
        debug_window_y_o    : out integer range 0 to IMG_HEIGHT/POOL_SIZE;
        debug_pixel_idx_o   : out integer range 0 to POOL_SIZE**2;
        debug_spike_count_o : out integer range 0 to CHANNELS;
        debug_accum_ch0_o   : out integer range -1024 to 1023;
        debug_accum_ch1_o   : out integer range -1024 to 1023
    );
    
end entity pooling;

architecture rtl of pooling is

    type pool_state_t is (
        IDLE,
        READ_PIXEL,
        PROCESS_PIXEL,
        COMPUTE_SPIKES,
        WRITE_OUTPUT,
        NEXT_WINDOW,
        DONE
    );
    
    signal current_state, next_state : pool_state_t := IDLE;
    
    -- Channel-wise parameters (internal arrays)
    type threshold_array_t is array (0 to CHANNELS - 1) of signed(NEURON_BIT_WIDTH - 1 downto 0);
    type decay_array_t is array (0 to CHANNELS - 1) of signed(NEURON_BIT_WIDTH - 1 downto 0);
    
    -- Initialize thresholds and decay values (these would be configured per application)
    function init_thresholds return threshold_array_t is
        variable thresholds : threshold_array_t;
    begin
        for ch in 0 to CHANNELS - 1 loop
            -- Lower thresholds for testing: 80, 100, 120, 140
            thresholds(ch) := to_signed(80 + ch * 20, NEURON_BIT_WIDTH);
        end loop;
        return thresholds;
    end function;
    
    function init_decay_values return decay_array_t is
        variable decay_vals : decay_array_t;
    begin
        for ch in 0 to CHANNELS - 1 loop
            -- Smaller decay values for testing: 2 for all channels
            decay_vals(ch) := to_signed(2, NEURON_BIT_WIDTH);
        end loop;
        return decay_vals;
    end function;
    
    signal channel_thresholds : threshold_array_t := init_thresholds;
    signal channel_decay : decay_array_t := init_decay_values;
    
    -- Window and pixel management
    signal window_x : integer range 0 to IMG_WIDTH/POOL_SIZE := 0;
    signal window_y : integer range 0 to IMG_HEIGHT/POOL_SIZE := 0;
    signal pixel_idx : integer range 0 to POOL_SIZE**2 := 0;
    signal pixel_x : integer range 0 to IMG_WIDTH := 0;
    signal pixel_y : integer range 0 to IMG_HEIGHT := 0;
    
    -- Membrane potential accumulators (one per channel)
    type membrane_accum_t is array (0 to CHANNELS - 1) of signed(NEURON_BIT_WIDTH + 2 downto 0);  -- Extra bits for accumulation
    signal membrane_sum : membrane_accum_t := (others => (others => '0'));
    
    -- Spike generation
    signal spike_vector : std_logic_vector(CHANNELS - 1 downto 0) := (others => '0');
    signal spike_count : integer range 0 to CHANNELS := 0;
    signal any_spikes : std_logic := '0';
    
    -- Control signals
    signal busy_reg : std_logic := '0';
    signal pooling_done_reg : std_logic := '0';
    signal read_pixel_data : std_logic_vector(CHANNELS * NEURON_BIT_WIDTH - 1 downto 0) := (others => '0');
    signal updated_pixel_data : std_logic_vector(CHANNELS * NEURON_BIT_WIDTH - 1 downto 0) := (others => '0');

    -- Helper function to convert state to std_logic_vector
    function state_to_slv(state : pool_state_t) return std_logic_vector is
    begin
        case state is
            when IDLE           => return "000";
            when READ_PIXEL     => return "001";
            when PROCESS_PIXEL  => return "010";
            when COMPUTE_SPIKES => return "011";
            when WRITE_OUTPUT   => return "100";
            when NEXT_WINDOW    => return "101";
            when DONE           => return "110";
            when others         => return "111";
        end case;
    end function;

    -- Helper function to calculate memory address from coordinates
    function calc_address(x : integer; y : integer) return std_logic_vector is
    begin
        return std_logic_vector(to_unsigned(y * IMG_WIDTH + x, ADDR_WIDTH));
    end function;

    -- Helper function to get pixel coordinates within window
    function get_pixel_coords(
        win_x : integer; 
        win_y : integer; 
        pix_idx : integer
    ) return vector2_t is
        variable coords : vector2_t;
        variable offset_x, offset_y : integer;
    begin
        offset_x := pix_idx mod POOL_SIZE;
        offset_y := pix_idx / POOL_SIZE;
        coords.x := win_x * POOL_SIZE + offset_x;
        coords.y := win_y * POOL_SIZE + offset_y;
        return coords;
    end function;

    -- Function to apply decay and check thresholds, then accumulate
    function process_pixel_with_decay(
        pixel_data : std_logic_vector;
        accumulators : membrane_accum_t;
        thresholds : threshold_array_t;
        decay_vals : decay_array_t;
        channels : integer;
        neuron_bits : integer;
        reset_val : integer
    ) return membrane_accum_t is
        variable new_accum : membrane_accum_t;
        variable membrane_val : signed(neuron_bits - 1 downto 0);
        variable decayed_val : signed(neuron_bits - 1 downto 0);
        variable threshold_val : signed(neuron_bits - 1 downto 0);
        variable decay_val : signed(neuron_bits - 1 downto 0);
        variable reset_value : signed(neuron_bits - 1 downto 0);
        variable min_val : signed(neuron_bits - 1 downto 0);
    begin
        reset_value := to_signed(reset_val, neuron_bits);
        min_val := to_signed(0, neuron_bits);  -- Don't go below 0
        
        for ch in 0 to channels - 1 loop
            -- Extract membrane potential for this channel
            membrane_val := signed(pixel_data((ch + 1) * neuron_bits - 1 downto ch * neuron_bits));
            
            -- Apply subtractive decay first
            decay_val := decay_vals(ch);
            decayed_val := membrane_val - decay_val;
            
            -- Ensure we don't go below 0
            if decayed_val < min_val then
                decayed_val := min_val;
            end if;
            
            -- Extract threshold for this channel
            threshold_val := thresholds(ch);
            
            -- Check if decayed membrane potential exceeds threshold
            if decayed_val >= threshold_val then
                -- Spike occurred - don't add to accumulator (spike-and-reset behavior)
                new_accum(ch) := accumulators(ch);
            else
                -- No spike - add decayed value to accumulator
                new_accum(ch) := accumulators(ch) + resize(decayed_val, neuron_bits + 3);
            end if;
        end loop;
        
        return new_accum;
    end function;

    -- Function to update pixel data (apply decay and reset spiked channels)
    function update_pixel_with_decay(
        pixel_data : std_logic_vector;
        thresholds : threshold_array_t;
        decay_vals : decay_array_t;
        channels : integer;
        neuron_bits : integer;
        reset_val : integer
    ) return std_logic_vector is
        variable updated_data : std_logic_vector(pixel_data'range);
        variable membrane_val : signed(neuron_bits - 1 downto 0);
        variable decayed_val : signed(neuron_bits - 1 downto 0);
        variable threshold_val : signed(neuron_bits - 1 downto 0);
        variable decay_val : signed(neuron_bits - 1 downto 0);
        variable reset_value : signed(neuron_bits - 1 downto 0);
        variable min_val : signed(neuron_bits - 1 downto 0);
    begin
        reset_value := to_signed(reset_val, neuron_bits);
        min_val := to_signed(0, neuron_bits);
        
        for ch in 0 to channels - 1 loop
            -- Extract membrane potential for this channel
            membrane_val := signed(pixel_data((ch + 1) * neuron_bits - 1 downto ch * neuron_bits));
            
            -- Apply subtractive decay first
            decay_val := decay_vals(ch);
            decayed_val := membrane_val - decay_val;
            
            -- Ensure we don't go below 0
            if decayed_val < min_val then
                decayed_val := min_val;
            end if;
            
            -- Extract threshold for this channel
            threshold_val := thresholds(ch);
            
            -- Check if decayed membrane potential exceeds threshold
            if decayed_val >= threshold_val then
                -- Reset this channel's membrane potential
                updated_data((ch + 1) * neuron_bits - 1 downto ch * neuron_bits) := std_logic_vector(reset_value);
            else
                -- Store decayed value
                updated_data((ch + 1) * neuron_bits - 1 downto ch * neuron_bits) := std_logic_vector(decayed_val);
            end if;
        end loop;
        
        return updated_data;
    end function;

    -- Function to generate spike vector from accumulated membrane potentials
    function generate_spike_vector(
        accumulators : membrane_accum_t;
        thresholds : threshold_array_t;
        channels : integer;
        neuron_bits : integer
    ) return std_logic_vector is
        variable spike_vec : std_logic_vector(channels - 1 downto 0);
        variable threshold_val : signed(neuron_bits - 1 downto 0);
        variable accum_val : signed(neuron_bits + 2 downto 0);
    begin
        spike_vec := (others => '0');
        
        for ch in 0 to channels - 1 loop
            -- Extract threshold for this channel
            threshold_val := thresholds(ch);
            
            -- Get accumulated value
            accum_val := accumulators(ch);
            
            -- Check if accumulated sum exceeds threshold
            if accum_val >= resize(threshold_val, neuron_bits + 3) then
                spike_vec(ch) := '1';
            end if;
        end loop;
        
        return spike_vec;
    end function;

    -- Function to count spikes in vector
    function count_spikes(spike_vec : std_logic_vector) return integer is
        variable count : integer := 0;
    begin
        for i in spike_vec'range loop
            if spike_vec(i) = '1' then
                count := count + 1;
            end if;
        end loop;
        return count;
    end function;

    -- Function to check if any spikes occurred
    function has_spikes(spike_vec : std_logic_vector) return std_logic is
    begin
        for i in spike_vec'range loop
            if spike_vec(i) = '1' then
                return '1';
            end if;
        end loop;
        return '0';
    end function;

    -- Function to create FIFO output data with coordinates
    function create_fifo_output(
        win_x : integer;
        win_y : integer;
        spike_vec : std_logic_vector;
        coord_width : integer
    ) return std_logic_vector is
        variable output_data : std_logic_vector(2 * coord_width + spike_vec'length - 1 downto 0);
        variable x_coord : std_logic_vector(coord_width - 1 downto 0);
        variable y_coord : std_logic_vector(coord_width - 1 downto 0);
    begin
        -- Convert coordinates to std_logic_vector
        x_coord := std_logic_vector(to_unsigned(win_x, coord_width));
        y_coord := std_logic_vector(to_unsigned(win_y, coord_width));
        
        -- Pack data: [x_coord][y_coord][spike_vector]
        output_data(spike_vec'length - 1 downto 0) := spike_vec;
        output_data(coord_width + spike_vec'length - 1 downto spike_vec'length) := y_coord;
        output_data(2 * coord_width + spike_vec'length - 1 downto coord_width + spike_vec'length) := x_coord;
        
        return output_data;
    end function;

begin

    -- Output assignments
    busy_o <= busy_reg;
    pooling_done_o <= pooling_done_reg;
    debug_state_o <= state_to_slv(current_state);
    debug_window_x_o <= window_x;
    debug_window_y_o <= window_y;
    debug_pixel_idx_o <= pixel_idx;
    debug_spike_count_o <= spike_count;
    debug_accum_ch0_o <= to_integer(membrane_sum(0)(NEURON_BIT_WIDTH - 1 downto 0));
    debug_accum_ch1_o <= to_integer(membrane_sum(1)(NEURON_BIT_WIDTH - 1 downto 0));
    
    -- Memory interface assignments
    mem_read_addr_o <= calc_address(pixel_x, pixel_y);
    mem_write_addr_o <= calc_address(pixel_x, pixel_y);
    mem_write_data_o <= updated_pixel_data;
    
    -- FIFO output data assignment
    fifo_write_data_o <= create_fifo_output(window_x, window_y, spike_vector, COORD_WIDTH);

    -- State register
    state_register : process(clk, rst_i)
    begin
        if rst_i = '1' then
            current_state <= IDLE;
        elsif rising_edge(clk) then
            if enable_i = '1' then
                current_state <= next_state;
            end if;
        end if;
    end process state_register;

    -- Next state logic
    state_machine : process(all)
    begin
        -- Default: stay in current state
        next_state <= current_state;
        
        case current_state is
            when IDLE =>
                if start_pooling_i = '1' then
                    next_state <= READ_PIXEL;
                end if;
                
            when READ_PIXEL =>
                -- Read current pixel
                next_state <= PROCESS_PIXEL;
                
            when PROCESS_PIXEL =>
                -- Process the pixel data and move to next pixel or compute spikes
                if pixel_idx >= POOL_SIZE**2 - 1 then
                    next_state <= COMPUTE_SPIKES;
                else
                    next_state <= READ_PIXEL;  -- Read next pixel
                end if;
                
            when COMPUTE_SPIKES =>
                -- Generate spike vector from accumulated data
                if any_spikes = '1' and fifo_full_i = '0' then
                    next_state <= WRITE_OUTPUT;
                else
                    next_state <= NEXT_WINDOW;  -- Skip output if no spikes or FIFO full
                end if;
                
            when WRITE_OUTPUT =>
                -- Write spike event to FIFO
                next_state <= NEXT_WINDOW;
                
            when NEXT_WINDOW =>
                -- Move to next window or finish
                if window_x >= (IMG_WIDTH/POOL_SIZE) - 1 and window_y >= (IMG_HEIGHT/POOL_SIZE) - 1 then
                    next_state <= DONE;
                else
                    next_state <= READ_PIXEL;
                end if;
                
            when DONE =>
                next_state <= IDLE;
        end case;
    end process state_machine;

    -- Main control process
    control_process : process(clk, rst_i)
        variable coords : vector2_t;
    begin
        if rst_i = '1' then
            busy_reg <= '0';
            pooling_done_reg <= '0';
            window_x <= 0;
            window_y <= 0;
            pixel_idx <= 0;
            pixel_x <= 0;
            pixel_y <= 0;
            membrane_sum <= (others => (others => '0'));
            spike_vector <= (others => '0');
            spike_count <= 0;
            any_spikes <= '0';
            mem_read_en_o <= '0';
            mem_write_en_o <= '0';
            fifo_write_en_o <= '0';
            
        elsif rising_edge(clk) and enable_i = '1' then
            -- Default: clear one-cycle signals
            pooling_done_reg <= '0';
            mem_read_en_o <= '0';
            mem_write_en_o <= '0';
            fifo_write_en_o <= '0';
            
            case current_state is
                when IDLE =>
                    if start_pooling_i = '1' then
                        busy_reg <= '1';
                        window_x <= 0;
                        window_y <= 0;
                        pixel_idx <= 0;
                        membrane_sum <= (others => (others => '0'));
                        
                        -- Calculate first pixel coordinates
                        coords := get_pixel_coords(0, 0, 0);
                        pixel_x <= coords.x;
                        pixel_y <= coords.y;
                    else
                        busy_reg <= '0';
                    end if;
                    
                when READ_PIXEL =>
                    -- Issue read request for current pixel
                    mem_read_en_o <= '1';
                    
                when PROCESS_PIXEL =>
                    -- Capture read data and process it
                    read_pixel_data <= mem_read_data_i;
                    
                    -- Update membrane accumulators (with decay and threshold checking)
                    membrane_sum <= process_pixel_with_decay(
                        mem_read_data_i,
                        membrane_sum,
                        channel_thresholds,
                        channel_decay,
                        CHANNELS,
                        NEURON_BIT_WIDTH,
                        RESET_VALUE
                    );
                    
                    -- Update pixel data (apply decay and reset spiked channels)
                    updated_pixel_data <= update_pixel_with_decay(
                        mem_read_data_i,
                        channel_thresholds,
                        channel_decay,
                        CHANNELS,
                        NEURON_BIT_WIDTH,
                        RESET_VALUE
                    );
                    
                    -- Write back updated pixel data
                    mem_write_en_o <= '1';
                    
                    -- Move to next pixel in window
                    if pixel_idx < POOL_SIZE**2 - 1 then
                        pixel_idx <= pixel_idx + 1;
                        coords := get_pixel_coords(window_x, window_y, pixel_idx + 1);
                        pixel_x <= coords.x;
                        pixel_y <= coords.y;
                    end if;
                    
                when COMPUTE_SPIKES =>
                    -- Generate spike vector from accumulated membrane potentials
                    spike_vector <= generate_spike_vector(
                        membrane_sum,
                        channel_thresholds,
                        CHANNELS,
                        NEURON_BIT_WIDTH
                    );
                    
                    spike_count <= count_spikes(generate_spike_vector(
                        membrane_sum,
                        channel_thresholds,
                        CHANNELS,
                        NEURON_BIT_WIDTH
                    ));
                    
                    any_spikes <= has_spikes(generate_spike_vector(
                        membrane_sum,
                        channel_thresholds,
                        CHANNELS,
                        NEURON_BIT_WIDTH
                    ));
                    
                when WRITE_OUTPUT =>
                    -- Write spike event with coordinates to FIFO
                    if fifo_full_i = '0' then
                        fifo_write_en_o <= '1';
                    end if;
                    
                when NEXT_WINDOW =>
                    -- Reset for next window
                    pixel_idx <= 0;
                    membrane_sum <= (others => (others => '0'));
                    any_spikes <= '0';
                    
                    -- Move to next window
                    if window_x < (IMG_WIDTH/POOL_SIZE) - 1 then
                        window_x <= window_x + 1;
                    else
                        window_x <= 0;
                        window_y <= window_y + 1;
                    end if;
                    
                    -- Calculate first pixel coordinates for next window
                    if window_x < (IMG_WIDTH/POOL_SIZE) - 1 then
                        coords := get_pixel_coords(window_x + 1, window_y, 0);
                    else
                        coords := get_pixel_coords(0, window_y + 1, 0);
                    end if;
                    pixel_x <= coords.x;
                    pixel_y <= coords.y;
                    
                when DONE =>
                    busy_reg <= '0';
                    pooling_done_reg <= '1';
            end case;
        end if;
    end process control_process;

end architecture rtl;