library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.conv_pool_pkg.all;

entity convolution is
    generic(
        IMG_WIDTH        : integer := 32;
        IMG_HEIGHT       : integer := 32;
        NEURON_BIT_WIDTH : integer := 9;
        KERNEL_SIZE      : integer := 3;
        CHANNELS_OUT     : integer := 12;
        ADDR_WIDTH       : integer := 10;
        BITS_PER_WEIGHT  : integer := 9
    );
    port(
        -- Standard control signals
        clk              : in  std_logic;
        rst_i            : in  std_logic;
        enable_i         : in  std_logic;
        
        -- Input from event capture
        data_valid_i     : in  std_logic;
        event_coord_i    : in  vector2_t;
        data_consumed_o  : out std_logic;
        
        -- Memory interface - dual port
        mem_read_addr_o  : out std_logic_vector(ADDR_WIDTH-1 downto 0);
        mem_read_en_o    : out std_logic;
        mem_read_data_i  : in  std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
        
        mem_write_addr_o : out std_logic_vector(ADDR_WIDTH-1 downto 0);
        mem_write_en_o   : out std_logic;
        mem_write_data_o : out std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
        
        -- Status signals
        busy_o           : out std_logic;
        
        -- Debug signals
        debug_state_o       : out std_logic_vector(2 downto 0);
        debug_coord_idx_o   : out integer range 0 to KERNEL_SIZE**2;
        debug_calc_idx_o    : out integer range 0 to KERNEL_SIZE**2;
        debug_valid_count_o : out integer range 0 to KERNEL_SIZE**2
    );
end entity convolution;

architecture rtl of convolution is

    type conv_state_t is (
        IDLE,
        CALC_COORDS,
        PIPELINE,
        DONE
    );
    
    signal current_state, next_state : conv_state_t := IDLE;
    
    -- Coordinate management
    type coords_array_t is array (0 to KERNEL_SIZE**2 - 1) of vector2_t;
    signal kernel_coords : coords_array_t := (others => (x => 0, y => 0));
    signal valid_coords_count : integer range 0 to KERNEL_SIZE**2 := 0;
    signal coord_calc_index : integer range 0 to KERNEL_SIZE**2 := 0;
    
    -- Pipeline counters
    signal read_index : integer range 0 to KERNEL_SIZE**2 := 0;
    signal write_index : integer range 0 to KERNEL_SIZE**2 := 0;
    
    -- Kernel weights - stored as LUT arrays (one weight per kernel position per output channel)
    type kernel_weights_t is array (0 to KERNEL_SIZE**2 - 1, 0 to CHANNELS_OUT - 1) 
         of signed(BITS_PER_WEIGHT - 1 downto 0);
    
    -- Initialize with example weights (replace with your actual weights)
    -- For now, using simple pattern: weight = kernel_pos + channel (can be negative)
    function init_kernel_weights return kernel_weights_t is
        variable weights : kernel_weights_t;
    begin
        for pos in 0 to KERNEL_SIZE**2 - 1 loop
            for ch in 0 to CHANNELS_OUT - 1 loop
                -- Create mix of positive and negative weights
                if (pos + ch) mod 3 = 0 then
                    weights(pos, ch) := to_signed(-(pos + ch + 1), BITS_PER_WEIGHT);  -- Negative
                else
                    weights(pos, ch) := to_signed((pos + ch + 1), BITS_PER_WEIGHT);   -- Positive
                end if;
            end loop;
        end loop;
        return weights;
    end function;
    
    signal kernel_weights : kernel_weights_t := init_kernel_weights;
    
    -- Pipeline registers
    signal current_coord : vector2_t := (x => 0, y => 0);
    signal read_data_reg : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0) := (others => '0');
    signal computed_data : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0) := (others => '0');
    signal write_coord : vector2_t := (x => 0, y => 0);
    signal pipeline_kernel_index : integer range 0 to KERNEL_SIZE**2 := 0;
    
    -- Control signals
    signal busy_reg : std_logic := '0';
    signal data_consumed_reg : std_logic := '0';

    -- Helper function to convert state to std_logic_vector
    function state_to_slv(state : conv_state_t) return std_logic_vector is
    begin
        case state is
            when IDLE         => return "000";
            when CALC_COORDS  => return "001";
            when PIPELINE     => return "010";
            when DONE         => return "011";
            when others       => return "111";
        end case;
    end function;

    -- Helper function to convert linear index to kernel offset
    function index_to_kernel_offset(idx : integer) return vector2_t is
        variable offset : vector2_t;
        variable half_kernel : integer := (KERNEL_SIZE - 1) / 2;
    begin
        -- Convert linear index to 2D kernel offset
        offset.x := (idx mod KERNEL_SIZE) - half_kernel;
        offset.y := (idx / KERNEL_SIZE) - half_kernel;
        return offset;
    end function;

    -- Helper function to calculate memory address
    function calc_address(coord : vector2_t) return std_logic_vector is
    begin
        return std_logic_vector(to_unsigned(coord.y * IMG_WIDTH + coord.x, ADDR_WIDTH));
    end function;

    -- Helper function to check if coordinate is within image bounds
    function is_valid_coord(coord : vector2_t) return boolean is
    begin
        return (coord.x >= 0 and coord.x < IMG_WIDTH and 
                coord.y >= 0 and coord.y < IMG_HEIGHT);
    end function;

    -- Event-based convolution computation function (spike input + weight addition)
    function compute_event_convolution(
        membrane_data : std_logic_vector;
        kernel_idx : integer;
        neuron_bits : integer;
        channels : integer;
        weights : kernel_weights_t
    ) return std_logic_vector is
        variable result : std_logic_vector(membrane_data'range);
        variable temp_membrane : signed(neuron_bits - 1 downto 0);
        variable temp_weight : signed(BITS_PER_WEIGHT - 1 downto 0);
        variable temp_sum : signed(neuron_bits downto 0); -- Extra bit for overflow
        variable max_value : signed(neuron_bits - 1 downto 0);
        variable min_value : signed(neuron_bits - 1 downto 0);
    begin
        max_value := to_signed(2**(neuron_bits - 1) - 1, neuron_bits);
        min_value := to_signed(-(2**(neuron_bits - 1)), neuron_bits);
        
        for ch in 0 to channels - 1 loop
            -- Extract current membrane potential for this channel
            temp_membrane := signed(membrane_data((ch + 1) * neuron_bits - 1 downto ch * neuron_bits));
            
            -- Get kernel weight for this position and channel (can be negative)
            temp_weight := weights(kernel_idx, ch);
            
            -- Event-based convolution: membrane + weight (spike input is implicit 1)
            temp_sum := resize(temp_membrane, neuron_bits + 1) + resize(temp_weight, neuron_bits + 1);
            
            -- Saturate on overflow/underflow
            if temp_sum > max_value then
                temp_sum(neuron_bits - 1 downto 0) := max_value;
            elsif temp_sum < min_value then
                temp_sum(neuron_bits - 1 downto 0) := min_value;
            end if;
            
            -- Store result
            result((ch + 1) * neuron_bits - 1 downto ch * neuron_bits) := 
                std_logic_vector(temp_sum(neuron_bits - 1 downto 0));
        end loop;
        
        return result;
    end function;

begin

    -- Output assignments
    busy_o <= busy_reg;
    data_consumed_o <= data_consumed_reg;
    debug_state_o <= state_to_slv(current_state);
    debug_coord_idx_o <= write_index;  -- Show write progress
    debug_calc_idx_o <= coord_calc_index;
    debug_valid_count_o <= valid_coords_count;
    
    -- Memory interface assignments
    mem_read_addr_o <= calc_address(current_coord);
    mem_write_addr_o <= calc_address(write_coord);
    mem_write_data_o <= computed_data;

    -- State register
    state_register : process(clk, rst_i)
    begin
        if rst_i = '1' then
            current_state <= IDLE;
        elsif rising_edge(clk) then
            if enable_i = '1' then
                current_state <= next_state;
            end if;
            -- When disabled, stay in current state (pause functionality)
        end if;
    end process state_register;

    -- Next state logic
    state_machine : process(all)
    begin
        -- Default: stay in current state
        next_state <= current_state;
        
        case current_state is
            when IDLE =>
                if data_valid_i = '1' then
                    next_state <= CALC_COORDS;
                end if;
                
            when CALC_COORDS =>
                -- Wait until all coordinates are calculated
                if coord_calc_index >= KERNEL_SIZE**2 then
                    if valid_coords_count > 0 then
                        next_state <= PIPELINE;
                    else
                        next_state <= DONE;  -- No valid coordinates found
                    end if;
                end if;
                
            when PIPELINE =>
                -- Pipeline is complete when all writes are done
                -- Last write happens when write_index = valid_coords_count
                if write_index >= valid_coords_count then
                    next_state <= DONE;
                end if;
                
            when DONE =>
                next_state <= IDLE;
        end case;
    end process state_machine;

    -- Main control process
    control_process : process(clk, rst_i)
        variable offset : vector2_t;
        variable kernel_coord : vector2_t;
        variable valid_count : integer range 0 to KERNEL_SIZE**2;
    begin
        if rst_i = '1' then
            busy_reg <= '0';
            data_consumed_reg <= '0';
            coord_calc_index <= 0;
            valid_coords_count <= 0;
            read_index <= 0;
            write_index <= 0;
            current_coord <= (x => 0, y => 0);
            write_coord <= (x => 0, y => 0);
            pipeline_kernel_index <= 0;
            mem_read_en_o <= '0';
            mem_write_en_o <= '0';
            kernel_coords <= (others => (x => 0, y => 0));
            
        elsif rising_edge(clk) and enable_i = '1' then
            -- Default: clear one-cycle signals
            data_consumed_reg <= '0';
            mem_read_en_o <= '0';
            mem_write_en_o <= '0';
            
            case current_state is
                when IDLE =>
                    if data_valid_i = '1' then
                        busy_reg <= '1';
                        coord_calc_index <= 0;
                        valid_coords_count <= 0;
                        read_index <= 0;
                        write_index <= 0;
                        kernel_coords <= (others => (x => 0, y => 0));
                    else
                        busy_reg <= '0';
                    end if;
                    
                when CALC_COORDS =>
                    -- Use variable for valid count to avoid timing issues
                    valid_count := valid_coords_count;
                    
                    -- Calculate kernel coordinates one per clock cycle
                    if coord_calc_index < KERNEL_SIZE**2 then
                        -- Get kernel offset for this index
                        offset := index_to_kernel_offset(coord_calc_index);
                        
                        -- Calculate actual coordinate
                        kernel_coord.x := event_coord_i.x + offset.x;
                        kernel_coord.y := event_coord_i.y + offset.y;
                        
                        -- Check if coordinate is valid and store it
                        if is_valid_coord(kernel_coord) then
                            kernel_coords(valid_count) <= kernel_coord;
                            valid_count := valid_count + 1;
                        end if;
                        
                        coord_calc_index <= coord_calc_index + 1;
                    end if;
                    
                    -- Update the signal with the variable value
                    valid_coords_count <= valid_count;
                    
                    -- Initialize pipeline counters when calculation is complete
                    if coord_calc_index = KERNEL_SIZE**2 - 1 then
                        read_index <= 0;
                        write_index <= 0;
                    end if;
                    
                when PIPELINE =>
                    -- Pipelined read/write operations
                    -- Reads are always 1 cycle ahead of writes due to BRAM latency
                    
                    -- Handle reads: issue read for next coordinate
                    if read_index < valid_coords_count then
                        current_coord <= kernel_coords(read_index);
                        mem_read_en_o <= '1';
                        read_index <= read_index + 1;
                    end if;
                    
                    -- Handle writes: write back result from previous read cycle
                    -- Can only write if we have data from a previous read (read_index > write_index)
                    if read_index > write_index and write_index < valid_coords_count then
                        mem_write_en_o <= '1';
                        write_index <= write_index + 1;
                    end if;
                    
                    -- Capture read data and compute result for next cycle's write
                    if mem_read_en_o = '1' then
                        -- Capture data that will be available next cycle
                        read_data_reg <= mem_read_data_i;
                        write_coord <= current_coord;  -- This will be our write address next cycle
                        pipeline_kernel_index <= read_index - 1;  -- Kernel index for the coordinate we're reading
                    end if;
                    
                    -- Compute convolution result for current write
                    if mem_write_en_o = '1' then
                        computed_data <= compute_event_convolution(
                            read_data_reg,
                            pipeline_kernel_index,
                            NEURON_BIT_WIDTH,
                            CHANNELS_OUT,
                            kernel_weights
                        );
                    end if;
                    
                when DONE =>
                    busy_reg <= '0';
                    data_consumed_reg <= '1';
            end case;
        end if;
    end process control_process;

end architecture rtl;