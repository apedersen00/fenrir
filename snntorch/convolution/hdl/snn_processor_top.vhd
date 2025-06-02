library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.conv_pool_pkg.all;

entity snn_processor_top is
    generic(
        -- Image and processing parameters
        IMG_WIDTH        : integer := 32;
        IMG_HEIGHT       : integer := 32;
        NEURON_BIT_WIDTH : integer := 9;
        KERNEL_SIZE      : integer := 3;
        POOL_SIZE        : integer := 2;
        CHANNELS_OUT     : integer := 12;
        ADDR_WIDTH       : integer := 10;
        BITS_PER_WEIGHT  : integer := 9;
        RESET_VALUE      : integer := 0;
        
        -- FIFO and coordinate parameters
        BITS_PER_COORD   : integer := 8;
        COORD_WIDTH      : integer := 8;
        
        -- BRAM parameters
        BRAM_DEPTH       : integer := 1024;
        BRAM_FILENAME    : string := ""
    );
    port(
        -- Standard control signals
        clk              : in  std_logic;
        rst_i            : in  std_logic;
        enable_i         : in  std_logic;
        
        -- Input interface (timestep flag)
        timestep_flag_i  : in  std_logic;
        input_ready_o    : out std_logic;
        
        -- Input event FIFO interface
        input_fifo_empty_i : in  std_logic;
        input_fifo_data_i  : in  std_logic_vector(2 * BITS_PER_COORD - 1 downto 0);
        input_fifo_read_o  : out std_logic;
        
        -- Output spike FIFO interface  
        output_fifo_full_i  : in  std_logic;
        output_fifo_data_o  : out std_logic_vector(2 * COORD_WIDTH + CHANNELS_OUT - 1 downto 0);
        output_fifo_write_o : out std_logic;
        
        -- Status and debug signals
        busy_o              : out std_logic;
        processing_active_o : out std_logic;
        
        -- Debug state outputs
        debug_main_state_o  : out std_logic_vector(2 downto 0);
        debug_event_state_o : out std_logic_vector(1 downto 0);
        debug_conv_state_o  : out std_logic_vector(2 downto 0);
        debug_pool_state_o  : out std_logic_vector(2 downto 0);
        
        -- Debug counters
        debug_events_processed_o : out integer range 0 to 65535;
        debug_spikes_generated_o : out integer range 0 to 65535
    );
end entity snn_processor_top;

architecture rtl of snn_processor_top is

    -- Main state machine
    type main_state_t is (
        IDLE,
        CAPTURE_TIMESTEP,
        PROCESS_EVENTS,
        CONVOLUTION_ACTIVE,
        POOLING_ACTIVE,
        PAUSED
    );
    
    signal current_state, next_state : main_state_t := IDLE;
    signal previous_state : main_state_t := IDLE;  -- Remember state before pause
    
    -- Timestep flag buffering
    signal timestep_flag_buffer : std_logic := '0';
    signal timestep_captured : std_logic := '0';
    
    -- Event capture signals
    signal event_data_valid : std_logic;
    signal event_coord : vector2_t;
    signal event_data_consumed : std_logic;
    signal event_capture_enable : std_logic;
    
    -- Convolution signals
    signal conv_enable : std_logic;
    signal conv_busy : std_logic;
    signal conv_mem_read_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal conv_mem_read_en : std_logic;
    signal conv_mem_write_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal conv_mem_write_en : std_logic;
    signal conv_mem_write_data : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
    
    -- Pooling signals
    signal pool_enable : std_logic;
    signal pool_start : std_logic;
    signal pool_busy : std_logic;
    signal pool_done : std_logic;
    signal pool_mem_read_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal pool_mem_read_en : std_logic;
    signal pool_mem_write_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal pool_mem_write_en : std_logic;
    signal pool_mem_write_data : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
    
    -- BRAM arbitration signals
    signal bram_read_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal bram_read_en : std_logic;
    signal bram_read_data : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
    signal bram_write_addr : std_logic_vector(ADDR_WIDTH-1 downto 0);
    signal bram_write_en : std_logic;
    signal bram_write_data : std_logic_vector(CHANNELS_OUT * NEURON_BIT_WIDTH - 1 downto 0);
    
    -- Control and status signals
    signal busy_reg : std_logic := '0';
    signal input_ready_reg : std_logic := '1';
    signal events_processed : integer range 0 to 65535 := 0;
    signal spikes_generated : integer range 0 to 65535 := 0;
    
    -- Internal state control
    signal convolution_done : std_logic := '0';
    signal no_more_events : std_logic := '0';
    signal events_available : std_logic := '0';
    
    -- Helper function to convert main state to std_logic_vector
    function main_state_to_slv(state : main_state_t) return std_logic_vector is
    begin
        case state is
            when IDLE              => return "000";
            when CAPTURE_TIMESTEP  => return "001";
            when PROCESS_EVENTS    => return "010";
            when CONVOLUTION_ACTIVE => return "011";
            when POOLING_ACTIVE    => return "100";
            when PAUSED            => return "111";
            when others            => return "101";
        end case;
    end function;

begin

    -- Output assignments
    busy_o <= busy_reg;
    input_ready_o <= input_ready_reg;
    processing_active_o <= timestep_flag_buffer;
    debug_main_state_o <= main_state_to_slv(current_state);
    debug_events_processed_o <= events_processed;
    debug_spikes_generated_o <= spikes_generated;
    
    -- Event availability detection
    events_available <= not input_fifo_empty_i;
    no_more_events <= input_fifo_empty_i;
    
    -- Convolution completion detection (when busy goes from 1 to 0)
    convolution_completion : process(clk, rst_i)
        variable conv_busy_prev : std_logic := '0';
    begin
        if rst_i = '1' then
            convolution_done <= '0';
            conv_busy_prev := '0';
        elsif rising_edge(clk) then
            convolution_done <= '0';  -- Default: clear the pulse
            
            if conv_busy_prev = '1' and conv_busy = '0' then
                convolution_done <= '1';  -- Generate completion pulse
            end if;
            
            conv_busy_prev := conv_busy;
        end if;
    end process convolution_completion;

    -- State register
    state_register : process(clk, rst_i)
    begin
        if rst_i = '1' then
            current_state <= IDLE;
            previous_state <= IDLE;
        elsif rising_edge(clk) then
            if enable_i = '1' then
                if current_state = PAUSED then
                    -- Resume from previous state
                    current_state <= previous_state;
                else
                    current_state <= next_state;
                end if;
            else
                -- When disabled, transition to PAUSED immediately (except from IDLE)
                if current_state /= IDLE and current_state /= PAUSED then
                    previous_state <= current_state;
                    current_state <= PAUSED;
                end if;
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
                if timestep_flag_i = '1' then
                    next_state <= CAPTURE_TIMESTEP;
                end if;
                
            when CAPTURE_TIMESTEP =>
                -- Always move to PROCESS_EVENTS after capturing timestep
                next_state <= PROCESS_EVENTS;
                
            when PROCESS_EVENTS =>
                -- Process available events through convolution
                if events_available = '1' then
                    next_state <= CONVOLUTION_ACTIVE;
                elsif no_more_events = '1' and not conv_busy = '1' then
                    -- No more events and no active convolution -> start pooling
                    next_state <= POOLING_ACTIVE;
                end if;
                
            when CONVOLUTION_ACTIVE =>
                if convolution_done = '1' then
                    -- Convolution completed, check for more events
                    if events_available = '1' then
                        next_state <= PROCESS_EVENTS;  -- More events to process
                    else
                        next_state <= POOLING_ACTIVE;  -- No more events, start pooling
                    end if;
                end if;
                
            when POOLING_ACTIVE =>
                if pool_done = '1' then
                    -- Pooling completed, ready for next timestep
                    next_state <= IDLE;
                end if;
                
            when PAUSED =>
                -- Resume from where we left off when enabled again
                -- This should be handled by the state_register process
                -- Stay in PAUSED until enable_i = '1'
                next_state <= PAUSED;
        end case;
    end process state_machine;

    -- Main control process
    control_process : process(clk, rst_i)
    begin
        if rst_i = '1' then
            busy_reg <= '0';
            input_ready_reg <= '1';
            timestep_flag_buffer <= '0';
            timestep_captured <= '0';
            events_processed <= 0;
            pool_start <= '0';
            event_data_consumed <= '0';
            
        elsif rising_edge(clk) and enable_i = '1' then
            -- Default: clear one-cycle signals
            pool_start <= '0';
            event_data_consumed <= '0';
            
            -- Handle timestep flag capture (can happen in any state)
            if timestep_flag_i = '1' and current_state = IDLE then
                timestep_flag_buffer <= '1';
                timestep_captured <= '1';
                busy_reg <= '1';
                input_ready_reg <= '0';
            elsif timestep_flag_i = '1' and current_state /= IDLE then
                -- Buffer timestep while processing (overwrite previous if needed)
                timestep_flag_buffer <= '1';
            end if;
            
            case current_state is
                when IDLE =>
                    -- Only clear flags when actually idle and not receiving new timestep
                    if timestep_flag_i = '0' then
                        busy_reg <= '0';
                        input_ready_reg <= '1';
                        timestep_captured <= '0';
                        timestep_flag_buffer <= '0';
                        events_processed <= 0;
                    end if;
                    
                when CAPTURE_TIMESTEP =>
                    -- Ensure we're set up for processing
                    busy_reg <= '1';
                    input_ready_reg <= '0';
                    
                when PROCESS_EVENTS =>
                    -- Enable event capture to accept new events
                    busy_reg <= '1';
                    
                when CONVOLUTION_ACTIVE =>
                    -- Handle convolution completion and data consumption
                    if convolution_done = '1' then
                        event_data_consumed <= '1';
                        events_processed <= events_processed + 1;
                    end if;
                    
                when POOLING_ACTIVE =>
                    -- Start pooling when entering this state
                    if next_state = POOLING_ACTIVE and current_state /= POOLING_ACTIVE then
                        pool_start <= '1';
                    end if;
                    
                when PAUSED =>
                    -- Maintain current state while paused
                    null;
            end case;
        end if;
    end process control_process;

    -- Module enable control
    event_capture_enable <= enable_i when (current_state = PROCESS_EVENTS or current_state = CONVOLUTION_ACTIVE) else '0';
    conv_enable <= enable_i when (current_state = CONVOLUTION_ACTIVE) else '0';
    pool_enable <= enable_i when (current_state = POOLING_ACTIVE) else '0';

    -- BRAM Arbitration: Convolution has priority, then pooling
    bram_arbitration : process(all)
    begin
        -- Default: no access
        bram_read_addr <= (others => '0');
        bram_read_en <= '0';
        bram_write_addr <= (others => '0');
        bram_write_en <= '0';
        bram_write_data <= (others => '0');
        
        if current_state = CONVOLUTION_ACTIVE then
            -- Convolution has access to BRAM
            bram_read_addr <= conv_mem_read_addr;
            bram_read_en <= conv_mem_read_en;
            bram_write_addr <= conv_mem_write_addr;
            bram_write_en <= conv_mem_write_en;
            bram_write_data <= conv_mem_write_data;
            
        elsif current_state = POOLING_ACTIVE then
            -- Pooling has access to BRAM
            bram_read_addr <= pool_mem_read_addr;
            bram_read_en <= pool_mem_read_en;
            bram_write_addr <= pool_mem_write_addr;
            bram_write_en <= pool_mem_write_en;
            bram_write_data <= pool_mem_write_data;
        end if;
    end process bram_arbitration;

    -- Spike counting process
    spike_counter : process(clk, rst_i)
    begin
        if rst_i = '1' then
            spikes_generated <= 0;
        elsif rising_edge(clk) then
            if output_fifo_write_o = '1' then
                spikes_generated <= spikes_generated + 1;
            end if;
        end if;
    end process spike_counter;

    -- Event Capture Module
    event_capture_inst : entity work.event_capture
    generic map (
        BITS_PER_COORD => BITS_PER_COORD,
        IMG_WIDTH      => IMG_WIDTH,
        IMG_HEIGHT     => IMG_HEIGHT
    )
    port map (
        clk             => clk,
        rst_i           => rst_i,
        enable_i        => event_capture_enable,
        fifo_empty_i    => input_fifo_empty_i,
        fifo_bus_i      => input_fifo_data_i,
        fifo_read_o     => input_fifo_read_o,
        data_valid_o    => event_data_valid,
        data_out_o      => event_coord,
        data_consumed_i => event_data_consumed,
        debug_state_o   => debug_event_state_o
    );

    -- Convolution Module
    convolution_inst : entity work.convolution_configurable
    generic map (
        IMG_WIDTH        => IMG_WIDTH,
        IMG_HEIGHT       => IMG_HEIGHT,
        NEURON_BIT_WIDTH => NEURON_BIT_WIDTH,
        KERNEL_SIZE      => KERNEL_SIZE,
        CHANNELS_OUT     => CHANNELS_OUT,
        ADDR_WIDTH       => ADDR_WIDTH,
        BITS_PER_WEIGHT  => BITS_PER_WEIGHT
    )
    port map (
        clk              => clk,
        rst_i            => rst_i,
        enable_i         => conv_enable,
        data_valid_i     => event_data_valid,
        event_coord_i    => event_coord,
        data_consumed_o  => open,  -- We handle this in the top level
        mem_read_addr_o  => conv_mem_read_addr,
        mem_read_en_o    => conv_mem_read_en,
        mem_read_data_i  => bram_read_data,
        mem_write_addr_o => conv_mem_write_addr,
        mem_write_en_o   => conv_mem_write_en,
        mem_write_data_o => conv_mem_write_data,
        busy_o           => conv_busy,
        debug_state_o    => debug_conv_state_o,
        debug_coord_idx_o => open,
        debug_calc_idx_o => open,
        debug_valid_count_o => open
    );

    -- Pooling Module
    pooling_inst : entity work.pooling
    generic map (
        IMG_WIDTH        => IMG_WIDTH,
        IMG_HEIGHT       => IMG_HEIGHT,
        NEURON_BIT_WIDTH => NEURON_BIT_WIDTH,
        POOL_SIZE        => POOL_SIZE,
        CHANNELS         => CHANNELS_OUT,
        ADDR_WIDTH       => ADDR_WIDTH,
        RESET_VALUE      => RESET_VALUE,
        COORD_WIDTH      => COORD_WIDTH
    )
    port map (
        clk              => clk,
        rst_i            => rst_i,
        enable_i         => pool_enable,
        start_pooling_i  => pool_start,
        pooling_done_o   => pool_done,
        mem_read_addr_o  => pool_mem_read_addr,
        mem_read_en_o    => pool_mem_read_en,
        mem_read_data_i  => bram_read_data,
        mem_write_addr_o => pool_mem_write_addr,
        mem_write_en_o   => pool_mem_write_en,
        mem_write_data_o => pool_mem_write_data,
        fifo_write_en_o  => output_fifo_write_o,
        fifo_write_data_o => output_fifo_data_o,
        fifo_full_i      => output_fifo_full_i,
        busy_o           => pool_busy,
        debug_state_o    => debug_pool_state_o,
        debug_window_x_o => open,
        debug_window_y_o => open,
        debug_pixel_idx_o => open,
        debug_spike_count_o => open,
        debug_accum_ch0_o => open,
        debug_accum_ch1_o => open
    );

    -- Dual Port BRAM
    bram_inst : entity work.DUAL_PORT_BRAM
    generic map (
        DEPTH    => BRAM_DEPTH,
        WIDTH    => CHANNELS_OUT * NEURON_BIT_WIDTH,
        FILENAME => BRAM_FILENAME
    )
    port map (
        i_we     => bram_write_en,
        i_waddr  => bram_write_addr,
        i_wdata  => bram_write_data,
        i_re     => bram_read_en,
        i_raddr  => bram_read_addr,
        o_rdata  => bram_read_data,
        i_clk    => clk
    );

end architecture rtl;