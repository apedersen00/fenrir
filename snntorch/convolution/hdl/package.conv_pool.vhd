library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
package conv_pool_pkg is

    type main_state_et is (
        IDLE,
        EVENT_CONV,
        PAUSE,
        POOL,
        CONFIG,
        RESET,
        READ_REQUEST
    );

    type event_tensor_t is record
        x_coord : integer;
        y_coord : integer;
        channel : integer; -- Single channel index (0 to N-1)
    end record;

    type vector2_t is record
        x : integer;
        y : integer;
    end record;
    type conv_thread_t is record
        start_idx : integer;
        end_idx : integer;
        last_result: std_logic_vector(107 downto 0);
        last_address: std_logic_vector(9 downto 0);
        next_address: std_logic_vector(9 downto 0);
    end record;
    -- for testing purposes
    type coords_3x3_t is array (0 to 8) of vector2_t;

    function state_to_slv(state : main_state_et) return std_logic_vector;

    function bus_to_event_tensor(
        event_bus : std_logic_vector;
        bits_per_coord : integer;
        bits_per_channel : integer
    ) return event_tensor_t;

    function int_to_channel_vector(
        channel_int : integer;
        channels_out : integer
    ) return std_logic_vector;

    function channel_vector_to_int(
        channel_vec : std_logic_vector
    ) return integer;

    
    function create_tensor(
        x_coord : integer;
        y_coord : integer;
        channel : integer
    ) return event_tensor_t;

    
    function tensor_to_bus(
        tensor : event_tensor_t;
        bits_per_coord : integer;
        bits_per_channel : integer
    ) return std_logic_vector;

    function convolution_1d(
        kernel_weights : std_logic_vector;
        membrane_potentials : std_logic_vector;
        bits_per_channel : integer;
        channels_out: integer
    ) return std_logic_vector;

    function fast_calc_address(
        coord : vector2_t;
        img_width: integer;
        address_bits : integer := 10 -- for 1024 address space
    ) return std_logic_vector;

end package conv_pool_pkg;

package body conv_pool_pkg is

    function state_to_slv(state : main_state_et) return std_logic_vector is
    begin
        return std_logic_vector(to_unsigned(main_state_et'pos(state), 3));
    end function;

    function bus_to_event_tensor(
        event_bus : std_logic_vector;
        bits_per_coord : integer;
        bits_per_channel : integer
    ) return event_tensor_t is
        variable tensor : event_tensor_t;
    begin
        tensor.x_coord := to_integer(unsigned(event_bus(2 * bits_per_coord - 1 downto bits_per_coord)));
        tensor.y_coord := to_integer(unsigned(event_bus(bits_per_coord - 1 downto 0)));
        tensor.channel := to_integer(unsigned(event_bus(2 * bits_per_coord + bits_per_channel - 1 downto 2 * bits_per_coord)));
        return tensor;
    end function;

    -- Convert integer channel to one-hot vector (if needed)
    function int_to_channel_vector(
        channel_int : integer;
        channels_out : integer
    ) return std_logic_vector is
        variable result : std_logic_vector(channels_out-1 downto 0) := (others => '0');
    begin
        if channel_int < channels_out and channel_int >= 0 then
            result(channel_int) := '1';
        end if;
        return result;
    end function;

    -- Convert channel vector back to integer
    function channel_vector_to_int(
        channel_vec : std_logic_vector
    ) return integer is
    begin
        for i in channel_vec'range loop
            if channel_vec(i) = '1' then
                return i;
            end if;
        end loop;
        return 0; -- default
    end function;

    -- FIXED: Simple tensor creation
    function create_tensor(
        x_coord : integer;
        y_coord : integer;
        channel : integer
    ) return event_tensor_t is
        variable tensor : event_tensor_t;
    begin
        tensor.x_coord := x_coord;
        tensor.y_coord := y_coord;
        tensor.channel := channel;
        return tensor;
    end function;

    -- FIXED: Flexible tensor to bus conversion
    function tensor_to_bus(
        tensor : event_tensor_t;
        bits_per_coord : integer;
        bits_per_channel : integer
    ) return std_logic_vector is
        variable result : std_logic_vector(2 * bits_per_coord + bits_per_channel - 1 downto 0);
    begin
        -- Pack coordinates and channel into bus
        result(bits_per_coord - 1 downto 0) := 
            std_logic_vector(to_unsigned(tensor.y_coord, bits_per_coord));
        result(2 * bits_per_coord - 1 downto bits_per_coord) := 
            std_logic_vector(to_unsigned(tensor.x_coord, bits_per_coord));
        result(2 * bits_per_coord + bits_per_channel - 1 downto 2 * bits_per_coord) := 
            std_logic_vector(to_unsigned(tensor.channel, bits_per_channel));
        return result;
    end function;

    function convolution_1d(
        kernel_weights : std_logic_vector;
        membrane_potentials : std_logic_vector;
        bits_per_channel : integer;
        channels_out: integer
    ) return std_logic_vector is
        variable result : std_logic_vector(channels_out * bits_per_channel - 1 downto 0);
        variable temp_sum : signed(bits_per_channel downto 0); -- Extra bit for overflow detection
        variable temp_kernel : signed(bits_per_channel - 1 downto 0);
        variable temp_membrane : signed(bits_per_channel - 1 downto 0);
        variable max_value : signed(bits_per_channel - 1 downto 0);
        variable min_value : signed(bits_per_channel - 1 downto 0);
    begin
        -- Pre-calculate bounds
        max_value := to_signed(2**(bits_per_channel - 1) - 1, bits_per_channel); -- Max positive for signed
        min_value := to_signed(0, bits_per_channel); -- Minimum (could be negative if needed)

        for i in 0 to channels_out - 1 loop
            -- Extract the kernel and membrane potentials for the current channel
            temp_kernel := signed(kernel_weights(i * bits_per_channel + bits_per_channel - 1 downto i * bits_per_channel));
            temp_membrane := signed(membrane_potentials(i * bits_per_channel + bits_per_channel - 1 downto i * bits_per_channel));
            
            -- Perform the convolution operation with extra bit for overflow detection
            temp_sum := resize(temp_membrane, bits_per_channel + 1) + resize(temp_kernel, bits_per_channel + 1);
            
            -- Check for overflow/underflow and clamp
            if temp_sum > max_value then
                temp_sum(bits_per_channel - 1 downto 0) := max_value;
            elsif temp_sum < min_value then
                temp_sum(bits_per_channel - 1 downto 0) := min_value;
            end if;

            -- Assign the result to the output vector
            result(i * bits_per_channel + bits_per_channel - 1 downto i * bits_per_channel) := 
                std_logic_vector(temp_sum(bits_per_channel - 1 downto 0));

        end loop;

    return result;
    end function convolution_1d;

    function fast_calc_address(
        coord : vector2_t;
        img_width: integer;
        address_bits : integer := 10 -- for 1024 address space
    ) return std_logic_vector is
    begin
    return std_logic_vector(to_unsigned(coord.y * img_width + coord.x, address_bits));
    end function fast_calc_address;

end package body conv_pool_pkg;