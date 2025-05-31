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
        RESET
    );

    type event_tensor_t is record
        x_coord : integer;
        y_coord : integer;
        channel : integer; -- Single channel index (0 to N-1)
    end record;

    function state_to_slv(state : main_state_et) return std_logic_vector;

    function bus_to_event_tensor(
        event_bus : std_logic_vector;
        bits_per_coord : integer;
        bits_per_channel : integer
    ) return event_tensor_t;

    -- Helper functions for channel manipulation
    function int_to_channel_vector(
        channel_int : integer;
        channels_out : integer
    ) return std_logic_vector;

    function channel_vector_to_int(
        channel_vec : std_logic_vector
    ) return integer;

    -- FIXED: Easy tensor creation for simulation
    function create_tensor(
        x_coord : integer;
        y_coord : integer;
        channel : integer
    ) return event_tensor_t;

    -- FIXED: Flexible tensor to bus conversion
    function tensor_to_bus(
        tensor : event_tensor_t;
        bits_per_coord : integer;
        bits_per_channel : integer
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

end package body conv_pool_pkg;