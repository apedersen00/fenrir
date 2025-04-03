library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package conv_system is

    -- CONFIG DOWNSAMPLER
    constant INPUT_WIDTH    : positive := 640;
    constant INPUT_HEIGHT   : positive := 480;
    constant OUTPUT_WIDTH   : positive := 32;
    constant OUTPUT_HEIGHT  : positive := 24;
    constant AXIS_SCALING   : positive := 20; 
    -- log 2 (OUTPUT_WIDTH) = 5 
    constant pixel_address_out_width : positive := 5;
    constant AER_EVENT_WIDTH : positive := 32;
    -- TYPE CONFIGURATIONS
    -- define event type as 2 bit signed integer
    subtype event_t is signed(1 downto 0);
    subtype timeref_clock is std_logic_vector(AER_EVENT_WIDTH - 1 - (2*pixel_address_out_width) downto 0);

    type DUMMY_AER_IN is record
        x : std_logic_vector(15 downto 0);
        y : std_logic_vector(15 downto 0);
        polarity : std_logic_vector(1 downto 0);
    end record;

    type EVENT_R is record
        x : std_logic_vector (pixel_address_out_width - 1 downto 0);
        y : std_logic_vector (pixel_address_out_width - 1 downto 0);
        polarity : event_t;
        timestamp : timeref_clock;
    end record;

    function downsample_input(
        aer_in : in DUMMY_AER_IN;
        timestamp : in timeref_clock
    ) return EVENT_R; 


end package conv_system;

package body conv_system is

    function downsample_input(
        aer_in : in DUMMY_AER_IN;
        timestamp : in timeref_clock
    ) return EVENT_R is

        variable aer_out : EVENT_R;
        variable x : integer;
        variable y : integer;

    begin
        
        x := to_integer(unsigned(aer_in.x)) / AXIS_SCALING;
        y := to_integer(unsigned(aer_in.y)) / AXIS_SCALING;

        aer_out.x := std_logic_vector(to_unsigned(x, pixel_address_out_width));
        aer_out.y := std_logic_vector(to_unsigned(y, pixel_address_out_width));

        aer_out.polarity := signed(aer_in.polarity);
        aer_out.timestamp := timestamp;
        
        return aer_out;
    end;

end package body conv_system;  