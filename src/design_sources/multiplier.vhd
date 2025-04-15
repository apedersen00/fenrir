/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: multiplier.vhd
    Description: Signed multiplier block.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity multiplier is
    generic (
        G_VEC_A_WIDTH   : integer;
        G_VEC_B_WIDTH   : integer
    );
    port (
        i_vec_a         : in std_logic_vector(G_VEC_A_WIDTH - 1 downto 0);
        i_vec_b         : in std_logic_vector(G_VEC_B_WIDTH - 1 downto 0);
        o_vec_out       : out std_logic_vector(G_VEC_A_WIDTH + G_VEC_B_WIDTH - 1 downto 0)
    );
end multiplier;

architecture Behavioral of multiplier is
    signal vec_out      : std_logic_vector(G_VEC_A_WIDTH + G_VEC_B_WIDTH - 1 downto 0);
begin

    process (i_vec_a, i_vec_b)
    begin
        vec_out         <= std_logic_vector(signed(i_vec_a) * signed(i_vec_b));
    end process;

    o_vec_out       <= vec_out;

end Behavioral;
