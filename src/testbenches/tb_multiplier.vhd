/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: tb_multiplier.vhd
    Description: Testbench for multiplier.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_multiplier is
end tb_multiplier;

architecture sim of tb_multiplier is

    -- width of multiplier inputs
    constant C_VEC_A_WIDTH  : integer := 8;
    constant C_VEC_B_WIDTH  : integer := 8;

    signal vec_a            : std_logic_vector(C_VEC_A_WIDTH - 1 downto 0);
    signal vec_b            : std_logic_vector(C_VEC_B_WIDTH - 1 downto 0);
    signal vec_out          : std_logic_vector(C_VEC_A_WIDTH + C_VEC_B_WIDTH - 1 downto 0);

begin

    uut: entity work.multiplier
        generic map (
            G_VEC_A_WIDTH => C_VEC_A_WIDTH,
            G_VEC_B_WIDTH => C_VEC_B_WIDTH
        )
        port map (
            i_vec_a => vec_a,
            i_vec_b => vec_b,
            o_vec_out => vec_out
        );

    stimulus: process
        variable a, b   : integer;
        variable result : integer;
    begin

        -- 5 * 3 = 15
        a := 5;
        b := 3;
        vec_a <= std_logic_vector(to_signed(a, C_VEC_A_WIDTH));
        vec_b <= std_logic_vector(to_signed(b, C_VEC_B_WIDTH));
        wait for 10 ns;
        result := a * b;
        assert signed(vec_out) = to_signed(result, vec_out'length)
            report "Test failed: " & integer'image(a) & " * " & integer'image(b) & " = " & integer'image(result)
            severity error;

        -- -28 * 3 = -84
        a := -28;
        b := 3;
        vec_a <= std_logic_vector(to_signed(a, C_VEC_A_WIDTH));
        vec_b <= std_logic_vector(to_signed(b, C_VEC_B_WIDTH));
        wait for 10 ns;
        result := a * b;
        assert signed(vec_out) = to_signed(result, vec_out'length)
            report "Test failed: " & integer'image(a) & " * " & integer'image(b) & " = " & integer'image(result)
            severity error;

        -- -20 * -3 = 60
        a := -20;
        b := -3;
        vec_a <= std_logic_vector(to_signed(a, C_VEC_A_WIDTH));
        vec_b <= std_logic_vector(to_signed(b, C_VEC_B_WIDTH));
        wait for 10 ns;
        result := a * b;
        assert signed(vec_out) = to_signed(result, vec_out'length)
            report "Test failed: " & integer'image(a) & " * " & integer'image(b) & " = " & integer'image(result)
            severity error;

        -- 11 * 0 = 0
        a := 1;
        b := 0;
        vec_a <= std_logic_vector(to_signed(a, C_VEC_A_WIDTH));
        vec_b <= std_logic_vector(to_signed(b, C_VEC_B_WIDTH));
        wait for 10 ns;
        result := a * b;
        assert signed(vec_out) = to_signed(result, vec_out'length)
            report "Test failed: " & integer'image(a) & " * " & integer'image(b) & " = " & integer'image(result)
            severity error;
        
        -- done
        report "All tests passed!" severity note;
        wait;

    end process;

end sim;
