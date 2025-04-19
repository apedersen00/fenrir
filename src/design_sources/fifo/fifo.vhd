/*
---------------------------------------------------------------------------------------------------
    Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------

    File: fifo.vhd
    Description: Ring-buffer style first-in-first-out (FIFO) utilizing dual-port BRAM.

    Author(s):
        - A. Pedersen, Aarhus University
        - A. Cherencq, Aarhus University

---------------------------------------------------------------------------------------------------
*/

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity BRAM_FIFO is
    generic (
        DEPTH : integer;    -- FIFO depth
        WIDTH : integer     -- FIFO width
    );
    port (
        -- write interface
        i_we            : in std_logic;                                 -- write enable
        i_wdata         : in std_logic_vector(WIDTH - 1 downto 0);      -- write data

        -- read interface
        i_re            : in std_logic;                                 -- read enable
        o_rdata         : out std_logic_vector(WIDTH - 1 downto 0);     -- read data

        -- flags
        o_empty         : out std_logic;                                -- FIFO empty flag
        o_empty_next    : out std_logic;                                -- FIFO empty next flag
        o_full          : out std_logic;                                -- FIFO full flag
        o_full_next     : out std_logic;                                -- FIFO full next flag

        -- auxiliary
        i_clk           : in std_logic;
        o_fault         : out std_logic
    );
end BRAM_FIFO;

architecture Behavioral of BRAM_FIFO is

-- INSERT FIFO HERE:

-- DUAL-PORT BRAM

-- WRITE_POINTER

-- READ_POINTER

-- FIFO_LOGIC

begin



end Behavioral;
