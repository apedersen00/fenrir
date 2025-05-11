---------------------------------------------------------------------------------------------------
--  Aarhus University (AU, Denmark)
---------------------------------------------------------------------------------------------------
--
--  File: fifo.vhd
--  Description: Ring-buffer style first-in-first-out (FIFO) utilizing dual-port BRAM.
--
--  Author(s):
--      - A. Pedersen, Aarhus University
--      - A. Cherencq, Aarhus University
--
--  Reference(s):
--      - https://vhdlwhiz.com/ring-buffer-fifo/
--
---------------------------------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

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
        o_rvalid        : out std_logic;                                -- read valid
        o_rdata         : out std_logic_vector(WIDTH - 1 downto 0);     -- read data

        -- flags
        o_empty         : out std_logic;                                -- FIFO empty flag
        o_empty_next    : out std_logic;                                -- FIFO empty next flag
        o_full          : out std_logic;                                -- FIFO full flag
        o_full_next     : out std_logic;                                -- FIFO full next flag

        -- number of elements in FIFO
        o_fill_count    : out std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);

        -- auxiliary
        i_clk           : in std_logic;
        i_rst           : in std_logic;
        o_fault         : out std_logic
    );
end BRAM_FIFO;

architecture Behavioral of BRAM_FIFO is

    subtype index_t is integer range DEPTH - 1 downto 0;
    signal head         : index_t;
    signal tail         : index_t;

    signal empty        : std_logic;
    signal full         : std_logic;
    signal fill_count   : integer range DEPTH - 1 downto 0;

    -- BRAM interface signals
    signal waddr     : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);
    signal raddr     : std_logic_vector(integer(ceil(log2(real(DEPTH))))-1 downto 0);

    procedure incr(signal index : inout index_t) is
    begin
        if index = index_t'high then
            index <= index_t'low;
        else
            index <= index + 1;
        end if;
    end procedure;

begin

    -- copy internal signals to output ports
    o_empty         <= empty;
    o_full          <= full;
    o_fill_count    <= std_logic_vector(to_unsigned(fill_count, o_fill_count'length));

    -- FIFO flags
    empty           <= '1' when fill_count = 0 else '0';
    o_empty_next    <= '1' when fill_count <= 1 else '0';
    full            <= '1' when fill_count >= DEPTH - 1 else '0';
    o_full_next     <= '1' when fill_count >= DEPTH - 2 else '0';

    fifo_bram: entity work.DUAL_PORT_BRAM
        generic map (
            DEPTH => DEPTH,
            WIDTH => WIDTH
        )
        port map (
            i_we    => i_we,
            i_waddr => waddr,
            i_wdata => i_wdata,
            i_re    => i_re,
            i_raddr => raddr,
            o_rdata => o_rdata,
            i_clk   => i_clk
        );

    PROC_HEAD : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                head <= 0;
            else
                if i_we = '1' and full = '0' then
                    incr(head);
                end if;
            end if;
        end if;
    end process;

    PROC_TAIL : process(i_clk)
    begin
        if rising_edge(i_clk) then
            if i_rst = '1' then
                tail <= 0;
            else
                o_rvalid <= '0';
                if i_re = '1' and empty = '0' then
                    incr(tail);
                    o_rvalid <= '1';
                end if;
            end if;
        end if;
    end process;

    PROC_BRAM : process(i_clk)
    begin
        if rising_edge(i_clk) then
            waddr <= std_logic_vector(to_unsigned(head, waddr'length));
            raddr <= std_logic_vector(to_unsigned(tail, raddr'length));
        end if;
    end process;

    PROC_COUNT : process(head, tail)
    begin
        if head < tail then
            fill_count <= head - tail + DEPTH;
        else
            fill_count <= head - tail;
        end if;
    end process;

end Behavioral;
