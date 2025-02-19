---
title: Neuron Memory
parent: Notes
---
```vhdl
entity neuron_memory is
    port (
        clk : in std_logic;
        rst : in std_logic;
        neuron_address : in std_logic_vector(3 downto 0);
        
        param_leak_str : out std_logic_vector(6 downto 0);
        param_thr : out std_logic_vector(11 downto 0);

        state_core : out std_logic_vector(11 downto 0);
        state_core_next : in std_logic_vector(11 downto 0);

        we : in std_logic;
        neuron_in : in std_logic_vector(31 downto 0)
    );
end neuron_memory;

```
