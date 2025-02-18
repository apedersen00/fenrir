---
title: LIF VHDL
parent: Notes
---

# Leaky Integrate-and-Fire VHDL Implementation

The `lif_neuron` entity has 6 inputs and 2 outputs.

```vhdl
entity lif_neuron is
    port (
        -- leakage stength parameter
        param_leak_str  : in std_logic_vector(6 downto 0);
        -- neuron firing threshold parameter
        param_thr       : in std_logic_vector(11 downto 0);

        -- core neuron state from SRAM
        state_core      : in std_logic_vector(11 downto 0);
        -- next core neuron state to SRAM
        state_core_next : out std_logic_vector(11 downto 0);

        -- synaptic weight
        syn_weight      : in std_logic_vector(3 downto 0);
        -- synaptic event trigger
        syn_event       : in std_logic;
        -- time reference event trigger
        time_ref        : in std_logic;

        -- neuron spike event output
        spike_out       : out std_logic
    );
end lif_neuron;
```

The neuron has two events:

- `event_leak <= syn_event and time_ref;`
    - A synaptic event is triggered in conjunction with a time reference event. Triggers a leakage event, lowering the membrane potential.
- `event_syn <= syn_event and (not time_ref);`
    - A synaptic event is triggered with no time reference event. This increases the membrane potential.

A spike is generated if the next membrane potential `state_core_next` exceeds the potential threshold `param_thr`. The membrane potential is signed. If it is negative no spike is generated.

The next membrane potential `state_core_next` is reset to 0 when a spike occurs.

The synaptic weight `syn_weight` is defined as a 4-bit vector. To allow arithmetic with 12-bit values we **sign-extend** it by concatenating it with `syn_weight_ext`.

## Process

A `process` is defined that is sensitive to all. It *processes* the `event_leak` and `event_syn`.

### `event_leak`

If the 11th bit of `state_core` is 1 the membrane potential is negative and the potential is updated with a positive leakage `state_leakp` so it moves towards zero. When the membrane potential is positive it is updated with `state_leakn`.

### `event_syn`

The membrane potential is updated with `state_syn`.

## Arithmetic

The leakage values `state_leakn` and `state_leakp` are calculated by subtracting and adding the leakage parameter from the state core value. In addition the values are capped if they change sign.

The vector `state_syn` is the new value we update `state_core` with. It is equal the sum of `state_core` and `syn_weight_ext`.