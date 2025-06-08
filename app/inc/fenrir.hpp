/******************************************************************************
 *  Aarhus University (AU, Denmark)
 ******************************************************************************
 *  File: fenrir.hpp
 *  Description: Header file for fenrir.cpp
 *
 *  Author(s):
 *      - A. Pedersen, Aarhus University
 *      - A. Cherencq, Aarhus University
******************************************************************************/

#ifndef FENRIR_H_
#define FENRIR_H_

/** ATTENTION!
 *  This address MUST be correct. Double check the device-tree.
 */
#define FENRIR_BASE_ADDR        0x43c00000

// AXI ouptuts
#define FC1_SYNLDR_OFFSET       0
#define FC1_NRNLDR_OFFSET       4
#define FC1_LIF_OFFSET          8
#define FC1_NRNWRT_OFFSET       12
#define FC2_SYNLDR_OFFSET       16
#define FC2_NRNLDR_OFFSET       20
#define FC2_LIF_OFFSET          24
#define FC2_NRNWRT_OFFSET       28
#define FENRIR_CTRL_OFFSET      32
#define FENRIR_WRITE_OFFSET     36

// AXI inputs
#define FLAGS_OFFSET            40
#define CLASS_COUNT_0_OFFSET    44
#define CLASS_COUNT_1_OFFSET    48
#define CLASS_COUNT_2_OFFSET    52
#define CLASS_COUNT_3_OFFSET    56
#define CLASS_COUNT_4_OFFSET    60
#define CLASS_COUNT_5_OFFSET    64
#define CLASS_COUNT_6_OFFSET    68
#define CLASS_COUNT_7_OFFSET    72
#define CLASS_COUNT_8_OFFSET    76
#define CLASS_COUNT_9_OFFSET    80
#define CLASS_COUNT_10_OFFSET   84

typedef struct {
    // Synapse Loader
    uint8_t bits_per_weight;        // 2 bits
    uint16_t synapse_layer_offset;  // 11 bits
    uint16_t synapse_neurons;       // 11 bits

    // Neuron Loader
    uint16_t neuron_layer_offset;   // 11 bits
    uint16_t neuron_neurons;        // 11 bits

    // LIF Config
    uint8_t weight_scalar;          // 8 bits
    uint16_t beta;                  // 12 bits
    uint16_t threshold;             // 12 bits

    // Neuron Writer
    uint16_t writer_layer_offset;   // 11 bits
    uint16_t writer_neurons;        // 11 bits
} FCConfig;

FCConfig fc1_cfg = {
    .bits_per_weight = 1,
    .synapse_layer_offset = 0,
    .synapse_neurons = 64,

    .neuron_layer_offset = 0,
    .neuron_neurons = 64,

    .weight_scalar = 10,
    .beta = 18,
    .threshold = 390,

    .writer_layer_offset = 0,
    .writer_neurons = 64
};

FCConfig fc2_cfg = {
    .bits_per_weight = 1,
    .synapse_layer_offset = 0,
    .synapse_neurons = 11,

    .neuron_layer_offset = 0,
    .neuron_neurons = 11,

    .weight_scalar = 10,
    .beta = 16,
    .threshold = 477,

    .writer_layer_offset = 0,
    .writer_neurons = 11
};

/**
 * @struct FenrirMemoryMap
 * @brief Struct for accessing FENRIR
 */
struct FenrirMemoryMap {
    // FC1 Config Registers (Offsets 0x00 - 0x0C)
    struct {
        struct { // Offset 0x00
            uint32_t synapse_neurons      : 11;
            uint32_t synapse_layer_offset : 11;
            uint32_t bits_per_weight      : 2;
            uint32_t _reserved0           : 8;
        } synapse_loader;
        struct { // Offset 0x04
            uint32_t neuron_neurons       : 11;
            uint32_t neuron_layer_offset  : 11;
            uint32_t _reserved1           : 10;
        } neuron_loader;
        struct { // Offset 0x08
            uint32_t threshold            : 12;
            uint32_t beta                 : 12;
            uint32_t weight_scalar        : 8;
        } lif_config;
        struct { // Offset 0x0C
            uint32_t writer_neurons       : 11;
            uint32_t writer_layer_offset  : 11;
            uint32_t _reserved3           : 10;
        } neuron_writer;
    } fc1_config;

    // FC2 Config Registers (Offsets 0x10 - 0x1C)
    struct {
        struct { // Offset 0x10
            uint32_t synapse_neurons      : 11;
            uint32_t synapse_layer_offset : 11;
            uint32_t bits_per_weight      : 2;
            uint32_t _reserved0           : 8;
        } synapse_loader;
        struct { // Offset 0x14
            uint32_t neuron_neurons       : 11;
            uint32_t neuron_layer_offset  : 11;
            uint32_t _reserved1           : 10;
        } neuron_loader;
        struct { // Offset 0x18
            uint32_t threshold            : 12;
            uint32_t beta                 : 12;
            uint32_t weight_scalar        : 8;
        } lif_config;
        struct { // Offset 0x1C
            uint32_t writer_neurons       : 11;
            uint32_t writer_layer_offset  : 11;
            uint32_t _reserved3           : 10;
        } neuron_writer;
    } fc2_config;

    // Control and Status Registers (Offsets 0x20 onwards)
    uint32_t control;           // Offset 0x20 (32)
    uint32_t write;             // Offset 0x24 (36)
    uint32_t flags;             // Offset 0x28 (40)
    uint32_t class_counts[11];  // Offsets 0x2C (44) to 0x54 (84)
};

#endif // FENRIR_H_
