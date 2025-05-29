/******************************************************************************
 *  Aarhus University (AU, Denmark)
 ******************************************************************************
 *  File: fenrir.h
 *  Description: Header file for FENRIR.
 *
 *  Author(s):
 *      - A. Pedersen, Aarhus University
 *      - A. Cherencq, Aarhus University
******************************************************************************/

#ifndef FENRIR_H_
#define FENRIR_H_

#include <stdint.h>

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

#endif // FENRIR_H_
