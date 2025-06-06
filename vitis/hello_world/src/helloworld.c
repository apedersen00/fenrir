#include "xparameters.h"
#include "fenrir.h"
#include "gesture_data_target_1.h"
#include "xil_io.h"

#define SDT

#define FENRIR_BASE             XPAR_FENRIR_AXI_0_BASEADDR

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

// function declerations
void write_config(uint32_t baseaddr, const FCConfig *cfg);
uint32_t pack_synapse_loader(const FCConfig *cfg);
uint32_t pack_neuron_loader(const FCConfig *cfg);
uint32_t pack_lif_config(const FCConfig *cfg);
uint32_t pack_neuron_writer(const FCConfig *cfg);
void write_event(uint32_t event);

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

int main (void) 
{
    int i;

    int event_counter_0 = 0;
    int event_counter_1 = 0;
    int event_counter_2 = 0;
    int event_counter_3 = 0;
    int event_counter_4 = 0;
    int event_counter_5 = 0;
    int event_counter_6 = 0;
    int event_counter_7 = 0;
    int event_counter_8 = 0;
    int event_counter_9 = 0;
    int event_counter_10 = 0;

    Xil_Out32(FENRIR_BASE + FENRIR_CTRL_OFFSET, 0x00000000);
    Xil_Out32(FENRIR_BASE + FENRIR_WRITE_OFFSET, 0x00000000);

    write_config(FENRIR_BASE + FC1_SYNLDR_OFFSET, &fc1_cfg);
    write_config(FENRIR_BASE + FC2_SYNLDR_OFFSET, &fc2_cfg);

    Xil_Out32(FENRIR_BASE + FENRIR_CTRL_OFFSET, 0x00000003);

    int idx = 0;
    while (1)
    {
        if (idx < NMNIST_EVENTS_SIZE) {
            unsigned int event = nmnist_events[idx];
            write_event(event);
        }
        idx = idx + 1;

        event_counter_0 = Xil_In32(FENRIR_BASE + CLASS_COUNT_0_OFFSET);
        event_counter_1 = Xil_In32(FENRIR_BASE + CLASS_COUNT_1_OFFSET);
        event_counter_2 = Xil_In32(FENRIR_BASE + CLASS_COUNT_2_OFFSET);
        event_counter_3 = Xil_In32(FENRIR_BASE + CLASS_COUNT_3_OFFSET);
        event_counter_4 = Xil_In32(FENRIR_BASE + CLASS_COUNT_4_OFFSET);
        event_counter_5 = Xil_In32(FENRIR_BASE + CLASS_COUNT_5_OFFSET);
        event_counter_6 = Xil_In32(FENRIR_BASE + CLASS_COUNT_6_OFFSET);
        event_counter_7 = Xil_In32(FENRIR_BASE + CLASS_COUNT_7_OFFSET);
        event_counter_8 = Xil_In32(FENRIR_BASE + CLASS_COUNT_8_OFFSET);
        event_counter_9 = Xil_In32(FENRIR_BASE + CLASS_COUNT_9_OFFSET);
        event_counter_10 = Xil_In32(FENRIR_BASE + CLASS_COUNT_10_OFFSET);

        if (idx % 1000 == 0) {
            printf("Spike counts:\n");
            printf("  S0 = %10u | S1 = %10u | S2 = %10u | S3 = %10u | S4 = %10u\n",
                    event_counter_0, event_counter_1, event_counter_2, event_counter_3, event_counter_4);
            printf("  S5 = %10u | S6 = %10u | S7 = %10u | S8 = %10u | S9 = %10u\n",
                    event_counter_5, event_counter_6, event_counter_7, event_counter_8, event_counter_9);
        }

        for (i=0; i<99999; i++);
    }

    return 0;
}

void print_binary(uint32_t value) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (value >> i) & 1);
        if (i % 4 == 0) printf(" "); // optional: space every 4 bits for readability
    }
    printf("\n");
}

void write_event(uint32_t event)
{
    static uint32_t count = 0;
    uint32_t data_to_write;

    if (count == 0) {
        data_to_write = (event & 0x7FFFFFFF) | 0x80000000;
    } else {
        data_to_write = event & 0x7FFFFFFF;
    }

    // print_binary(data_to_write);
    Xil_Out32(FENRIR_BASE + FENRIR_WRITE_OFFSET, data_to_write);
    count = (count + 1) % 2;
}

void write_config(uint32_t baseaddr, const FCConfig *cfg)
{
    Xil_Out32(baseaddr + 0, pack_synapse_loader(cfg));
    Xil_Out32(baseaddr + 4, pack_neuron_loader(cfg));
    Xil_Out32(baseaddr + 8, pack_lif_config(cfg));
    Xil_Out32(baseaddr + 12, pack_neuron_writer(cfg));
}

uint32_t pack_synapse_loader(const FCConfig *cfg)
{
    return (cfg->bits_per_weight & 0x3) << 22 |
           (cfg->synapse_layer_offset & 0x7FF) << 11 |
           (cfg->synapse_neurons & 0x7FF);
}

uint32_t pack_neuron_loader(const FCConfig *cfg)
{
    return (cfg->neuron_layer_offset & 0x7FF) << 11 |
           (cfg->neuron_neurons & 0x7FF);
}

uint32_t pack_lif_config(const FCConfig *cfg)
{
    return (cfg->weight_scalar & 0xFF) << 24 |
           (cfg->beta & 0xFFF) << 12 |
           (cfg->threshold & 0xFFF);
}

uint32_t pack_neuron_writer(const FCConfig *cfg)
{
    return (cfg->writer_layer_offset & 0x7FF) << 11 |
           (cfg->writer_neurons & 0x7FF);
}