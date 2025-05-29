#include "xparameters.h"
#include "xgpio.h"
#include "xgpiops.h"
#include "fenrir.h"
#include "nmnist_fpga_data.h"

#define SDT

static XGpio fenrir_ctrl;
static XGpio fifo;
static XGpio gpio_event_counter_0;
static XGpio gpio_event_counter_1;
static XGpio gpio_event_counter_2;
static XGpio gpio_event_counter_3;
static XGpio gpio_event_counter_4;
static XGpio gpio_event_counter_5;
static XGpio gpio_event_counter_6;
static XGpio gpio_event_counter_7;
static XGpio gpio_event_counter_8;
static XGpio gpio_event_counter_9;

// function declerations
void write_config(uint32_t baseaddr, const FCConfig *cfg);
uint32_t pack_synapse_loader(const FCConfig *cfg);
uint32_t pack_neuron_loader(const FCConfig *cfg);
uint32_t pack_lif_config(const FCConfig *cfg);
uint32_t pack_neuron_writer(const FCConfig *cfg);
void write_event(XGpio *fifo, uint32_t event);

int main (void) 
{
    int i;
    int sw_check;
    int xStatus;

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

    XGpioPs_Config *GpioConfigPtr;
	XGpio_Initialize(&fenrir_ctrl, XPAR_FENRIR_CTRL_BASEADDR);
    XGpio_Initialize(&fifo, XPAR_FIFO_WRITE_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_0, XPAR_EVENT_COUNT_0_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_1, XPAR_EVENT_COUNT_1_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_2, XPAR_EVENT_COUNT_2_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_3, XPAR_EVENT_COUNT_3_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_4, XPAR_EVENT_COUNT_4_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_5, XPAR_EVENT_COUNT_5_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_6, XPAR_EVENT_COUNT_6_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_7, XPAR_EVENT_COUNT_7_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_8, XPAR_EVENT_COUNT_8_BASEADDR);
	XGpio_Initialize(&gpio_event_counter_9, XPAR_EVENT_COUNT_9_BASEADDR);

    XGpio_DiscreteWrite(&fenrir_ctrl, 1, 0b0000);
    XGpio_DiscreteWrite(&fifo, 1, 0x00000000);

    FCConfig fc1_cfg = {
        .bits_per_weight = 1,
        .synapse_layer_offset = 0,
        .synapse_neurons = 10,

        .neuron_layer_offset = 0,
        .neuron_neurons = 10,

        .weight_scalar = 10,
        .beta = 230,
        .threshold = 67,

        .writer_layer_offset = 0,
        .writer_neurons = 10
    };

    write_config(XPAR_FENRIR_AXI_0_BASEADDR, &fc1_cfg);
    XGpio_DiscreteWrite(&fenrir_ctrl, 1, 0b0011);

    int idx = 0;
    while (1)
    {
        if (idx < NMNIST_EVENTS_SIZE) {
            unsigned int event = nmnist_events[idx];
            write_event(&fifo, event);
            idx = idx + 1;
        }

        event_counter_0 = XGpio_DiscreteRead(&gpio_event_counter_0, 1);
        event_counter_1 = XGpio_DiscreteRead(&gpio_event_counter_1, 1);
        event_counter_2 = XGpio_DiscreteRead(&gpio_event_counter_2, 1);
        event_counter_3 = XGpio_DiscreteRead(&gpio_event_counter_3, 1);
        event_counter_4 = XGpio_DiscreteRead(&gpio_event_counter_4, 1);
        event_counter_5 = XGpio_DiscreteRead(&gpio_event_counter_5, 1);
        event_counter_6 = XGpio_DiscreteRead(&gpio_event_counter_6, 1);
        event_counter_7 = XGpio_DiscreteRead(&gpio_event_counter_7, 1);
        event_counter_8 = XGpio_DiscreteRead(&gpio_event_counter_8, 1);
        event_counter_9 = XGpio_DiscreteRead(&gpio_event_counter_9, 1);

        printf("Spike counts: S0 = %10u | S1 = %10u | S2 = %10u\n", 
               event_counter_0, event_counter_1, event_counter_2);

        for (i=0; i<9999999; i++);
    }

    return 0;
}

void write_event(XGpio *fifo, uint32_t event)
{
    static uint32_t count = 0;
    uint32_t data_to_write;

    if (count == 0) {
        data_to_write = (event & 0x7FFFFFFF) | 0x80000000;
    } else {
        data_to_write = event & 0x7FFFFFFF;
    }

    XGpio_DiscreteWrite(fifo, 1, data_to_write);
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
    return (cfg->bits_per_weight & 0x3) << 21 |
           (cfg->synapse_layer_offset & 0x7FF) << 10 |
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