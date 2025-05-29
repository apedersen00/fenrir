#include "xparameters.h"
#include "xgpio.h"
#include "xgpiops.h"

#define SDT

static XGpioPs psGpioInstancePtr;
static XGpio switches;
static XGpio fenrir_conf;
static XGpio spike_0;
static XGpio spike_1;
static XGpio spike_2;
static XGpio fifo;

int main (void) 
{
    int i;
    int sw_check;
    int xStatus;

    int spike_0_count = 0;
    int spike_1_count = 0;
    int spike_2_count = 0;

    XGpioPs_Config *GpioConfigPtr;
	XGpio_Initialize(&switches, XPAR_SWITCHES_BASEADDR);
	XGpio_Initialize(&fenrir_conf, XPAR_FENRIR_CTRL_BASEADDR);
	XGpio_Initialize(&spike_0, XPAR_SPIKE_0_BASEADDR);
	XGpio_Initialize(&spike_1, XPAR_SPIKE_1_BASEADDR);
	XGpio_Initialize(&spike_2, XPAR_SPIKE_2_BASEADDR);
    XGpio_Initialize(&spike_2, XPAR_SPIKE_2_BASEADDR);
    XGpio_Initialize(&fifo, XPAR_FIFO_WRITE_BASEADDR);

	// PS GPIO Intialization
	GpioConfigPtr = XGpioPs_LookupConfig(XPAR_XGPIOPS_0_BASEADDR);
	if (GpioConfigPtr == NULL)
    {
        return XST_FAILURE;
    }

	xStatus = XGpioPs_CfgInitialize(&psGpioInstancePtr, GpioConfigPtr, GpioConfigPtr->BaseAddr);
	if (XST_SUCCESS != xStatus)
    {
        return XST_FAILURE;
    }

	//EMIO PIN Setting to Input port
	XGpioPs_SetDirectionPin(&psGpioInstancePtr, 54, 0);
	XGpioPs_SetOutputEnablePin(&psGpioInstancePtr, 54,0);

    XGpio_DiscreteWrite(&fifo, 1, 0x00000000);

    int count = 0;
    while (1)
    {
        switch (count) {
            case 0:
                XGpio_DiscreteWrite(&fifo, 1, 0x800000AA);
                break;
            case 1:
                XGpio_DiscreteWrite(&fifo, 1, 0x000000BB);
                break;
            case 2:
                XGpio_DiscreteWrite(&fifo, 1, 0x800000CC);
                break;
            case 3:
                XGpio_DiscreteWrite(&fifo, 1, 0x000000DD);
                break;
        }
        count = (count + 1) % 4;

        spike_0_count = XGpio_DiscreteRead(&spike_0, 1);
        spike_1_count = XGpio_DiscreteRead(&spike_1, 1);
        spike_2_count = XGpio_DiscreteRead(&spike_2, 1);

        sw_check = XGpio_DiscreteRead(&switches, 1);
        XGpio_DiscreteWrite(&fenrir_conf, 1, sw_check);

        printf("Spike counts: S0 = %10u | S1 = %10u | S2 = %10u\n", 
               spike_0_count, spike_1_count, spike_2_count);

        for (i=0; i<9999999; i++);
    }

    return 0;
}
