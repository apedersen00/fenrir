#include "xparameters.h"
#include "xgpio.h"
#include "xgpiops.h" // For XGpioPs, if used (not explicitly in original snippet's main logic)
#include "fenrir.h"
#include "gesture_data_target_1.h"

// --- UART Includes ---
#include "xuartps.h"      // For Zynq PS UART driver
#include "xil_printf.h"   // For xil_printf

#define SDT // Standalone Device Tree definition from original code

// --- UART Configuration ---
// IMPORTANT: Verify XPAR_XUARTPS_0_DEVICE_ID against your xparameters.h
// For Zynq-7000 it's often XPAR_XUARTPS_0_DEVICE_ID
// For Zynq UltraScale+ MPSoC it might be XPAR_PSU_UART_0_DEVICE_ID or XPAR_XUARTPS_0_DEVICE_ID if using Cadence UART
#define UART_DEVICE_ID      XPAR_XUARTPS_0_BASEADDR
#define UART_BAUD_RATE      115200

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

static XUartPs UartPs; // UART Instance

// function declarations
void write_config(uint32_t baseaddr, const FCConfig *cfg);
uint32_t pack_synapse_loader(const FCConfig *cfg);
uint32_t pack_neuron_loader(const FCConfig *cfg);
uint32_t pack_lif_config(const FCConfig *cfg);
uint32_t pack_neuron_writer(const FCConfig *cfg);
void write_event(XGpio *fifo, uint32_t event);
int InitializeUart(u16 DeviceId); // UART Initialization function

// --- UART Initialization Function ---
int InitializeUart(u16 DeviceId) {
    XUartPs_Config *Config;
    int Status;

    // Look up the configuration in the config table
    Config = XUartPs_LookupConfig(XPAR_XUARTPS_0_BASEADDR);
    if (NULL == Config) {
        xil_printf("XUartPs_LookupConfig failed for DeviceId: %u\r\n", DeviceId);
        return XST_FAILURE;
    }

    // Initialize the UART driver
    Status = XUartPs_CfgInitialize(&UartPs, Config, Config->BaseAddress);
    if (Status != XST_SUCCESS) {
        xil_printf("XUartPs_CfgInitialize failed: %d\r\n", Status);
        return XST_FAILURE;
    }

    // Check self-test (optional but good practice)
    Status = XUartPs_SelfTest(&UartPs);
    if (Status != XST_SUCCESS) {
        xil_printf("XUartPs_SelfTest failed: %d\r\n", Status);
        // Continue anyway, self-test is not always critical for basic operation
    }

    // Set baud rate
    Status = XUartPs_SetBaudRate(&UartPs, UART_BAUD_RATE);
    if (Status != XST_SUCCESS) {
        xil_printf("XUartPs_SetBaudRate failed for %d baud\r\n", UART_BAUD_RATE);
        return XST_FAILURE;
    }

    // Optional: Set RX FIFO threshold (e.g., trigger when 1 byte received)
    // For polling with XUartPs_Recv, this isn't strictly necessary but doesn't hurt.
    // XUartPs_SetFifoThreshold(&UartPs, 1);

    xil_printf("UART Initialized successfully (Device ID: %u) at %d baud (8N1)\r\n", DeviceId, UART_BAUD_RATE);
    return XST_SUCCESS;
}


int main (void)
{
    //int i; // Not used in UART echo section, kept for potential re-use of original loop
    //int sw_check; // Not used in original snippet, can be removed
    int xStatus;

    // Event counters from original code - kept if other parts of the system use them
    // int event_counter_0 = 0;
    // ... (other event_counter declarations)

    // --- Initialize Platform and Peripherals ---
    // In a full Vitis/SDK project, you might have an init_platform() call here.
    // For this example, we directly initialize peripherals.

    xil_printf("\r\n--- FENRIR UART Echo Test Application ---\r\n");

    // --- Initialize GPIOs (from original code) ---
    xStatus = XGpio_Initialize(&fenrir_ctrl, XPAR_FENRIR_CTRL_BASEADDR);
    if (xStatus != XST_SUCCESS) xil_printf("Error initializing fenrir_ctrl\r\n");
    xStatus = XGpio_Initialize(&fifo, XPAR_FIFO_WRITE_BASEADDR);
    if (xStatus != XST_SUCCESS) xil_printf("Error initializing fifo\r\n");
    // ... (Initialize other GPIOs as in your original code) ...
    XGpio_Initialize(&gpio_event_counter_0, XPAR_EVENT_COUNT_0_BASEADDR);
    XGpio_Initialize(&gpio_event_counter_1, XPAR_EVENT_COUNT_1_BASEADDR);
    XGpio_Initialize(&gpio_event_counter_2, XPAR_EVENT_COUNT_2_BASEADDR);
    // ... and so on for other event counters


    // --- Initialize UART ---
    xStatus = InitializeUart(UART_DEVICE_ID);
    if (xStatus != XST_SUCCESS) {
        xil_printf("UART Initialization failed. Halting.\r\n");
        return XST_FAILURE;
    }

    // --- Original FENRIR Configuration (Commented out for simple UART echo test) ---
    /*
    XGpio_DiscreteWrite(&fenrir_ctrl, 1, 0b0000);
    XGpio_DiscreteWrite(&fifo, 1, 0x00000000); // Assuming channel 1 for data

    // XGpio_DiscreteWrite(&fenrir_ctrl, 1, 0b0001);

    FCConfig fc1_cfg = {
        .bits_per_weight = 1, .synapse_layer_offset = 0, .synapse_neurons = 10,
        .neuron_layer_offset = 0, .neuron_neurons = 10,
        .weight_scalar = 10, .beta = 230, .threshold = 67,
        .writer_layer_offset = 0, .writer_neurons = 10
    };
    write_config(XPAR_FENRIR_AXI_0_BASEADDR, &fc1_cfg);
    XGpio_DiscreteWrite(&fenrir_ctrl, 1, 0b0011); // Enable config?
    */

    // --- UART Receive and Echo Loop ---
    u8 UartRxBuffer[3]; // Buffer to hold 3 received bytes for one flattened address
    u32 ReceivedByteCount;
    uint32_t flattened_address;

    xil_printf("Waiting for 3-byte address packets from UART...\r\n");

    while (1)
    {
        ReceivedByteCount = 0;
        // Loop to ensure all 3 bytes for a single address are received
        while(ReceivedByteCount < 3) {
            // XUartPs_Recv is blocking if RX FIFO is empty and no timeout is set.
            // It returns the number of bytes actually received.
            ReceivedByteCount += XUartPs_Recv(&UartPs, &UartRxBuffer[ReceivedByteCount], (3 - ReceivedByteCount));
        }

        if (ReceivedByteCount == 3) {
            // Reconstruct the 24-bit flattened address (MSB first, as sent by host)
            flattened_address = ((uint32_t)UartRxBuffer[0] << 16) |
                                ((uint32_t)UartRxBuffer[1] << 8)  |
                                ((uint32_t)UartRxBuffer[2]);

            // Echo the received value
            // %06X for 24-bit hex, %u for decimal. Add \r for proper terminal display.
            xil_printf("UART RX Addr: 0x%06X (%u)\r\n", flattened_address, flattened_address);

            // --- OPTIONAL: Process the received event ---
            // If you want to use this address with your FENRIR system:
            // 1. Ensure 'fifo' XGpio instance is initialized.
            // 2. Call your write_event function:
            //    write_event(&fifo, flattened_address);
            //    xil_printf("Event 0x%06X sent to FENRIR FIFO.\r\n", flattened_address);
        }

        // --- Original NMNIST Event Sending and GPIO Reading (Commented out) ---
        /*
        if (idx < NMNIST_EVENTS_SIZE) {
            unsigned int event = nmnist_events[idx];
            write_event(&fifo, event);
        }
        idx = idx + 1;

        event_counter_0 = XGpio_DiscreteRead(&gpio_event_counter_0, 1);
        event_counter_1 = XGpio_DiscreteRead(&gpio_event_counter_1, 1);
        event_counter_2 = XGpio_DiscreteRead(&gpio_event_counter_2, 1);
        // ... (read other counters) ...

        // Using xil_printf instead of printf for consistency
        xil_printf("Spike counts: S0 = %10u | S1 = %10u | S2 = %10u\r\n",
               event_counter_0, event_counter_1, event_counter_2);

        for (i=0; i<999999; i++); // Original delay loop
        */
    }

    // Cleanup code (not usually reached in bare-metal while(1) loops)
    // XUartPs_DisableUart(&UartPs); // Example of disabling UART
    xil_printf("--- Application End (should not be reached in while(1)) ---\r\n");
    return 0;
}

// --- Existing functions (write_event, write_config, packer functions) ---
// These are kept as they are part of your system's IP interaction logic.

void write_event(XGpio *fifo_inst, uint32_t event) // Renamed 'fifo' to 'fifo_inst' to avoid conflict if global 'fifo' is used
{
    static uint32_t count = 0; // This static counter might need review if events come from UART
    uint32_t data_to_write;

    // The logic with 'count' seems specific to how 'fenrir' expects events.
    // If events are now single 24-bit addresses, this might need adjustment
    // or the MSB setting logic might be handled by the FENRIR IP itself.
    // For now, keeping original logic.
    if (count == 0) {
        data_to_write = (event & 0x7FFFFFFF) | 0x80000000;
    } else {
        data_to_write = event & 0x7FFFFFFF;
    }

    XGpio_DiscreteWrite(fifo_inst, 1, data_to_write); // Assuming channel 1 for data
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