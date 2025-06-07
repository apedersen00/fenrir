/******************************************************************************
 *  Aarhus University (AU, Denmark)
 ******************************************************************************
 *  File: fenrir.cpp
 *  Description: Main target applicaiton for interfacing with DVS and FENRIR.
 *
 *  Author(s):
 *      - A. Pedersen, Aarhus University
 *      - A. Cherencq, Aarhus University
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <atomic>
#include <csignal>
#include <chrono>

#include <libcaercpp/devices/dvxplorer.hpp>

#include "fenrir.hpp"

#define MAP_SIZE 4096
#define TARGET_WIDTH 240 // If changing this, also update DVX_DVS_CHIP config

using namespace std;
using namespace std::chrono;

static int initShutdownHandler();
static void globalShutdownSignalHandler(int signal);
static void usbShutdownHandler(void *ptr);
static void writeEvent(volatile FenrirMemoryMap *regs, uint32_t event);

static atomic_bool globalShutdown(false);


int main(void) {
    if (initShutdownHandler() == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    // Open /dev/mem for write/read access
    int mem_fd;
    if ((mem_fd = open("/dev/mem", O_RDWR | O_SYNC)) < 0) {
        perror("Failed to open /dev/mem");
        return EXIT_FAILURE;
    }

    // Map memory
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1) {
        perror("Failed to get page size");
        close(mem_fd);
        return EXIT_FAILURE;
    }

    void *virtual_base;
    virtual_base = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, FENRIR_BASE_ADDR);

    if (virtual_base == MAP_FAILED) {
        perror("mmap() failed");
        close(mem_fd);
        return EXIT_FAILURE;
    }

    // --- FENRIR configuration ---
    volatile FenrirMemoryMap *fenrir_regs = (volatile FenrirMemoryMap *)virtual_base;

    // Configure FENRIR
    fenrir_regs->fc1_config.synapse_loader.bits_per_weight      = fc1_cfg.bits_per_weight;
    fenrir_regs->fc1_config.synapse_loader.synapse_layer_offset = fc1_cfg.synapse_layer_offset;
    fenrir_regs->fc1_config.synapse_loader.synapse_neurons      = fc1_cfg.synapse_neurons;
    fenrir_regs->fc1_config.neuron_loader.neuron_layer_offset   = fc1_cfg.neuron_layer_offset;
    fenrir_regs->fc1_config.neuron_loader.neuron_neurons        = fc1_cfg.neuron_neurons;
    fenrir_regs->fc1_config.lif_config.weight_scalar            = fc1_cfg.weight_scalar;
    fenrir_regs->fc1_config.lif_config.beta                     = fc1_cfg.beta;
    fenrir_regs->fc1_config.lif_config.threshold                = fc1_cfg.threshold;
    fenrir_regs->fc1_config.neuron_writer.writer_layer_offset   = fc1_cfg.writer_layer_offset;
    fenrir_regs->fc1_config.neuron_writer.writer_neurons        = fc1_cfg.writer_neurons;

    fenrir_regs->fc2_config.synapse_loader.bits_per_weight      = fc2_cfg.bits_per_weight;
    fenrir_regs->fc2_config.synapse_loader.synapse_layer_offset = fc2_cfg.synapse_layer_offset;
    fenrir_regs->fc2_config.synapse_loader.synapse_neurons      = fc2_cfg.synapse_neurons;
    fenrir_regs->fc2_config.neuron_loader.neuron_layer_offset   = fc2_cfg.neuron_layer_offset;
    fenrir_regs->fc2_config.neuron_loader.neuron_neurons        = fc2_cfg.neuron_neurons;
    fenrir_regs->fc2_config.lif_config.weight_scalar            = fc2_cfg.weight_scalar;
    fenrir_regs->fc2_config.lif_config.beta                     = fc2_cfg.beta;
    fenrir_regs->fc2_config.lif_config.threshold                = fc2_cfg.threshold;
    fenrir_regs->fc2_config.neuron_writer.writer_layer_offset   = fc2_cfg.writer_layer_offset;
    fenrir_regs->fc2_config.neuron_writer.writer_neurons        = fc2_cfg.writer_neurons;

    // Init DVS and get info
    auto handle = libcaer::devices::dvXplorer(1);
    auto info   = handle.infoGet();
    printf("%s --- ID: %d, DVS X: %d, DVS Y: %d, Firmware: %d, Logic: %d.\n", info.deviceString, info.deviceID,
        info.dvsSizeX, info.dvsSizeY, info.firmwareVersion, info.logicVersion);

    /** Configure DVS
     *  - Only positive events
     *  - Low sensor bias
     *  - Crop to 480x480
     *  - Subsample to 240x240
     */
    handle.sendDefaultConfig();
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_ON_ONLY, true);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_BIAS_SIMPLE_VERY_LOW, true);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_CROPPER_X_START_ADDRESS, 0);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_CROPPER_X_END_ADDRESS, 480);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_CROPPER_Y_START_ADDRESS, 0);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_CROPPER_Y_END_ADDRESS, 480);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_CROPPER_ENABLE, true);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_SUBSAMPLE_ENABLE, true);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_SUBSAMPLE_HORIZONTAL, DVX_DVS_CHIP_SUBSAMPLE_HORIZONTAL_HALF);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_SUBSAMPLE_VERTICAL, DVX_DVS_CHIP_SUBSAMPLE_VERTICAL_HALF);

    // Start receiving data
    handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);
    handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

    auto last_timer_check = steady_clock::now();
    const auto timer_interval = milliseconds(10);

    // Init class counters
    uint64_t running_class_counts[11] = {0};
    uint32_t last_read_counts[11] = {0};

    for (int i = 0; i < 11; i++) {
        last_read_counts[i] = fenrir_regs->class_counts[i];
    }

	while (!globalShutdown.load(memory_order_relaxed)) {
        // Receive event packet
		std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
		if (packetContainer == nullptr) {
			continue;
		}

		for (auto &packet : *packetContainer) {
			if (packet == nullptr) {
				continue;
			}

			if (packet->getEventType() == POLARITY_EVENT) {
                // Cast base packet to PolarityEventPacket
				std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity
					= std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                for (const auto &event : *polarity) {
                    uint16_t x = event.getX();
                    uint16_t y = event.getY();

                    uint32_t flat = (uint32_t)y * TARGET_WIDTH + (uint32_t)x;
                    writeEvent(fenrir_regs, flat);
                }
			}
		}

        auto current_time = steady_clock::now();
        if (duration_cast<milliseconds>(current_time - last_timer_check) >= timer_interval) {
            last_timer_check = current_time;

            uint64_t max_count = 0;
            int max_index = -1;

            for (int i = 0; i < 11; i++) {
                // Read class count, compute and store delta
                uint32_t current_count = fenrir_regs->class_counts[i];
                uint32_t delta = current_count - last_read_counts[i];
                running_class_counts[i] += delta;
                last_read_counts[i] = current_count;

                if (running_class_counts[i] > max_count) {
                    max_count = running_class_counts[i];
                    max_index = i;
                }
            }

            if (max_index != -1) {
                printf("Most active neuron is Class %d with a running total of %llu events.\n", 
                       max_index, (unsigned long long)max_count);
            }
        }

	}

    handle.dataStop();
    if (munmap(virtual_base, MAP_SIZE) == -1) {
        perror("munmap() failed");
    }
    close(mem_fd);

    printf("Shutdown successful.\n");

    return EXIT_SUCCESS;
}

static void writeEvent(volatile FenrirMemoryMap *regs, uint32_t event) {
    static uint32_t write_count = 0;
    uint32_t data_to_write;

    if (write_count == 0) {
        data_to_write = (event & 0x7FFFFFFF) | 0x80000000;
    }
    else {
        data_to_write = event & 0x7FFFFFFF;
    }
    
    regs->write = data_to_write;
    write_count = (write_count + 1) % 2;
}


static int initShutdownHandler() {
    struct sigaction shutdownAction;
    shutdownAction.sa_handler = &globalShutdownSignalHandler;
    shutdownAction.sa_flags   = 0;
    sigemptyset(&shutdownAction.sa_mask);
    sigaddset(&shutdownAction.sa_mask, SIGTERM);
    sigaddset(&shutdownAction.sa_mask, SIGINT);

    if (sigaction(SIGTERM, &shutdownAction, NULL) == -1) {
        libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
            "Failed to set signal handler for SIGTERM. Error: %d.", errno);
        return (EXIT_FAILURE);
    }

    if (sigaction(SIGINT, &shutdownAction, NULL) == -1) {
        libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
            "Failed to set signal handler for SIGINT. Error: %d.", errno);
        return (EXIT_FAILURE);
    }

    return 1;
}

static void globalShutdownSignalHandler(int signal) {
    if (signal == SIGTERM || signal == SIGINT) {
        globalShutdown.store(true);
    }
}

static void usbShutdownHandler(void *ptr) {
    globalShutdown.store(true);
}