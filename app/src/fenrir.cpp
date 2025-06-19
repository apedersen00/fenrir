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
#include <unordered_set>
#include <vector>
#include <string>

#include <libcaercpp/devices/dvxplorer.hpp>
#include <libcaercpp/filters/dvs_noise.hpp>

#include "fenrir.hpp"
// #include "gesture_data_target_1.h"

#define MAP_SIZE 4096
#define TARGET_WIDTH 60 // If changing this, also update DVX_DVS_CHIP config
#define RUN_TEST 0

using namespace std;
using namespace std::chrono;

static int initShutdownHandler();
static void globalShutdownSignalHandler(int signal);
static void usbShutdownHandler(void *ptr);
static void writeEvent(volatile FenrirMemoryMap *regs, uint32_t event);

static atomic_bool globalShutdown(false);

const std::vector<std::string> class_map = {
    "hand_clapping",
    "right_hand_wave",
    "left_hand_wave",
    "right_hand_clockwise",
    "right_hand_counter_clockwise",
    "left_hand_clockwise",
    "left_hand_counter_clockwise",
    "forearm_roll", 
    "drums",
    "guitar",
    "random_other_gestures"
};

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
     *  - Subsample to 60x60
     */
    handle.sendDefaultConfig();
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_ON_ONLY, true);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_BIAS_SIMPLE, true);

    libcaer::filters::DVSNoise noiseFilter(info.dvsSizeX, info.dvsSizeY);

    noiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TIME, 5000);
    noiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_ENABLE, true);
    noiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_TIME, 1000);
    noiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_ENABLE, true);

    printf("Denoising filter configured and enabled.\n");

    // Start receiving data
    handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);
    handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

    const auto timestep_interval    = microseconds(16600);
    const auto counters_interval    = microseconds(3000000);
    const uint32_t TIMESTEP_EVENT   = 4096;

    auto last_timestep_check = steady_clock::now();
    auto last_counters_check = steady_clock::now();

    // Init class counters
    uint64_t running_class_counts[11] = {0};
    uint32_t last_read_counts[11] = {0};

    for (int i = 0; i < 11; i++) {
        last_read_counts[i] = fenrir_regs->class_counts[i];
    }

    std::unordered_set<uint32_t> fired_addresses_in_timestep;

    fenrir_regs->control = 0x00000003;

    uint32_t event_counter = 0;
    uint32_t event_cnt_all = 0;

#if RUN_TEST == 1

    printf("--- Starting N-MNIST data test ---\n");

    // 1. Read the initial state of the counters to establish a baseline
    uint32_t initial_counts[11] = {0};
    printf("Reading initial hardware counter values...\n");
    for (int i = 0; i < 11; i++) {
        initial_counts[i] = fenrir_regs->class_counts[i];
    }

    // 2. Write all events from the header file with a small delay
    printf("Writing %d events to hardware with a 20us delay between each...\n", NMNIST_EVENTS_SIZE);
    for (int i = 0; i < NMNIST_EVENTS_SIZE; i++) {
        writeEvent(fenrir_regs, nmnist_events[i]);
        usleep(20); // Add a 20 microsecond delay to pace the events
    }
    
    printf("All events written.\n");

    // 4. Wait for a moment to allow the hardware to complete processing
    sleep(1); // Wait for 1 second

    // 5. Read the final results and calculate the delta from the initial state
    printf("\n--- Final Classification Counts (for this test run) ---\n");
    uint32_t max_delta = 0;
    int max_index = -1;

    for (int i = 0; i < 11; i++) {
        uint32_t final_count = fenrir_regs->class_counts[i];
        // Calculate the difference to see what happened during this test
        uint32_t delta = final_count - initial_counts[i];
        
        printf("  Class '%s': %u new events (Initial: %u, Final: %u)\n", 
               class_map[i].c_str(), delta, initial_counts[i], final_count);
        
        if (delta > max_delta) {
            max_delta = delta;
            max_index = i;
        }
    }

    // 6. Print the final prediction based on the delta
    if (max_index != -1 && max_delta > 0) {
        printf("\nPredicted Gesture: %s (%u events)\n", class_map[max_index].c_str(), max_delta);
    } else {
        printf("\nNo gesture recognized.\n");
    }

#else

    while (!globalShutdown.load(memory_order_relaxed)) {
        std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
        if (packetContainer == nullptr) {
            continue;
        }

        for (auto &packet : *packetContainer) {
            if (packet == nullptr || packet->getEventType() != POLARITY_EVENT) {
                continue;
            }

            auto polarityPacket = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

            noiseFilter.apply(*polarityPacket);

            const uint16_t CROP_X_MAX       = 512;
            const uint16_t CROP_Y_MAX       = 512;
            const uint16_t SUBSAMPLE_SHIFT  = 3;    // 2^3 = 8

            for (const auto &event : *polarityPacket) {
                if (!event.isValid()) {
                    continue;
                }

                uint16_t x = event.getX();
                uint16_t y = event.getY();

                if (x >= CROP_X_MAX || y >= CROP_Y_MAX) {
                    continue;
                }

                uint16_t subsampled_x = x >> SUBSAMPLE_SHIFT;
                uint16_t subsampled_y = y >> SUBSAMPLE_SHIFT;
                uint32_t flat_address = (uint32_t)subsampled_y * TARGET_WIDTH + (uint32_t)subsampled_x;

                if (fired_addresses_in_timestep.insert(flat_address).second) {
                    writeEvent(fenrir_regs, flat_address);
                }
            }
        }

        auto current_time = steady_clock::now();

        if (duration_cast<microseconds>(current_time - last_timestep_check) >= timestep_interval) {

            last_timestep_check = current_time;

            fired_addresses_in_timestep.clear();

            fenrir_regs->write = TIMESTEP_EVENT;
        }

        if (duration_cast<microseconds>(current_time - last_counters_check) >= counters_interval) {
            last_counters_check = current_time;

            uint32_t max_delta = 0;
            int max_index = -1;

            printf("--- Interval Report ---\n");

            for (int i = 0; i < 11; i++) {
                uint32_t current_count = fenrir_regs->class_counts[i];

                uint32_t delta = current_count - last_read_counts[i];
                last_read_counts[i] = current_count;

                if (delta > 0) {
                    printf("  Class '%s': %u new events\n", class_map[i].c_str(), delta);
                }

                if (delta > max_delta) {
                    max_delta = delta;
                    max_index = i;
                }
            }

            printf("Number of events: %u\n", event_counter);
            printf("All events: %u\n", event_cnt_all);
            event_counter = 0;
            event_cnt_all = 0;

            if (max_index != -1 && max_delta > 0) {
                printf("\nMost active gesture THIS interval: %s (%u events)\n\n",
                       class_map[max_index].c_str(), max_delta);
            } else {
                printf("\nNo significant gesture activity in this interval.\n\n");
            }
        }

	}
#endif

    fenrir_regs->control = 0x00000000;

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

    return EXIT_SUCCESS;
}

static void globalShutdownSignalHandler(int signal) {
    if (signal == SIGTERM || signal == SIGINT) {
        globalShutdown.store(true);
    }
}

static void usbShutdownHandler(void *ptr) {
    globalShutdown.store(true);
}