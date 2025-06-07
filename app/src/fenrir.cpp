#include <libcaercpp/devices/dvxplorer.hpp>

#include <atomic>
#include <csignal>
#include <chrono>

using namespace std;
using namespace std::chrono;

static void initShutdownHandler();
static void globalShutdownSignalHandler(int signal);
static void usbShutdownHandler(void *ptr);

static atomic_bool globalShutdown(false);

int main(void) {
    initShutdownHandler();

    // Init DVS and get info
    auto handle = libcaer::devices::dvXplorer(1);
    auto info   = handle.infoGet();
    printf("%s --- ID: %d, DVS X: %d, DVS Y: %d, Firmware: %d, Logic: %d.\n", info.deviceString, info.deviceID,
        info.dvsSizeX, info.dvsSizeY, info.firmwareVersion, info.logicVersion);

    // Configure DVS
    handle.sendDefaultConfig();
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_ON_ONLY, true);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_BIAS_SIMPLE_VERY_LOW, true);

    // Start receiving data
    handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);
    handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

	int positiveCount = 0;
	auto lastPrintTime = steady_clock::now();

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
				std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity
					= std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                for (int i = 0; i < packet->getEventCapacity(); i++) {
                    libcaer::events::PolarityEvent &iEvent = (*polarity)[i];

                    uint16_t x = iEvent.getX();
                    uint16_t y = iEvent.getY();

                    printf("Event - x: %d, y: %d\n", x, y);
                }
			}
		}
	}

    handle.dataStop();
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}

static void initShutdownHandler() {
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
}

static void globalShutdownSignalHandler(int signal) {
    if (signal == SIGTERM || signal == SIGINT) {
        globalShutdown.store(true);
    }
}

static void usbShutdownHandler(void *ptr) {
    globalShutdown.store(true);
}