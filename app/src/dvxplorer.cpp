#include <libcaercpp/devices/dvxplorer.hpp>

#include <atomic>
#include <csignal>

#include <chrono> // For std::chrono

using namespace std;
using namespace std::chrono; // Use chrono for timing

static atomic_bool globalShutdown(false);

static void globalShutdownSignalHandler(int signal) {
    // Simply set the running flag to false on SIGTERM and SIGINT (CTRL+C) for global shutdown.
    if (signal == SIGTERM || signal == SIGINT) {
        globalShutdown.store(true);
    }
}

static void usbShutdownHandler(void *ptr) {
    (void) (ptr); // UNUSED.

    globalShutdown.store(true);
}

int main(void) {
	// Install signal handler for global shutdown.
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

    // Open a DVS, give it a device ID of 1, and don't care about USB bus or SN restrictions.
    auto handle = libcaer::devices::dvXplorer(1);

    // Let's take a look at the information we have on the device.
    auto info = handle.infoGet();

    printf("%s --- ID: %d, DVS X: %d, DVS Y: %d, Firmware: %d, Logic: %d.\n", info.deviceString, info.deviceID,
        info.dvsSizeX, info.dvsSizeY, info.firmwareVersion, info.logicVersion);

    // Send the default configuration before using the device.
    // No configuration is sent automatically!
    handle.sendDefaultConfig();

    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_ON_ONLY, true);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_BIAS_SIMPLE_VERY_LOW, true);

    // Now let's get start getting some data from the device. We just loop in blocking mode,
    // no notification needed regarding new events. The shutdown notification, for example if
    // the device is disconnected, should be listened to.
    handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);

    // Let's turn on blocking data-get mode to avoid wasting resources.
    handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

	using namespace std::chrono;
	int positiveCount = 0;
	auto lastPrintTime = steady_clock::now();

	while (!globalShutdown.load(memory_order_relaxed)) {
		std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
		if (packetContainer == nullptr) {
			continue; // Skip if nothing there.
		}

		for (auto &packet : *packetContainer) {
			if (packet == nullptr) {
				continue; // Skip if nothing there.
			}

			if (packet->getEventType() == POLARITY_EVENT) {
				std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity
					= std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

				for (const auto &e : *polarity) {
					if (e.getPolarity()) {
						positiveCount++;
					}
				}
			}
		}

		auto now = steady_clock::now();
		if (duration_cast<seconds>(now - lastPrintTime).count() >= 1) {
			printf("Positive events in last second: %d\n", positiveCount);
			positiveCount = 0;
			lastPrintTime = now;
		}
	}

    handle.dataStop();

    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}