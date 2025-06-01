#include <libcaercpp/devices/dvxplorer.hpp>

#include <atomic>
#include <csignal>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp> // Required for cv::resize

#include <chrono> // For std::chrono

#if !defined(LIBCAER_HAVE_OPENCV) || LIBCAER_HAVE_OPENCV == 0
#   error "This example requires OpenCV support in libcaer to be enabled."
#endif

#define DEMO 0 // Set DEMO to 0 to enable the visualization and new features

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

#if DEMO == 0
	// Define downscaling factor (e.g., 2 for half resolution)
	const int DOWNSCALE_FACTOR = 2;
	const int DOWNSCALED_WIDTH = info.dvsSizeX / DOWNSCALE_FACTOR;
	const int DOWNSCALED_HEIGHT = info.dvsSizeY / DOWNSCALE_FACTOR;

	// Define upscaling factor for display only
	const int DISPLAY_UPSCALE_FACTOR = 4; // Make it 4 times larger than the native downscaled resolution
	const int DISPLAY_WIDTH = DOWNSCALED_WIDTH * DISPLAY_UPSCALE_FACTOR;
	const int DISPLAY_HEIGHT = DOWNSCALED_HEIGHT * DISPLAY_UPSCALE_FACTOR;

	cv::namedWindow("ORIGINAL_EVENTS",
		cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO | cv::WindowFlags::WINDOW_GUI_EXPANDED);
	cv::namedWindow("DOWNSCALED_EVENTS",
		cv::WindowFlags::WINDOW_NORMAL); // Allows resizing by user
	cv::namedWindow("BINNED_EVENTS",
		cv::WindowFlags::WINDOW_NORMAL); // Allows resizing by user

	cv::resizeWindow("DOWNSCALED_EVENTS", DISPLAY_WIDTH, DISPLAY_HEIGHT);
	cv::resizeWindow("BINNED_EVENTS", DISPLAY_WIDTH, DISPLAY_HEIGHT);

	// Variables for event counting in downscaled space
	long long totalDownscaledEventsCount = 0; // Using long long for potentially large counts
	auto lastPrintTime = steady_clock::now();


	while (!globalShutdown.load(memory_order_relaxed)) {
		std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
		if (packetContainer == nullptr) {
			continue; // Skip if nothing there.
		}

		cv::Mat originalEventsFrame(info.dvsSizeY, info.dvsSizeX, CV_8UC3, cv::Vec3b{127, 127, 127});
		cv::Mat binnedEventsFrame(DOWNSCALED_HEIGHT, DOWNSCALED_WIDTH, CV_32FC1, 0.0f); // Stores event counts

		for (auto &packet : *packetContainer) {
			if (packet == nullptr) {
				continue; // Skip if nothing there.
			}

			if (packet->getEventType() == POLARITY_EVENT) {
				std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity
					= std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

				for (const auto &e : *polarity) {
					// Original events for display
					originalEventsFrame.at<cv::Vec3b>(e.getY(), e.getX())
						= e.getPolarity() ? cv::Vec3b{255, 255, 255} : cv::Vec3b{0, 0, 0};

					// --- Event Binning and Counting ---
					int binnedX = e.getX() / DOWNSCALE_FACTOR;
					int binnedY = e.getY() / DOWNSCALE_FACTOR;

					if (binnedX >= 0 && binnedX < DOWNSCALED_WIDTH &&
						binnedY >= 0 && binnedY < DOWNSCALED_HEIGHT) {
						// For counting, we care about each event, regardless of polarity for total count
						// If you only want positive events, use: if (e.getPolarity()) totalDownscaledEventsCount++;
						totalDownscaledEventsCount++;
						// Accumulate for visual binning (polarity-aware for display)
						binnedEventsFrame.at<float>(binnedY, binnedX) += (e.getPolarity() ? 1.0f : -1.0f);
					}
				}
			}
		}

		// Display original events
		cv::imshow("ORIGINAL_EVENTS", originalEventsFrame);

		// --- Downscaling and Upscaling for Display ---
		cv::Mat downscaledEventsFrame;
		cv::resize(originalEventsFrame, downscaledEventsFrame,
				cv::Size(DOWNSCALED_WIDTH, DOWNSCALED_HEIGHT), 0, 0, cv::INTER_NEAREST);

		cv::Mat displayDownscaledFrame;
		cv::resize(downscaledEventsFrame, displayDownscaledFrame,
				cv::Size(DISPLAY_WIDTH, DISPLAY_HEIGHT), 0, 0, cv::INTER_NEAREST);
		cv::imshow("DOWNSCALED_EVENTS", displayDownscaledFrame);


		// --- Visualize Binned Events ---
		cv::Mat displayBinnedFrame;
		double minVal, maxVal;
		cv::minMaxLoc(binnedEventsFrame, &minVal, &maxVal);

		if (maxVal > 0) {
			binnedEventsFrame.convertTo(displayBinnedFrame, CV_8UC1, 255.0 / maxVal, 0);
		} else {
			displayBinnedFrame = cv::Mat::zeros(DOWNSCALED_HEIGHT, DOWNSCALED_WIDTH, CV_8UC1);
		}
		cv::cvtColor(displayBinnedFrame, displayBinnedFrame, cv::COLOR_GRAY2BGR);

		cv::Mat finalDisplayBinnedFrame;
		cv::resize(displayBinnedFrame, finalDisplayBinnedFrame,
				cv::Size(DISPLAY_WIDTH, DISPLAY_HEIGHT), 0, 0, cv::INTER_NEAREST);
		cv::imshow("BINNED_EVENTS", finalDisplayBinnedFrame);

		cv::waitKey(1); // Small delay to allow window events to be processed

		// --- Print event count every second ---
		auto now = steady_clock::now();
		if (duration_cast<seconds>(now - lastPrintTime).count() >= 1) {
			printf("Events in downscaled bins in last second: %lld\n", totalDownscaledEventsCount);
			totalDownscaledEventsCount = 0; // Reset count for the next second
			lastPrintTime = now;
		}
	}
#elif DEMO == 1
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
#endif

    handle.dataStop();

    // Close automatically done by destructor.
#if DEMO == 0
    cv::destroyWindow("ORIGINAL_EVENTS");
	cv::destroyWindow("DOWNSCALED_EVENTS");
	cv::destroyWindow("BINNED_EVENTS");
#endif

    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}