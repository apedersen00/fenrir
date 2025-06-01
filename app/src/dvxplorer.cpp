#include <libcaercpp/devices/dvxplorer.hpp>

#include <atomic>
#include <csignal>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp> // Required for cv::resize

#include <chrono> // For std::chrono
#include <iostream> // For printf/cout if needed

#if !defined(LIBCAER_HAVE_OPENCV) || LIBCAER_HAVE_OPENCV == 0
#   error "This example requires OpenCV support in libcaer to be enabled."
#endif

#define DEMO 0 // Set DEMO to 0 to enable the visualization and new features

using namespace std;
using namespace std::chrono; // Use chrono for timing

static atomic_bool globalShutdown(false);

static void globalShutdownSignalHandler(int signal) {
    if (signal == SIGTERM || signal == SIGINT) {
        globalShutdown.store(true);
    }
}

static void usbShutdownHandler(void *ptr) {
    (void) (ptr);
    globalShutdown.store(true);
}

int main(void) {
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

    auto handle = libcaer::devices::dvXplorer(1);
    auto info = handle.infoGet();

    printf("%s --- ID: %d, DVS X: %d, DVS Y: %d, Firmware: %d, Logic: %d.\n", info.deviceString, info.deviceID,
        info.dvsSizeX, info.dvsSizeY, info.firmwareVersion, info.logicVersion);

    handle.sendDefaultConfig();
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_ON_ONLY, true);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_BIAS_SIMPLE_VERY_LOW, true);
    handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);
    handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

#if DEMO == 0
    const int DOWNSCALE_FACTOR = 10;
    const int DOWNSCALED_WIDTH = info.dvsSizeX / DOWNSCALE_FACTOR;
    const int DOWNSCALED_HEIGHT = info.dvsSizeY / DOWNSCALE_FACTOR;
    const int DISPLAY_UPSCALE_FACTOR = 20; // You can adjust this for the single window's appearance
    const int DISPLAY_WIDTH = DOWNSCALED_WIDTH * DISPLAY_UPSCALE_FACTOR;
    const int DISPLAY_HEIGHT = DOWNSCALED_HEIGHT * DISPLAY_UPSCALE_FACTOR;

    // --- MODIFIED: Only one window needed ---
    cv::namedWindow("BINNED_EVENTS", cv::WindowFlags::WINDOW_NORMAL);
    cv::resizeWindow("BINNED_EVENTS", DISPLAY_WIDTH, DISPLAY_HEIGHT);

    const auto frameBinDuration = std::chrono::microseconds(16600); // Approx 16.6 ms
    auto nextBinTime = std::chrono::steady_clock::now() + frameBinDuration;
    long long eventsInCurrentBin = 0;

    cv::Mat binnedEventsFrame(DOWNSCALED_HEIGHT, DOWNSCALED_WIDTH, CV_32FC1, 0.0f);

    while (!globalShutdown.load(memory_order_relaxed)) {
        std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
        
        if (packetContainer != nullptr) {
            for (auto &packet : *packetContainer) {
                if (packet == nullptr) {
                    continue;
                }

                if (packet->getEventType() == POLARITY_EVENT) {
                    std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity
                        = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                    for (const auto &e : *polarity) {
                        // --- Event Binning and Counting for 16.6ms Frame ---
                        int binnedX = e.getX() / DOWNSCALE_FACTOR;
                        int binnedY = e.getY() / DOWNSCALE_FACTOR;

                        if (binnedX >= 0 && binnedX < DOWNSCALED_WIDTH &&
                            binnedY >= 0 && binnedY < DOWNSCALED_HEIGHT) {
                            eventsInCurrentBin++;
                            binnedEventsFrame.at<float>(binnedY, binnedX) += (e.getPolarity() ? 1.0f : -1.0f);
                        }
                    }
                }
            }
        }

        auto currentTime = std::chrono::steady_clock::now();

        if (currentTime >= nextBinTime) {
            cv::Mat displayBinnedTemp;
            double minVal, maxVal;
            cv::minMaxLoc(binnedEventsFrame, &minVal, &maxVal);

            if (minVal == maxVal) {
                displayBinnedTemp = cv::Mat::zeros(DOWNSCALED_HEIGHT, DOWNSCALED_WIDTH, CV_8UC1);
                if (minVal > 0) displayBinnedTemp.setTo(cv::Scalar(255));
            } else {
                cv::Mat tempNormFrame = binnedEventsFrame.clone();
                tempNormFrame -= minVal;
                double newMax = maxVal - minVal;
                tempNormFrame.convertTo(displayBinnedTemp, CV_8UC1, 255.0 / newMax, 0);
            }

            cv::Mat displayBinnedFrameBGR;
            cv::cvtColor(displayBinnedTemp, displayBinnedFrameBGR, cv::COLOR_GRAY2BGR);

            cv::Mat finalDisplayBinnedFrame;
            cv::resize(displayBinnedFrameBGR, finalDisplayBinnedFrame,
                    cv::Size(DISPLAY_WIDTH, DISPLAY_HEIGHT), 0, 0, cv::INTER_NEAREST);
            cv::imshow("BINNED_EVENTS", finalDisplayBinnedFrame);

            printf("Events in downscaled bin (%.1fms): %lld\n",
                   std::chrono::duration<double, std::milli>(frameBinDuration).count(),
                   eventsInCurrentBin);

            binnedEventsFrame.setTo(0.0f);
            eventsInCurrentBin = 0;

            do {
                nextBinTime += frameBinDuration;
            } while (nextBinTime <= currentTime);
        }

        // --- MODIFIED: Removed display of ORIGINAL_EVENTS and DOWNSCALED_EVENTS ---

        cv::waitKey(1);
    }
#elif DEMO == 1
// ... (original DEMO == 1 code remains unchanged)
    using namespace std::chrono;
    int positiveCount = 0;
    auto lastPrintTime = steady_clock::now();

    while (!globalShutdown.load(memory_order_relaxed)) {
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

#if DEMO == 0
    // --- MODIFIED: Only one window to destroy ---
    cv::destroyWindow("BINNED_EVENTS");
#endif

    printf("Shutdown successful.\n");
    return (EXIT_SUCCESS);
}