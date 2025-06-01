#include <libcaercpp/devices/dvxplorer.hpp>

#include <atomic>
#include <csignal>
#include <cmath> // For std::round
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
    bool eventOnOnlyIsEnabled = true; 
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_ON_ONLY, eventOnOnlyIsEnabled);
    handle.configSet(DVX_DVS_CHIP, DVX_DVS_CHIP_BIAS_SIMPLE_VERY_LOW, true);
    handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);
    handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

#if DEMO == 0
    const int DOWNSCALE_FACTOR = 10;
    const int DOWNSCALED_WIDTH = info.dvsSizeX / DOWNSCALE_FACTOR;
    const int DOWNSCALED_HEIGHT = info.dvsSizeY / DOWNSCALE_FACTOR;
    const int DISPLAY_UPSCALE_FACTOR = 20; 
    const int DISPLAY_WIDTH = DOWNSCALED_WIDTH * DISPLAY_UPSCALE_FACTOR;
    const int DISPLAY_HEIGHT = DOWNSCALED_HEIGHT * DISPLAY_UPSCALE_FACTOR;

    cv::namedWindow("BINNED_EVENTS", cv::WindowFlags::WINDOW_NORMAL);
    if (DOWNSCALED_WIDTH > 0 && DOWNSCALED_HEIGHT > 0) {
        cv::resizeWindow("BINNED_EVENTS", DISPLAY_WIDTH, DISPLAY_HEIGHT);
    }

    const auto frameBinDuration = std::chrono::microseconds(16600); 
    auto nextBinTime = std::chrono::steady_clock::now() + frameBinDuration;
    
    // --- MODIFIED: Renamed counter for clarity ---
    long long mergedEventsInBin = 0; // Counts unique active downscaled pixels

    if (DOWNSCALED_WIDTH <= 0 || DOWNSCALED_HEIGHT <= 0) {
        fprintf(stderr, "ERROR: Downscaled dimensions are not positive (%dx%d). Adjust DOWNSCALE_FACTOR or check camera info.\n", DOWNSCALED_WIDTH, DOWNSCALED_HEIGHT);
        return EXIT_FAILURE;
    }
    cv::Mat binnedEventsFrame(DOWNSCALED_HEIGHT, DOWNSCALED_WIDTH, CV_32FC1, 0.0f); // For visualizing raw event density
    // --- NEW: Grid to track active downscaled pixels in the current bin ---
    cv::Mat activityGrid(DOWNSCALED_HEIGHT, DOWNSCALED_WIDTH, CV_8UC1, cv::Scalar(0)); 

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
                        int binnedX = e.getX() / DOWNSCALE_FACTOR;
                        int binnedY = e.getY() / DOWNSCALE_FACTOR;

                        if (binnedX >= 0 && binnedX < DOWNSCALED_WIDTH &&
                            binnedY >= 0 && binnedY < DOWNSCALED_HEIGHT) {
                            // Accumulate in binnedEventsFrame for visualization (raw event count per bin)
                            binnedEventsFrame.at<float>(binnedY, binnedX) += 1.0f; 
                                
                            // --- MODIFIED: Count unique active downscaled pixels ---
                            if (activityGrid.at<unsigned char>(binnedY, binnedX) == 0) {
                                activityGrid.at<unsigned char>(binnedY, binnedX) = 1; // Mark as active
                                mergedEventsInBin++; // Increment count of unique active bins
                            }
                        }
                    }
                }
            }
        }

        auto currentTime = std::chrono::steady_clock::now();

        if (currentTime >= nextBinTime) {
            // Visualization of binnedEventsFrame (raw event density) remains the same
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
                if (newMax > 0) { 
                   tempNormFrame.convertTo(displayBinnedTemp, CV_8UC1, 255.0 / newMax, 0);
                } else {
                    displayBinnedTemp = cv::Mat::zeros(DOWNSCALED_HEIGHT, DOWNSCALED_WIDTH, CV_8UC1);
                }
            }

            cv::Mat displayBinnedFrameBGR;
            cv::cvtColor(displayBinnedTemp, displayBinnedFrameBGR, cv::COLOR_GRAY2BGR);

            cv::Mat finalDisplayBinnedFrame;
            cv::resize(displayBinnedFrameBGR, finalDisplayBinnedFrame,
                    cv::Size(DISPLAY_WIDTH, DISPLAY_HEIGHT), 0, 0, cv::INTER_NEAREST);
            cv::imshow("BINNED_EVENTS", finalDisplayBinnedFrame);

            // --- MODIFIED: Print the new merged event count ---
            double totalRawDVSEvents = cv::sum(binnedEventsFrame)[0]; // Still useful for context
            printf("Merged events (active downscaled pixels) in bin (%.1fms): %lld (Total raw DVS events: %.0f)\n",
                   std::chrono::duration<double, std::milli>(frameBinDuration).count(),
                   mergedEventsInBin,
                   totalRawDVSEvents);
            
            // --- Reset for the next bin ---
            binnedEventsFrame.setTo(0.0f);
            activityGrid.setTo(cv::Scalar(0)); // Reset the activity tracker
            mergedEventsInBin = 0;             // Reset the merged event counter

            do {
                nextBinTime += frameBinDuration;
            } while (nextBinTime <= currentTime);
        }

        cv::waitKey(1);
    }
#elif DEMO == 1
// ... (original DEMO == 1 code remains unchanged)
#endif

    handle.dataStop();

#if DEMO == 0
    cv::destroyWindow("BINNED_EVENTS");
#endif

    printf("Shutdown successful.\n");
    return (EXIT_SUCCESS);
}