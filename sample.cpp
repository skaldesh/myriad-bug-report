/*
 * Myriad X / OpenVINO Bug report
 */

#include <string>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <atomic>
#include <csignal>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace std::chrono_literals;

std::atomic<bool> interrupt = {false};
std::atomic<uint> fpsCount  = {0};

void interruptHandler(int signum) {
    interrupt = true;
    std::cerr << "Interrupted! Stopping now..." << std::endl;
}

bool interrupted() {
    return interrupt;
}

void fpsRoutine() {
    // Print 11 times the fps counter, then append a newline.
    std::cout << "FPS: ";
    int count = 0;
    while (!interrupted()) {
        std::this_thread::sleep_for(1s);
        count++;
        std::cout << std::to_string(fpsCount.exchange(0));
        if (count > 10) {
            std::cout << std::endl << std::flush;
            count = 0;
        } else {
            std::cout << ", " << std::flush;
        }
    }
}

void inferenceRoutine(std::string model_path, std::string config_path) {
    auto dm = cv::dnn::DetectionModel(model_path, config_path);
    dm.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    dm.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);

    // Black input image.
    const cv::Mat input(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<int> classIDs;
    std::vector<float> confs;
    std::vector<cv::Rect> boxes;

    try {
        while (!interrupted()) {
            // Do inference.
            dm.detect(input, classIDs, confs, boxes);
            fpsCount++;
        }
    } catch (const std::exception& e) {
        std::cout << "Exception in inference routine: " << std::string(e.what()) << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Register signal handler to detect interrupts (e.g. Ctrl+C, docker stop, ...).  
    signal(SIGINT, interruptHandler);
    signal(SIGTERM, interruptHandler);

    if (argc < 2) {
        std::cout << "Usage: sample <num-vpus>" << std::endl;
        return 1;
    }
    int numVPUs = atoi(argv[1]);

    // Spawn routines.
    auto fpsThread = std::thread(fpsRoutine);
    std::vector<std::thread> infThreads;
    for (int i = 0; i < numVPUs; ++i) {
        infThreads.push_back(std::thread(inferenceRoutine, "/model.bin", "/model.xml"));
    }

    // Wait until all threads have exited.
    fpsThread.join();
    for (int i = 0; i < numVPUs; ++i) {
        infThreads[i].join();
    }

    std::cout << "All threads gracefully exited" << std::endl << std::flush;
    return 0;
}
