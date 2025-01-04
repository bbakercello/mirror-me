#include <filesystem>
#include <cstdlib>
#include <iostream>

void trim_audio(const std::string& input_path, const std::string& output_path, int start_sec, int duration_sec) {
    // Check if input file exists
    if (!std::filesystem::exists(input_path)) {
        std::cerr << "Input file not found: " << input_path << std::endl;
        return;
    }

    // Build FFmpeg command
    std::string command = "ffmpeg -y -i " + input_path +
                          " -ss " + std::to_string(start_sec) +
                          " -t " + std::to_string(duration_sec) +
                          " " + output_path;

    // Execute the command
    int ret = std::system(command.c_str());
    if (ret == 0) {
        std::cout << "Trimmed audio saved to " << output_path << std::endl;
    } else {
        std::cerr << "Error trimming audio!" << std::endl;
    }
}

int main() {
    trim_audio("./examples/audio/dev_audio/itmightcomeback.wav", "./examples/audio/test_audio/trimmed_audio.wav", 0, 5);
    return 0;
}
