// MSVC (c++20)
// Format of input files:
// Plain text files named data_<size>.txt.
// Line 1: integer N (total count of elements).
// Lines 2..(N+1): exactly one floating-point number (double) per line,
// drawn from a uniform distribution in the range [-10.0, 10.0].

#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <algorithm>
#include <thread>
#include <random>
#include <fstream>
#include <chrono>
#include <string>
#include <format>
#include <cmath>
#include <stdexcept>
#include <cstddef>
#include <functional>

struct FileInfo {
    std::string name;
    std::size_t count;
};

const std::vector<FileInfo> FILES = {
    {"data_100k.txt", 100'000},
    {"data_1m.txt",   1'000'000},
    {"data_10m.txt",  10'000'000}
};

const std::vector<std::size_t> THREADS = {
    1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024
};

const int WARMUP = 1;
const int REPEATS = 5;
const double EPS = 1e-4;

struct alignas(64) ResultCell {
    double val{ 0.0 };
};

volatile double g_result = 0.0;

template <typename Func>
double measureMedianUs(Func&& func) {
    for (int i = 0; i < WARMUP; ++i) {
        func();
    }

    std::vector<double> times;
    times.reserve(REPEATS);

    for (int i = 0; i < REPEATS; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        func();
        auto t2 = std::chrono::high_resolution_clock::now();
        times.push_back(
            std::chrono::duration<double, std::micro>(t2 - t1).count()
        );
    }

    std::ranges::sort(times);
    return times[times.size() / 2];
}

void makeFiles() {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (const auto& f : FILES) {
        std::ifstream check(f.name);
        if (check.is_open()) {
            continue;
        }

        std::ofstream out(f.name);
        out << f.count << "\n";
        for (std::size_t i = 0; i < f.count; ++i) {
            out << dist(rng) << "\n";
        }
    }
}

std::vector<double> loadData(const std::string& fileName) {
    std::ifstream in(fileName);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    std::size_t n = 0;
    in >> n;

    std::vector<double> arr(n);
    for (std::size_t i = 0; i < n; ++i) {
        in >> arr[i];
    }
    return arr;
}

double myParallelReduce(const std::vector<double>& arr, std::size_t k) {
    const std::size_t n = arr.size();
    if (k <= 1 || n < k) {
        return std::reduce(arr.begin(), arr.end(), 0.0);
    }

    const std::size_t chunkSize = n / k;
    const std::size_t rem = n % k;

    std::vector<ResultCell> parts(k);
    std::vector<std::jthread> pool;
    pool.reserve(k);

    std::size_t start = 0;
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t currentChunk = chunkSize + (i < rem ? 1 : 0);
        const std::size_t finish = start + currentChunk;

        pool.emplace_back([&arr, &parts, i, start, finish]() {
            parts[i].val = std::reduce(arr.begin() + start, arr.begin() + finish, 0.0);
            });

        start = finish;
    }

    for (auto& th : pool) {
        th.join();
    }

    return std::transform_reduce(
        parts.begin(), parts.end(), 0.0, std::plus<>{},
        [](const ResultCell& cell) { return cell.val; }
    );
}

void testPolicies(const std::vector<double>& arr) {
    std::cout << "\n[1 & 2] Standard Policies Benchmark (Median Time in us)\n";
    std::cout << std::format("{:<36} | {:>14}\n", "Execution Policy", "Time (us)");
    std::cout << std::string(53, '-') << "\n";

    auto benchmarkPolicy = [&](const std::string& title, auto&& op) {
        double t = measureMedianUs([&]() {
            g_result = op();
            });
        std::cout << std::format("{:<36} | {:>14.2f}\n", title, t);
        };

    benchmarkPolicy("accumulate (sequential)", [&]() { return std::accumulate(arr.begin(), arr.end(), 0.0); });
    benchmarkPolicy("reduce (no policy)", [&]() { return std::reduce(arr.begin(), arr.end(), 0.0); });
    benchmarkPolicy("reduce (std::execution::seq)", [&]() { return std::reduce(std::execution::seq, arr.begin(), arr.end(), 0.0); });
    benchmarkPolicy("reduce (std::execution::par)", [&]() { return std::reduce(std::execution::par, arr.begin(), arr.end(), 0.0); });
    benchmarkPolicy("reduce (std::execution::par_unseq)", [&]() { return std::reduce(std::execution::par_unseq, arr.begin(), arr.end(), 0.0); });
    benchmarkPolicy("reduce (std::execution::unseq)", [&]() { return std::reduce(std::execution::unseq, arr.begin(), arr.end(), 0.0); });
}

void testCustomReduce(const std::vector<double>& arr, unsigned int numCores) {
    std::cout << "\n[3] Custom Parallel Algorithm Scaling Table (Varying K)\n";
    std::cout << std::format("{:>6} | {:>14} | {:>10} | {:>10}\n", "K", "Time (us)", "Speedup", "Validation");
    std::cout << std::string(52, '-') << "\n";

    const double correctVal = std::reduce(arr.begin(), arr.end(), 0.0);
    double t1 = 0.0;
    double bestTime = 1e18;
    std::size_t bestK = 1;

    for (std::size_t k : THREADS) {
        double myVal = 0.0;
        double t = measureMedianUs([&]() {
            myVal = myParallelReduce(arr, k);
            g_result = myVal;
            });

        if (k == 1) {
            t1 = t;
        }
        if (t < bestTime) {
            bestTime = t;
            bestK = k;
        }

        const double speedup = (t > 0.0) ? (t1 / t) : 0.0;
        const bool ok = std::fabs(myVal - correctVal) <= (EPS * std::fabs(correctVal) + 1e-5);

        std::cout << std::format("{:>6} | {:>14.2f} | {:>9.2f}x | {:>10}\n",
            k, t, speedup, ok ? "OK" : "MISMATCH");
    }

    std::cout << std::string(52, '-') << "\n";
    std::cout << std::format("Optimal K:                {} threads\n", bestK);
    std::cout << std::format("Optimal Time:             {:.2f} us\n", bestTime);
    std::cout << std::format("Hardware CPU Threads:     {}\n", numCores);
    std::cout << std::format("Ratio (Best K / HW):      {:.2f}\n", static_cast<double>(bestK) / numCores);
}

int main(int argc, char* argv[]) {
    try {
        makeFiles();

        std::string mode = "O2";
        if (argc > 1) {
            mode = argv[1];
        }
        else {
            std::cout << "No mode specified, using default: O2\n";
        }

        const unsigned int numCores = std::thread::hardware_concurrency();

        std::cout << "Project 2 variant 13 (std::reduce)\n";
        std::cout << std::format("Optimization level:     {}\n", mode);
        std::cout << std::format("CPU hardware threads:   {}\n", numCores);

        for (const auto& f : FILES) {
            std::cout << std::format("\n Processing dataset: {} ({} elements)\n", f.name, f.count);
            const std::vector<double> arr = loadData(f.name);

            testPolicies(arr);
            testCustomReduce(arr, numCores);
        }

        std::cout << "\nExecution finished successfully.\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}