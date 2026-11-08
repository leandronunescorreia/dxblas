// dxblas_gpumem_roundtrip
//
// CPU -> GPU -> CPU roundtrip benchmark and validation for gpuMem<T>.
//
// What it measures, per element type and per data size:
//   cold             first Context::copy on fresh gpuMem objects: includes texture
//                    creation and any staging growth (single sample, not averaged)
//   upload submit    wall time of Context::copy returning (CPU-side submission only)
//   upload complete  Context::copy plus waiting for the GPU to finish (D3D11 EVENT query)
//   readback         copy_to_cpu_batch of both inputs, started with the GPU idle
//
// Consistency measures:
//   * warm-up iterations are discarded, measured iterations are adaptive in count
//     (many for small sizes, fewer for huge ones) unless --iterations is given
//   * the GPU is always idle at the start of every timed phase
//   * readback destinations are pre-filled with a sentinel outside the timed region,
//     so first-touch page faults are not billed to the readback
//   * results are validated bit-exactly on the cold run and on the last iteration
//   * statistics: median, min and coefficient of variation (CV%) per phase;
//     throughput is computed from the median (decimal GB/s)
//   * optional per-iteration raw samples go to a CSV file (--csv)

#include <dxblas/dxblas.hpp>
#include <dxblas/device_selection.hpp>

#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;

namespace {

// ---------------------------------------------------------------- options --

struct Options {
    int iterations = 0;       // 0 = adaptive
    int warmup = 0;           // 0 = adaptive
    bool large = false;       // also run the 256 Mi-element case
    std::string type = "all"; // float32 | int32 | int16 | uint8 | all
    std::string csv_path;
};

void print_usage() {
    std::cout << "Usage: dxblas_gpumem_roundtrip [options]\n"
              << "  --iterations N   measured iterations per case (default: adaptive, 7..200)\n"
              << "  --warmup N       discarded warm-up iterations (default: adaptive, 2..10)\n"
              << "  --type T         float32 | int32 | int16 | uint8 | all (default: all)\n"
              << "  --large          also run 67,108,864 elements (needs roughly 1-2.5 GiB)\n"
              << "  --csv FILE       write per-iteration raw samples to FILE\n"
              << "  --help           show this text\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next_int = [&](int& out) {
            if (i + 1 >= argc)
                return false;
            try {
                const int value = std::stoi(argv[++i]);
                if (value <= 0)
                    return false;
                out = value;
                return true;
            } catch (...) {
                return false;
            }
        };
        if (arg == "--iterations") {
            if (!next_int(options.iterations))
                return false;
        } else if (arg == "--warmup") {
            if (!next_int(options.warmup))
                return false;
        } else if (arg == "--type") {
            if (i + 1 >= argc)
                return false;
            options.type = argv[++i];
            if (options.type != "all" && options.type != "float32" && options.type != "int32" &&
                options.type != "int16" && options.type != "uint8")
                return false;
        } else if (arg == "--large") {
            options.large = true;
        } else if (arg == "--csv") {
            if (i + 1 >= argc)
                return false;
            options.csv_path = argv[++i];
        } else {
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------- statistics --

struct Stats {
    double min = std::numeric_limits<double>::quiet_NaN();
    double median = std::numeric_limits<double>::quiet_NaN();
    double mean = std::numeric_limits<double>::quiet_NaN();
    double cv_percent = std::numeric_limits<double>::quiet_NaN();
};

Stats compute_stats(std::vector<double> values) {
    Stats s;
    if (values.empty())
        return s;
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    s.min = values.front();
    s.median = (n % 2 != 0) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    double sum = 0.0;
    for (const double v : values)
        sum += v;
    s.mean = sum / static_cast<double>(n);
    double variance = 0.0;
    for (const double v : values)
        variance += (v - s.mean) * (v - s.mean);
    variance = n > 1 ? variance / static_cast<double>(n - 1) : 0.0;
    s.cv_percent = s.mean > 0.0 ? 100.0 * std::sqrt(variance) / s.mean : 0.0;
    return s;
}

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double gb_per_s(std::size_t bytes, double ms) {
    if (!(ms > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(bytes) / (ms * 1.0e6);
}

std::string human_bytes(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (bytes >= (1u << 20))
        out << static_cast<double>(bytes) / (1 << 20) << " MiB";
    else if (bytes >= (1u << 10))
        out << static_cast<double>(bytes) / (1 << 10) << " KiB";
    else
        out << bytes << " B";
    return out.str();
}

// ---------------------------------------------------------------- helpers --

void print_hresult(const std::string& operation, HRESULT hr) {
    std::cerr << operation << " failed: HRESULT 0x" << std::hex << static_cast<unsigned long>(hr)
              << std::dec << '\n';
}

// Waits until every command submitted so far has finished on the GPU.
class GpuFence {
  public:
    HRESULT initialize(ID3D11Device* device, ID3D11DeviceContext* context) {
        if (!device || !context)
            return E_INVALIDARG;
        D3D11_QUERY_DESC desc{};
        desc.Query = D3D11_QUERY_EVENT;
        context_ = context;
        return device->CreateQuery(&desc, &query_);
    }

    HRESULT wait() {
        if (!context_ || !query_)
            return E_UNEXPECTED;
        context_->End(query_.Get());
        context_->Flush();
        for (;;) {
            BOOL done = FALSE;
            const HRESULT hr = context_->GetData(query_.Get(), &done, sizeof(done), 0);
            if (FAILED(hr))
                return hr;
            if (hr == S_OK && done)
                return S_OK;
            std::this_thread::yield();
        }
    }

  private:
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11Query> query_;
};

std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Deterministic, cheap, reproducible data (no RNG state to carry around).
template <class T> void fill_pattern(std::vector<T>& values, std::uint32_t salt) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint32_t h = mix(static_cast<std::uint32_t>(i) + salt * 0x9E3779B9u);
        if constexpr (std::is_floating_point_v<T>) {
            values[i] = static_cast<T>(static_cast<float>(h >> 8) * (1.0f / 16777216.0f) * 2000.0f -
                                       1000.0f);
        } else {
            values[i] = static_cast<T>(h);
        }
    }
}

// Returns the first mismatching index, or count when identical (bit-exact).
template <class T>
std::size_t first_mismatch(const std::vector<T>& actual, const std::vector<T>& expected) {
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0)
            return i;
    return expected.size();
}

// ------------------------------------------------------------------- case --

struct Sample {
    double submit_ms = 0.0;
    double complete_ms = 0.0; // includes submit
    double readback_ms = 0.0;
};

struct CsvWriter {
    std::ofstream file;
    bool enabled() const {
        return file.is_open();
    }
};

int pick_iterations(const Options& options, std::size_t bytes_per_input) {
    if (options.iterations > 0)
        return options.iterations;
    constexpr std::size_t target_traffic = 256u << 20; // bytes moved per case
    const std::size_t traffic = std::max<std::size_t>(bytes_per_input * 4, 1);
    return static_cast<int>(std::clamp<std::size_t>(target_traffic / traffic, 7, 200));
}

int pick_warmup(const Options& options, int iterations) {
    if (options.warmup > 0)
        return options.warmup;
    return std::clamp(iterations / 5, 2, 10);
}

template <class T>
bool run_case(dxblas::Context& context, GpuFence& fence, const char* type_name, std::size_t count,
              const Options& options, CsvWriter& csv, bool& failed) {
    const std::size_t bytes = count * sizeof(T);
    const int iterations = pick_iterations(options, bytes);
    const int warmup = pick_warmup(options, iterations);

    std::vector<T> cpu_a(count), cpu_b(count), back_a(count), back_b(count);
    fill_pattern(cpu_a, 1);
    fill_pattern(cpu_b, 2);

    // Fresh objects: the first run below is the "cold" one (texture creation).
    dxblas::gpuMem<T> a(count), b(count), out(count);

    const auto run_once = [&](Sample& sample) -> HRESULT {
        a.assign(cpu_a.data(), count);
        b.assign(cpu_b.data(), count);
        // Untimed: make the destination resident and detect a readback that did nothing.
        std::memset(back_a.data(), 0xCD, bytes);
        std::memset(back_b.data(), 0xCD, bytes);

        const auto t0 = Clock::now();
        HRESULT hr = context.copy(a, b, out, count);
        const auto t1 = Clock::now();
        if (FAILED(hr))
            return hr;
        hr = fence.wait();
        const auto t2 = Clock::now();
        if (FAILED(hr))
            return hr;

        hr = context.copy_to_cpu_batch({
            {&a, back_a.data(), count},
            {&b, back_b.data(), count},
        });
        const auto t3 = Clock::now();
        if (FAILED(hr))
            return hr;

        sample.submit_ms = ms_between(t0, t1);
        sample.complete_ms = ms_between(t0, t2);
        sample.readback_ms = ms_between(t2, t3);
        return S_OK;
    };

    const auto validate = [&](const char* when) -> bool {
        const std::size_t bad_a = first_mismatch(back_a, cpu_a);
        const std::size_t bad_b = first_mismatch(back_b, cpu_b);
        if (bad_a == count && bad_b == count)
            return true;
        std::cerr << type_name << " count=" << count << " (" << when << "): MISMATCH in "
                  << (bad_a != count ? "A" : "B") << " at element "
                  << (bad_a != count ? bad_a : bad_b) << '\n';
        return false;
    };

    const std::string label = std::string(type_name) + " count=" + std::to_string(count);

    // Cold sample (single, validated, never averaged).
    Sample cold;
    HRESULT hr = run_once(cold);
    if (FAILED(hr)) {
        print_hresult(label + " cold run", hr);
        failed = true;
        return false;
    }
    if (!validate("cold")) {
        failed = true;
        return false;
    }

    for (int i = 0; i < warmup; ++i) {
        Sample discard;
        hr = run_once(discard);
        if (FAILED(hr)) {
            print_hresult(label + " warm-up", hr);
            failed = true;
            return false;
        }
    }

    std::vector<double> submit, complete, readback, roundtrip;
    submit.reserve(iterations);
    complete.reserve(iterations);
    readback.reserve(iterations);
    roundtrip.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        Sample s;
        hr = run_once(s);
        if (FAILED(hr)) {
            print_hresult(label + " measured run", hr);
            failed = true;
            return false;
        }
        submit.push_back(s.submit_ms);
        complete.push_back(s.complete_ms);
        readback.push_back(s.readback_ms);
        roundtrip.push_back(s.complete_ms + s.readback_ms);
        if (csv.enabled()) {
            csv.file << type_name << ',' << count << ',' << bytes << ",measured," << i << ','
                     << s.submit_ms << ',' << s.complete_ms << ',' << s.readback_ms << '\n';
        }
    }
    if (!validate("last iteration")) {
        failed = true;
        return false;
    }
    if (csv.enabled()) {
        csv.file << type_name << ',' << count << ',' << bytes << ",cold,0," << cold.submit_ms << ','
                 << cold.complete_ms << ',' << cold.readback_ms << '\n';
    }

    const Stats st_submit = compute_stats(submit);
    const Stats st_complete = compute_stats(complete);
    const Stats st_readback = compute_stats(readback);
    const Stats st_roundtrip = compute_stats(roundtrip);

    const std::size_t moved = bytes * 2; // two inputs each way
    std::cout << std::fixed << std::setprecision(3) << std::right << std::setw(12) << count
              << std::setw(11) << human_bytes(bytes) << std::setw(5) << iterations << std::setw(10)
              << cold.complete_ms << std::setw(10) << st_submit.median << std::setw(10)
              << st_complete.median << std::setw(10) << st_complete.min << std::setw(8)
              << std::setprecision(2) << gb_per_s(moved, st_complete.median) << std::setprecision(3)
              << std::setw(10) << st_readback.median << std::setw(10) << st_readback.min
              << std::setw(8) << std::setprecision(2) << gb_per_s(moved, st_readback.median)
              << std::setprecision(3) << std::setw(10) << st_roundtrip.median
              << std::setprecision(1) << std::setw(7) << st_complete.cv_percent << std::setw(7)
              << st_readback.cv_percent << '\n';
    return true;
}

template <class T>
bool run_type(dxblas::Context& context, GpuFence& fence, const char* type_name,
              const Options& options, CsvWriter& csv) {
    std::cout << "\n=== " << type_name << " (" << sizeof(T) << " byte elements) ===\n"
              << std::right << std::setw(12) << "elements" << std::setw(11) << "bytes/in"
              << std::setw(5) << "n" << std::setw(10) << "cold ms" << std::setw(10) << "submit"
              << std::setw(10) << "up med" << std::setw(10) << "up min" << std::setw(8) << "up GB/s"
              << std::setw(10) << "rb med" << std::setw(10) << "rb min" << std::setw(8) << "rb GB/s"
              << std::setw(10) << "trip med" << std::setw(7) << "upCV%" << std::setw(7) << "rbCV%"
              << '\n'
              << std::string(138, '-') << '\n';

    // Sizes cover: tiny, a partial single row, row boundaries (16384 is the texture
    // row width), partial last rows, and sizes from 1 MiB-ish to 64 MiB (float).
    std::vector<std::size_t> sizes = {1,     5,     255,    4096,      4097,      16384,
                                      16385, 65536, 262144, 1'000'003, 4'194'304, 16'777'216};
    if (options.large)
        sizes.push_back(67'108'864);

    bool failed = false;
    for (const std::size_t count : sizes) {
        if (!run_case<T>(context, fence, type_name, count, options, csv, failed))
            return false;
    }
    return !failed;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage();
        return 2;
    }

    dxblas::Context context;
    HRESULT hr = context.initialize();
    if (FAILED(hr)) {
        print_hresult("Context::initialize", hr);
        return 1;
    }

    // Native device is only used for the completion fence and the adapter name.
    dxblas::NativeD3D11 native{};
    hr = dxblas::get_native_d3d11(context, native);
    if (FAILED(hr)) {
        print_hresult("get_native_d3d11", hr);
        return 1;
    }
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> immediate;
    device.Attach(native.device); // get_native_d3d11 returns AddRef'd pointers
    immediate.Attach(native.context);

    GpuFence fence;
    hr = fence.initialize(device.Get(), immediate.Get());
    if (FAILED(hr)) {
        print_hresult("CreateQuery(EVENT)", hr);
        return 1;
    }

    std::string adapter = "unknown";
    {
        ComPtr<IDXGIDevice> dxgi_device;
        ComPtr<IDXGIAdapter> dxgi_adapter;
        DXGI_ADAPTER_DESC desc{};
        if (SUCCEEDED(device.As(&dxgi_device)) &&
            SUCCEEDED(dxgi_device->GetAdapter(&dxgi_adapter)) &&
            SUCCEEDED(dxgi_adapter->GetDesc(&desc)))
            adapter = dxblas::detail::adapter_name_utf8(desc.Description);
    }

    CsvWriter csv;
    if (!options.csv_path.empty()) {
        csv.file.open(options.csv_path, std::ios::out | std::ios::trunc);
        if (!csv.enabled()) {
            std::cerr << "Cannot open CSV file: " << options.csv_path << '\n';
            return 1;
        }
        csv.file << std::fixed << std::setprecision(6)
                 << "type,elements,bytes_per_input,kind,iteration,"
                    "upload_submit_ms,upload_complete_ms,readback_ms\n";
    }

    std::cout
        << "dxblas_gpumem_roundtrip\n"
        << "Adapter: " << adapter << '\n'
        << "Phases: submit = Context::copy returns; up = copy + GPU completion (EVENT query);\n"
        << "        rb = copy_to_cpu_batch of both inputs with the GPU idle; trip = up + rb.\n"
        << "Throughput counts both inputs (2 x bytes/in), decimal GB/s, from the median.\n"
        << "cold = first copy on fresh gpuMem objects (includes texture creation); all times in "
           "ms.\n"
        << "CV% = coefficient of variation; values above ~10% mean the numbers are noisy.\n"
        << "Iterations per case are adaptive unless --iterations is given; every run is\n"
        << "validated bit-exactly on the cold run and on the last measured iteration.\n";

    bool ok = true;
    const auto wants = [&](const char* name) {
        return options.type == "all" || options.type == name;
    };
    if (ok && wants("float32"))
        ok = run_type<float>(context, fence, "float32", options, csv);
    if (ok && wants("int32"))
        ok = run_type<std::int32_t>(context, fence, "int32", options, csv);
    if (ok && wants("int16"))
        ok = run_type<std::int16_t>(context, fence, "int16", options, csv);
    if (ok && wants("uint8"))
        ok = run_type<std::uint8_t>(context, fence, "uint8", options, csv);

    std::cout << (ok ? "\nAll GPU memory roundtrip scenarios passed.\n"
                     : "\nRoundtrip FAILED (see messages above).\n");
    return ok ? 0 : 3;
}