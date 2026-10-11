#include <dxblas/dxblas.hpp>

#include <iostream>
#include <vector>

namespace {
bool check(HRESULT hr, const char* operation)
{
    if (SUCCEEDED(hr)) return true;
    std::cerr << operation << " failed (HRESULT 0x" << std::hex
              << static_cast<unsigned long>(hr) << std::dec << ")\n";
    return false;
}
}

int main()
{
    dxblas::Context context;
    if (!check(context.initialize(), "Context::initialize")) return 1;

    const std::vector<float> cpu_a{1, 2, 3, 4, 5};
    const std::vector<float> cpu_b{10, 20, 30, 40, 50};
    dxblas::gpuMem<float> a(cpu_a), b(cpu_b), out(cpu_a.size());

    // Upload once. The following operations use the textures owned by gpuMem.
    if (!check(context.copy(a, b, out, cpu_a.size()), "Context::copy")) return 2;
    if (!check(context.sum(a, b, out, cpu_a.size()), "Context::sum(gpuMem)")) return 3;
    std::vector<float> result(cpu_a.size());
    if (!check(context.copy_to_cpu(out, result.data(), result.size()), "copy_to_cpu")) return 4;

    std::cout << "gpuMem sum: ";
    for (float value : result) std::cout << value << ' ';
    std::cout << '\n';

    // Re-upload because copy() intentionally releases the CPU staging payload.
    dxblas::gpuMem<float> ma(cpu_a), mb(cpu_b), mout(cpu_a.size());
    if (!check(context.copy(ma, mb, mout, cpu_a.size()), "Context::copy for mul")) return 5;
    if (!check(context.mul(ma, mb, mout, cpu_a.size()), "Context::mul(gpuMem)")) return 6;
    if (!check(context.copy_to_cpu(mout, result.data(), result.size()), "copy_to_cpu mul")) return 7;
    std::cout << "gpuMem mul: ";
    for (float value : result) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
}
