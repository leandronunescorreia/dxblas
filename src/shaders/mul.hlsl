// dxBLAS D3D11 classic graphics pipeline: element-wise multiplication.
#ifndef DTYPE
#define DTYPE float
#endif

Texture2D<DTYPE> input_a : register(t0);
Texture2D<DTYPE> input_b : register(t1);

struct PSInput
{
    float4 position : SV_Position;
};

#ifdef INTEGER_OUTPUT
float4 main(PSInput input) : SV_Target
{
    int2 pixel = int2(input.position.xy);
    uint value = input_a.Load(int3(pixel, 0)) * input_b.Load(int3(pixel, 0));
    return float4(
        float(value & 0xffu),
        float((value >> 8) & 0xffu),
        float((value >> 16) & 0xffu),
        float((value >> 24) & 0xffu)) / 255.0;
}
#else
DTYPE main(PSInput input) : SV_Target
{
    int2 pixel = int2(input.position.xy);
    return input_a.Load(int3(pixel, 0)) * input_b.Load(int3(pixel, 0));
}
#endif
