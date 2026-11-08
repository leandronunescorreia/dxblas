// dxBLAS D3D11 pixel shader. The shader manager specializes DTYPE and the
// output signature at runtime by compiling concrete variants.
#ifndef DTYPE
#define DTYPE float
#endif

Texture2D<DTYPE> input_a : register(t0);
Texture2D<DTYPE> input_b : register(t1);

struct PSInput
{
    float4 position : SV_Position;
};

#if defined(UNSIGNED_INTEGER_OUTPUT)
uint main(PSInput input) : SV_Target
{
    int2 pixel = int2(input.position.xy);
    return input_a.Load(int3(pixel, 0)) + input_b.Load(int3(pixel, 0));
}
#elif defined(SIGNED_INTEGER_OUTPUT)
int main(PSInput input) : SV_Target
{
    int2 pixel = int2(input.position.xy);
    return input_a.Load(int3(pixel, 0)) + input_b.Load(int3(pixel, 0));
}
#else
DTYPE main(PSInput input) : SV_Target
{
    int2 pixel = int2(input.position.xy);
    return input_a.Load(int3(pixel, 0)) + input_b.Load(int3(pixel, 0));
}
#endif
