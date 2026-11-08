// Row-major FP32 GEMM using flattened gpuMem texture storage. The source
// arrays may have texture widths unrelated to the logical matrix dimensions.
Texture2D<float> matrix_a : register(t0);
Texture2D<float> matrix_b : register(t1);
cbuffer GemmConstants : register(b0)
{
    uint M;
    uint N;
    uint K;
    uint AWidth;
    uint BWidth;
    uint OutWidth;
    uint Padding0;
    uint Padding1;
};
struct PSInput { float4 position : SV_Position; };
float main(PSInput input) : SV_Target
{
    const uint outputIndex = (uint)input.position.y * OutWidth + (uint)input.position.x;
    if (outputIndex >= M * N) return 0.0f;
    const uint row = outputIndex / N;
    const uint column = outputIndex - row * N;
    float accumulator = 0.0f;
    [loop]
    for (uint inner = 0; inner < K; ++inner)
    {
        const uint aIndex = row * K + inner;
        const uint bIndex = inner * N + column;
        const float av = matrix_a.Load(int3((int)(aIndex % AWidth), (int)(aIndex / AWidth), 0));
        const float bv = matrix_b.Load(int3((int)(bIndex % BWidth), (int)(bIndex / BWidth), 0));
        accumulator += av * bv;
    }
    return accumulator;
}
