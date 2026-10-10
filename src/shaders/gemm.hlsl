// FP32 GEMM implemented with a pixel shader and render-to-texture.
// No Direct3D compute shader, UAV, or structured buffer is used.
Texture2D<float> matrix_a : register(t0); // M x K, row-major
Texture2D<float> matrix_b : register(t1); // K x N, row-major

cbuffer GemmConstants : register(b0)
{
    uint M;
    uint N;
    uint K;
    uint Padding;
};

struct PSInput
{
    float4 position : SV_Position;
};

float main(PSInput input) : SV_Target
{
    const uint column = (uint)input.position.x;
    const uint row = (uint)input.position.y;
    if (row >= M || column >= N)
        return 0.0f;

    float accumulator = 0.0f;
    [loop]
    for (uint inner = 0; inner < K; ++inner)
    {
        const float a = matrix_a.Load(int3((int)inner, (int)row, 0));
        const float b = matrix_b.Load(int3((int)column, (int)inner, 0));
        accumulator += a * b;
    }
    return accumulator;
}
