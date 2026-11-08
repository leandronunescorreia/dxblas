// Fused FP32 GEMM + Transpose B + Bias Addition:
// Computes Out[M, N] = A[M, K] * B[N, K]^T + c[N] in a single pixel shader pass.
// Matrix B is provided in its original (N x K) row-major orientation.
// Transposition of B is performed directly in the inner loop index calculation.
// No compute shaders are used; execution follows the classic graphics pipeline.

Texture2D<float> matrix_a : register(t0);
Texture2D<float> matrix_b : register(t1);
Texture2D<float> vector_c : register(t2);

cbuffer FusedGemmConstants : register(b0)
{
    uint M;
    uint N;
    uint K;
    uint AWidth;
    uint BWidth;
    uint CWidth;
    uint OutWidth;
    uint Padding;
};

struct PSInput
{
    float4 position : SV_Position;
};

float main(PSInput input) : SV_Target
{
    const uint outputIndex = (uint)input.position.y * OutWidth + (uint)input.position.x;
    if (outputIndex >= M * N)
        return 0.0f;

    const uint row = outputIndex / N;
    const uint column = outputIndex - row * N;

    float accumulator = 0.0f;
    [loop]
    for (uint inner = 0; inner < K; ++inner)
    {
        // A is M x K: element at (row, inner)
        const uint aIndex = row * K + inner;
        // B is N x K: element at (column, inner) which is B^T[inner, column]
        const uint bIndex = column * K + inner;

        const float av = matrix_a.Load(int3((int)(aIndex % AWidth), (int)(aIndex / AWidth), 0));
        const float bv = matrix_b.Load(int3((int)(bIndex % BWidth), (int)(bIndex / BWidth), 0));
        accumulator += av * bv;
    }

    // Bias vector c has length N: element at (column)
    const float cv = vector_c.Load(int3((int)(column % CWidth), (int)(column / CWidth), 0));
    return accumulator + cv;
}
