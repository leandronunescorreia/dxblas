// Row-major matrix transpose using D3D11 pixel shader and Texture2D render targets.
// No compute shaders are used; execution follows the classic graphics pipeline.
#ifndef DTYPE
#define DTYPE float
#endif

Texture2D<DTYPE> input_matrix : register(t0);

cbuffer TransposeConstants : register(b0)
{
    uint Rows;
    uint Cols;
    uint InWidth;
    uint OutWidth;
};

struct PSInput
{
    float4 position : SV_Position;
};

#if defined(UNSIGNED_INTEGER_OUTPUT)
uint main(PSInput input) : SV_Target
#elif defined(SIGNED_INTEGER_OUTPUT)
int main(PSInput input) : SV_Target
#else
DTYPE main(PSInput input) : SV_Target
#endif
{
    const uint outputIndex = (uint)input.position.y * OutWidth + (uint)input.position.x;
    if (outputIndex >= Rows * Cols)
        return (DTYPE)0;

    const uint out_row = outputIndex / Rows;
    const uint out_col = outputIndex - out_row * Rows;

    const uint in_row = out_col;
    const uint in_col = out_row;
    const uint in_index = in_row * Cols + in_col;

    const int2 in_pixel = int2((int)(in_index % InWidth), (int)(in_index / InWidth));
    return input_matrix.Load(int3(in_pixel, 0));
}
