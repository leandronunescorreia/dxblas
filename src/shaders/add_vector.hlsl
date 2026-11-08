// Row-major matrix + vector addition (broadcasting vector across all matrix rows)
// using D3D11 pixel shader and Texture2D render targets.
#ifndef DTYPE
#define DTYPE float
#endif

Texture2D<DTYPE> input_matrix : register(t0);
Texture2D<DTYPE> input_vector : register(t1);

cbuffer AddVectorConstants : register(b0)
{
    uint Rows;
    uint Cols;
    uint MatWidth;
    uint VecWidth;
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
    const uint outputIndex = (uint)input.position.y * MatWidth + (uint)input.position.x;
    if (outputIndex >= Rows * Cols)
        return (DTYPE)0;

    const uint row = outputIndex / Cols;
    const uint col = outputIndex - row * Cols;

    const int2 mat_pixel = int2((int)(outputIndex % MatWidth), (int)(outputIndex / MatWidth));
    const int2 vec_pixel = int2((int)(col % VecWidth), (int)(col / VecWidth));

    return input_matrix.Load(int3(mat_pixel, 0)) + input_vector.Load(int3(vec_pixel, 0));
}
