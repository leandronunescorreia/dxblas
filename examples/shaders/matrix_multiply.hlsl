// Tiled FP32 matrix multiplication: C = A * B, row-major matrices.
// Compile as Shader Model 5.0 for Direct3D 11.
#define TILE 16

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

cbuffer MatrixParams : register(b0)
{
    uint N;
    uint3 _padding;
};

groupshared float tileA[TILE][TILE];
groupshared float tileB[TILE][TILE];

[numthreads(TILE, TILE, 1)]
void main(
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint row = dispatchThreadId.y;
    const uint col = dispatchThreadId.x;
    const uint localRow = groupThreadId.y;
    const uint localCol = groupThreadId.x;

    float acc = 0.0f;
    const uint tiles = (N + TILE - 1) / TILE;

    for (uint t = 0; t < tiles; ++t)
    {
        const uint aCol = t * TILE + localCol;
        const uint bRow = t * TILE + localRow;

        tileA[localRow][localCol] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        tileB[localRow][localCol] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE; ++k)
            acc += tileA[localRow][k] * tileB[k][localCol];

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}
