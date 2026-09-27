// dxBLAS Compute Shader: Element-wise Multiplication
// Computes output_buf[i] = input_a[i] * input_b[i]

#ifndef DTYPE
#define DTYPE float
#endif

Buffer<DTYPE> input_a : register(t0);
Buffer<DTYPE> input_b : register(t1);
RWBuffer<DTYPE> output_buf : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output_buf[id.x] = input_a[id.x] * input_b[id.x];
}
