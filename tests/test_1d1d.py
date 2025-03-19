import sys
sys.path.append("./")

import numpy as np
import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor

DEBUG = False
torch.set_printoptions(profile="full")


def save_tensor(tensor, filename):
    print(f"Save tensor to file: {filename}")
    numpy_array = tensor.clone().to(torch.float32).cpu().numpy()
    np.savetxt(filename, numpy_array, fmt='%.3e')


def create_scales(N, K):
    # Create an empty tensor with shape [N, K]
    x = torch.empty(N, K, dtype=torch.float32, device="cuda")

    # Assign values to the tensor
    for i in range(N):
        for j in range(K):
            # x[i, j] = i * 100 + j + 1
            x[i, j] = i / 8 + 1

    return x


def create_tensor(dims, use_scale_ones=True):
    assert len(dims) == 2
    assert dims[1] % 128 == 0, f"K%128={dims[1]}%128={dims[1] % 128} must be 0!"

    qx = torch.ones(dims, device='cuda', dtype=torch.float32)  # [M, K]
    # init qx with fp8 values like 1, 0.5, 0.25, 0.125, so we avoid quantization error
    # value_list = [ 1.0 / 2**i for i in range(6)]
    # rows, cols = dims
    # qx = torch.tensor(
    #     [[random.choice(value_list) for _ in range(cols)] for _ in range(rows)],
    #     device="cuda", dtype=torch.float32
    # )

    if use_scale_ones:
        sx = torch.ones((dims[0], dims[1]//128), device='cuda', dtype=torch.float32)  # [M, K/128]
    else:
        sx = create_scales(dims[0], dims[1]//128)

    sx_expanded = sx.repeat_interleave(128, dim = 1)
    x = qx * sx_expanded

    # print(f"QX shape: {list(qx.shape)}")
    # print(f"QX first 2 rows: {qx[:2, :]}")
    # print(f"SX shape: {list(sx.shape)}")
    # print(f"X shape: {list(x.shape)}")
    # print(f"X first 2 rows: {x[:2, :]}")

    return qx.to(torch.float8_e4m3fn), sx, x.to(torch.bfloat16)


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    # Tensor shape: [M, K], scale shape: [M, K/128]
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_token_cast_to_fp8(y)

    # Save scale A into file
    scale_a = x_fp8[1].clone()
    filename = f"scale_a_mk.txt"
    numpy_array = scale_a.cpu().numpy()
    np.savetxt(filename, numpy_array, fmt='%.3e')

    # Save scale B into file
    scale_b = create_scales(n, int(k//128))
    filename = f"scale_b_nk.txt"
    numpy_array = scale_b.cpu().numpy()
    np.savetxt(filename, numpy_array, fmt='%4.f')
    y_fp8 = (y_fp8[0], scale_b)

    if DEBUG:
        print(f"[DEBUG] Before TMA aligned, scale A shape: {list(x_fp8[1].shape)}, stride: {(x_fp8[1].stride())}")
        print(x_fp8[1])

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    y_fp8 = (y_fp8[0], get_col_major_tma_aligned_tensor(y_fp8[1]))

    if DEBUG:
        print(f"[DEBUG ]After TMA aligned, scale A shape: {list(x_fp8[1].shape)}, stride: {(x_fp8[1].stride())}")
        print(x_fp8[1])

    return x_fp8, y_fp8, out, ref_out


def construct2(m, n, k):
    qx, sx, x = create_tensor((m, k), use_scale_ones=False)
    qy, sy, y = create_tensor((n, k), use_scale_ones=False)

    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8 = (qx, get_col_major_tma_aligned_tensor(sx))
    y_fp8 = (qy, get_col_major_tma_aligned_tensor(sy))

    return x_fp8, y_fp8, out, ref_out


def test_gemm() -> None:
    print('Testing GEMM 1D1D:')

    # for m in (64, 128, 4096):
    for m in (4096, ):
      for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:

        print(f"M, N, K: {m}, {n}, {k}")
        x_fp8, y_fp8, out, ref_out = construct2(m=m, n=n, k=k)

        deep_gemm.gemm_fp8_fp8_bf16_nt_1d1d(x_fp8, y_fp8, out)

        try:
            print(f"Test allclose for M, N, K: {m}, {n}, {k}")
            torch.testing.assert_close(out, ref_out)
        except AssertionError as e:
            print(f"M, N, K: {m}, {n}, {k}, All close Failed!")
            print(e)
            filename = f"res_diff_m{m}_n{n}_k{k}.txt"
            save_tensor(torch.abs(ref_out - out), filename)
        else:
            print(f"All close passed!\n")

        # noinspection PyShadowingNames
        # Construct new tensors every time to avoid L2 cache acceleration
        # construct2 is slow, so to move it out
        x_fp8, y_fp8, out, ref_out = construct2(m=m, k=k, n=n)
        def test_func():
            deep_gemm.gemm_fp8_fp8_bf16_nt_1d1d(x_fp8, y_fp8, out)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s\n')
    print()



if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    # test_m_grouped_gemm_contiguous()
