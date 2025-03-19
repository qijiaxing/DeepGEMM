# DeepGEMM 1d1d kernel

This is a fp8 GEMM kernel with 1d1d scaling, modified from DeepGEMM's 1d2d gemm kernel.

It performs GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling [M, K] and 1x128 RHS scaling [N, K].
So RHS tensor shape is [N, K], its scale tensor shape is [N, K//128].
This kernel can be used for WGRAD calculation during training.

Related files are 
* `deep_gemm/include/deep_gemm/fp8_gemm_1d1d.cuh`
* `deep_gemm/jit_kernels/gemm_1d1d.py`
* `tests/test_1d1d.py`

This repo is just my own CUDA optimization practice.
Currently this 1d1d kernel's perf is slower than DeepGEMM's 1d2d kernel.
Any suggestions or further optimization are welcomed!

Perf on a H800 PCIe GPU (TFLOPS)
```
M	           N, K	  1d1d	1d2d
4096	2112, 7168	  628	 681
4096	24576, 1536	  537	 631
4096	32768, 512	  375	 445
4096	7168, 16384	  703	 638
4096	4096, 7168	  670	 740
4096	7168, 2048	  540	 667
```

TODO:
* Validate if it works for grouped GEMM.
