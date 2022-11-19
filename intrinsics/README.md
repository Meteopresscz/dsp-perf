# Using SSE and AVX intrinsics

This repository demonstrates how to apply `result = (data & 0xEFFF) | ((data & 0xE000)>>1)` onto an array of int16\_t, convert to float and deinterleave the result into two buffers.

This is used when you use [bladeRF in MIMO mode](https://nuand.com/libbladeRF-doc/v2.2.1/group___s_t_r_e_a_m_i_n_g___f_o_r_m_a_t.html) and use the 13th bit for your own metadata.

There is a simple Python implementation, a naive C implementation, a hand-unrolled C implementation and hand-optimized SSE and AVX implementation.

Run `run.sh` to run the implementations and see the results.
