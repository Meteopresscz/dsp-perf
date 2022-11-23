#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <volk/volk.h>
#include <immintrin.h>

#define SAMPLES 30000
#define ITERATIONS 10000

/* timing measurements */
struct timespec start, end;
#define GIGA 1000000000L
static uint64_t meas() {
  clock_gettime(CLOCK_REALTIME, &end);
  uint64_t diff = (GIGA*(end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec);
  memcpy(&start, &end, sizeof(struct timespec));
  return diff;
}

/* malloc wrapper function, either aligned or default malloc */
void* xmalloc(size_t s) {
  //return malloc(s);
  return aligned_alloc(64, s);
}

float lut[4096];
void prepare_lut() {
  for (int i=0; i<4096; i++) {
    int16_t nr = ((i+2048) % 4096) - 2048;
    lut[i]=(float)nr;
    //printf("lut[%i] = %f\n", i, lut[i]);
  }
}

int main() {

  /* read input */
  FILE * fi = fopen("input", "rb");

  int16_t * inbuf = xmalloc(8*SAMPLES);
  fread(inbuf, 8*SAMPLES, 1, fi);

  float * outbuf1 = xmalloc(8*SAMPLES);
  float * outbuf2 = xmalloc(8*SAMPLES);

  prepare_lut();

  meas();

  /* perform the main loop */
  for(int benchmark = 0; benchmark<ITERATIONS; benchmark++) {

#ifdef NAIVE
#pragma message "Using NAIVE implementation"
    for(int i = 0; i<4*SAMPLES; i++) {

      // fix sign-extend
      int16_t raw = inbuf[i];
      raw = (raw & 0xEFFF) | ((raw & 0xE000)>>1);

      // make float from int16
      float raw_f = (float)raw;

      // decide where to put it - horizontal or vertical buffer
      int imod = i % 4;
      if(imod < 2) {
        int idx = i/2 + imod;
        outbuf1[idx] = raw_f;
      } else {
        int idx = i/2 + imod - 3;
        outbuf2[idx] = raw_f;
      }
    }
#endif
#ifdef UNROLL
#pragma message "Using UNROLL implementation"
#pragma GCC ivdep
    for(int i = 0; i<4*SAMPLES; i+=4) {
      int16_t raw0 = inbuf[i+0];
      int16_t raw1 = inbuf[i+1];
      int16_t raw2 = inbuf[i+2];
      int16_t raw3 = inbuf[i+3];

      raw0 = (raw0 & 0xEFFF) | ((raw0 & 0xE000)>>1);
      raw1 = (raw1 & 0xEFFF) | ((raw1 & 0xE000)>>1);
      raw2 = (raw2 & 0xEFFF) | ((raw2 & 0xE000)>>1);
      raw3 = (raw3 & 0xEFFF) | ((raw3 & 0xE000)>>1);

      int i2 = i >> 1;

      outbuf1[i2    ] = (float)raw0;
      outbuf1[i2 + 1] = (float)raw1;

      outbuf2[i2    ] = (float)raw2;
      outbuf2[i2 + 1] = (float)raw3;
    }
#endif
#ifdef LUT
    float *op1=outbuf1;
    float *op2=outbuf2;
    int16_t *ip=inbuf;
    while (ip<inbuf+4*SAMPLES) {
      *op1++=lut[*ip++ & 0x0fff];
      *op1++=lut[*ip++ & 0x0fff];
      *op2++=lut[*ip++ & 0x0fff];
      *op2++=lut[*ip++ & 0x0fff];
    }
#endif
#ifdef X64
    for (int i = 0; i < SAMPLES*4; i += 4) {
      uint64_t* raw = (uint64_t*)&inbuf[i + 0];
#define META_MASK 0x1000100010001000ULL
      uint64_t mask = (*raw >> 1) & META_MASK;
      uint64_t mask2 = ~META_MASK;

      //printf("%16lX %16lX %16lX\n", mask, mask2, *raw);

      *raw &= mask2;
      //printf("%16lX\n", *raw);
      *raw |= mask;
      //printf("%16lX\n\n", *raw);

      const int i2 = i >> 1;

      outbuf1[i2] = (float)inbuf[i + 0];
      outbuf1[i2 + 1] = (float)inbuf[i + 1];

      outbuf2[i2] = (float)inbuf[i + 2];
      outbuf2[i2 + 1] = (float)inbuf[i + 3];
    }
#endif
#ifdef SSE
#pragma message "Using SSE4.1 implementation"
    __m128i EFFF = _mm_set1_epi16(0xEFFF);
    __m128i E000 = _mm_set1_epi16(0xE000);

    for(int i = 0; i<4*SAMPLES; i+=8) {

      // load 8 items to r
      __m128i r = _mm_stream_load_si128((void*)(inbuf + i));

      // fix meta/sign extend: perform r = (r & 0xEFFF) | ((r & 0xE000)>>1)
      __m128i rR = _mm_and_si128(EFFF, r);
      __m128i rL = _mm_and_si128(E000, r);
      rL = _mm_srai_epi16(rL, 1);
      r = _mm_or_si128(rR, rL);

      // convert low 4 elements (4xint16) to 4xint32
      __m128i e = _mm_cvtepi16_epi32(r);
      // convert to 4xfloat
      __m128 f = _mm_cvtepi32_ps(e);

      int i2 = i >> 1;

      // store result
      _mm_storel_pi((void*)(outbuf1+i2), f);
      _mm_storeh_pi((void*)(outbuf2+i2), f);

      // do the same with high 4 elements
      e = _mm_cvtepi16_epi32(_mm_bsrli_si128(r,8));
      f = _mm_cvtepi32_ps(e);

      i2 += 2;
      _mm_storel_pi((void*)(outbuf1+i2), f);
      _mm_storeh_pi((void*)(outbuf2+i2), f);
    }
#endif
#ifdef AVX
#pragma message "Using AVX2 implementation"
    __m128i EFFF = _mm_set1_epi16(0xEFFF);
    __m128i E000 = _mm_set1_epi16(0xE000);

    for(int i = 0; i<4*SAMPLES; i+=8) {

      // load 8 items to r
      __m128i r = _mm_stream_load_si128((void*)(inbuf + i));

      // fix meta/sign extend: perform r = (r & 0xEFFF) | ((r & 0xE000)>>1)
      __m128i rR = _mm_and_si128(EFFF, r);
      __m128i rL = _mm_and_si128(E000, r);
      rL = _mm_srai_epi16(rL, 1);
      r = _mm_or_si128(rR, rL);

      // convert 8xint16 to 8xint32
      __m256i e = _mm256_cvtepi16_epi32(r);
      // convert to 8xfloat, cast to 64bit ints for the next permute step
      __m256i f = _mm256_castps_si256( _mm256_cvtepi32_ps(e) );

      // deinterleave polarizations
      f = _mm256_permute4x64_epi64(f, 0b11011000);

      int i2 = i >> 1;

      _mm256_storeu2_m128i((void*)(outbuf2+i2), (void*)(outbuf1+i2), f);
    }
#endif
  }

  /* compute time it took */
  double delta = meas();
  double usec = delta/1000;
  printf("%i iterations in %f s = %f us/iteration\n", ITERATIONS, usec/1000000, usec/ITERATIONS);

  /* write out result */
  fclose(fi);
  fi = fopen("pol1.c.bin", "wb");
  fwrite(outbuf1, 8*SAMPLES, 1, fi);
  fclose(fi);
  fi = fopen("pol2.c.bin", "wb");
  fwrite(outbuf2, 8*SAMPLES, 1, fi);
  fclose(fi);

}

