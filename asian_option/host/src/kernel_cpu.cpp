#include <omp.h>
#include <cmath>
#include "CL/opencl.h"

typedef cl_float2 float2;

// Mersenne twister constants
#define MT_M 397 
#define MT_N 624 
#define MATRIX_A   0x9908b0dfUL 
#define UPPER_MASK 0x80000000UL 
#define LOWER_MASK 0x7fffffffUL 

// Used to ensure that the uniformly generated random numbers are in the range (0,1)
#define CLAMP_ZERO 0x1.0p-126f 
#define CLAMP_ONE  0x1.fffffep-1f

// In this implementations, we will create vectors of 64 random numbers per clock cycle
// Each of these random numbers will be used to simulate the movement of a stock price
// for a single timestep. In this case, we are simulating 64 timesteps per clock cycle.
#define VECTOR 64 
#define VECTOR_DIV2 32 
#define VECTOR_DIV4 16 

static const float pi = 3.14159265;

struct mersenne_twister {
#define MT_(i) mt[(mt_base_ + (i)) % MT_N]
  mersenne_twister(unsigned int seed) {
    unsigned int ival[VECTOR];
    unsigned int ival_base = 0;
    unsigned int state = seed;
    int shift = (MT_N / VECTOR + 1) * VECTOR - MT_N;
    for (int i = 0; i < VECTOR; i++)
      ival[i] = seed;
    for (int n = 0; n < MT_N; n++) {
      ival[ival_base % VECTOR]= state;
      ival_base++;
      state = (1812433253U * (state ^ (state >> 30)) + n) & 0xffffffffUL;
      if (n % VECTOR == 47) {
        int mt_base = (n / VECTOR) * VECTOR - shift;
        for (int i = 0; i < VECTOR; i++) {
          if (mt_base + i >= 0)
            mt[mt_base + i] = ival[(ival_base + i) % VECTOR];
        }
      }
    }
    mt_base_ = 0;
  }

inline float *next() {
    unsigned int y[VECTOR];
    for (int i = 0; i < VECTOR; i++) {
      y[i] = (MT_(i) & UPPER_MASK) | (MT_(i+1) & LOWER_MASK);
      y[i] = MT_(i+MT_M) ^ (y[i] >> 1) ^ (y[i] & 0x1UL ? MATRIX_A : 0x0UL);
    }

    for (int i = 0; i < VECTOR; i++) {
      MT_(i) = y[i];


      y[i] ^= (y[i] >> 11);
      y[i] ^= (y[i] << 7) & 0x9d2c5680UL;
      y[i] ^= (y[i] << 15) & 0xefc60000UL;
      y[i] ^= (y[i] >> 18);

      U[i] = (float)y[i] / 4294967296.0f;
      if (U[i] == 0.0f) U[i] = CLAMP_ZERO; 
      if (U[i] == 1.0f) U[i] = CLAMP_ONE;
    }

    mt_base_ += VECTOR;
    return U;
  }

  unsigned mt_base_;
  unsigned int mt[MT_N];
  float U[VECTOR];
};

inline float2 box_muller(float a, float b)
{
   float radius = sqrt(-2.0f * log(a));
   float angle = 2.0f*b;
   float2 result;
   angle *= pi;
   result.x = radius*cos(angle);
   result.y = radius*sin(angle);
   return result;
}

double kernel_cpu(int nthreads, int m, int n, float drift, float vol, float S_0, float K) {
  double sum;
  #pragma omp parallel reduction(+:sum)
  {
    float *U;
    float Z[VECTOR];
    float U0[VECTOR_DIV4];
    float U1[VECTOR_DIV4];
    float U2[VECTOR_DIV4];
    float U3[VECTOR_DIV4];
    mersenne_twister rng(777 + omp_get_thread_num());
    #pragma omp for
    for (int tid = 0; tid < nthreads; tid++) {
      for (int path = 0; path < m; path++) {
        float S = S_0;
        float arithmetic_average = 0.0f;
        for (int t_i = 0; t_i < n/VECTOR; t_i++) {
          U = rng.next();

          for (int i = 0; i < VECTOR_DIV2; i++) {
            float2 z = box_muller(U[2*i], U[2*i+1]);
            Z[2*i] = z.x;
            Z[2*i+1] = z.y;
          }

          for (int i = 0; i < VECTOR; i++) {
            float gauss_rnd = Z[i];
            S *= drift * exp(vol * gauss_rnd);
            arithmetic_average += S;
          }
        }
        arithmetic_average /= (float) n;

        float call_value = arithmetic_average - K;
        if (call_value > 0.0f)
          sum += call_value;
      }
    }
  }
  return sum;
}
