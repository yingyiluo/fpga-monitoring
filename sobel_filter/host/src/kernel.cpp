#include <stdlib.h>
#include <omp.h>
#include "defines.h"

#define VECTORIZE_NUM 16

static const int Gx[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
static const int Gy[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};

void sobel_cpu(unsigned int *frame_in, unsigned int *frame_out, int iterations, unsigned int threshold)
{
    int count;
    #pragma omp parallel for private(count) shared(frame_in, frame_out, iterations, threshold)
//  for (count = -(2 * COLS + 2 + VECTORIZE_NUM); count < iterations; count += VECTORIZE_NUM) {
    for (count = 0; count < iterations; count += VECTORIZE_NUM) {
        int k;
        for (k = 0; k < VECTORIZE_NUM; k++) {
            int x_dir = 0;
            int y_dir = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    if (count + i * COLS + j + k < iterations) {
                        unsigned int pixel = frame_in[count + i * COLS + j + k];
                        unsigned int b = pixel & 0xff;
                        unsigned int g = (pixel >> 8) & 0xff;
                        unsigned int r = (pixel >> 16) & 0xff;

                        unsigned int luma = r * 66 + g * 129 + b * 25;
                        luma = (luma + 128) >> 8;
                        luma += 16;

                        x_dir += luma * Gx[i][j];
                        y_dir += luma * Gy[i][j];
                    } else {
                        x_dir += 16 * Gx[i][j];
                        y_dir += 16 * Gy[i][j];
                    }
                }
            int temp = abs(x_dir) + abs(y_dir);
            if (count + k < iterations) {
                if (temp > threshold)
                    frame_out[count + k] = 0xffffff;
                else
                    frame_out[count + k] = 0;
            }
        }
    }
}
