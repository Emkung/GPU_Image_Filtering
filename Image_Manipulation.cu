#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#define SIZE 250

typedef struct {
  uint8_t r, g, b;
} pixel;

typedef struct {
  union { int width, w; };
  union { int height, h; };
  union { pixel *pixels, *p; };
} sprite;


int loadFile(sprite *sprite, const char *filename){}

//NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
__global__ void RGB_To_Greyscale(int pixels) { //takes in return_v from c file
   int cur_index = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = 0; i < pixels_read; i++) {
       pixel pixel = sprite.p[i];
       printf("R: %d,   G: %d,   B: %d\n", pixel.r, pixel.g, pixel.b);
   }
}

int main() {

  int size = SIZE;

  float *x, *y, *z;
  cudaMallocManaged(&x, SIZE*sizeof(float) * size * size);
  cudaMallocManaged(&y, SIZE*sizeof(float) * size * size);
  cudaMallocManaged(&z, SIZE*sizeof(float) * size * size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      x[i * size + j] = 1; // x[i][j]
      y[i * size + j] = 1;
    }
  printf("\n");
  }

  MatrixMulOnDevice<<<1, 128>>>(x, y, z, size);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaDeviceSynchronize();

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      printf("%f ", z[i * size + j]);
      if (z[i * size + j] != size) {
        printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
      }
    }
    printf("\n");
  }

  cudaFree(x);
  cudaFree(y);
  cudaFree(z);

  double t1 = get_clock();
  printf("time per call: %f\n", t1 - t0);

  return 0;
}
