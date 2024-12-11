#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

typedef struct {
  uint8_t r, g, b;
} pixel;

typedef struct {
  union { int width, w; };
  union { int height, h; };
  union { uint8_t *pixels, *p; };
  union { uint8_t bytesPerPixel, bpp; };
} sprite;


extern "C" int loadFile(sprite *sprite, const char *filename);

extern "C" bool writeFile(sprite *sprite, const int depth, const char *writeFile);

//NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
__global__ void RGB_To_Greyscale(int* pixels_rgb_arr, int* output, int size) { //takes in arr with rgb values
   int cur_index = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = 0; i < pixels_read; i++) {
    //   pixel pixel = sprite.p[i];
      // printf("R: %d,   G: %d,   B: %d\n", pixel.r, pixel.g, pixel.b);
   }
}

int main(int argc, char *argv[]) {
  static sprite sprite;
  int pixels_read = loadFile(&sprite, argv[1]);
  int size = pixels_read*3;
  int *x, *y;
  cudaMallocManaged(&x, sizeof(size) * size);
  cudaMallocManaged(&x, sizeof(size) * size);
  printf("%d\n", pixels_read);

  for (int i = 0; i < pixels_read; i++) {
    int pxIdx = i*sprite.bpp;
    x[pxIdx] = sprite.p[pxIdx+2];
    x[pxIdx + 1] = sprite.p[pxIdx+1];
    x[pxIdx + 2] = sprite.p[pxIdx]; 
//    printf("R: %d,   G: %d,   B: %d\n", sprite.p[pxIdx+2], sprite.p[pxIdx+1], sprite.p[pxIdx]);
  }
//  printf("\n");

  //MatrixMulOnDevice<<<1, 128>>>(x, y, z, size);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  //cudaDeviceSynchronize();

  for (int i = 0; i < size; i++) {
      printf("%d ", x[i]);
  }
  printf("\n");
  //printf("%d ", sprite.bpp);
//  printf("\n");
  free(sprite.p);
  cudaFree(x);

  return 0;
}
