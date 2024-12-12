#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#define BLOCKSIZE 1024
typedef struct {
  union { int width, w; };
  union { int height, h; };
  union { uint8_t *pixels, *p; };
  union { uint8_t bytesPerPixel, bpp; };
} sprite;

extern "C" int loadFile(sprite *sprite, const char *filename);

extern "C" bool writeFile(sprite *sprite, const char *writeFile);

//NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
__global__ void RGBToGreyscale(int* pixels_rgb_arr, int* output, int size) { //takes in arr with rgb values
   int cur_index = blockIdx.x * blockDim.x + threadIdx.x;
   if (cur_index%3 == 0){
    int greyVal = 0.114*pixels_rgb_arr[cur_index] + 0.587 * pixels_rgb_arr[cur_index + 1] + 0.299 * pixels_rgb_arr[cur_index + 2];
    //printf("curindex: %d ", cur_index);
    for (int i = cur_index; i < cur_index + 3; i++){
      output[i] = greyVal;
    }
   }
}

int main(int argc, char *argv[]) {
  static sprite sprite;
  int pixels_read = loadFile(&sprite, argv[1]);
  int size = pixels_read*3;
  int *x, *y;
  cudaMallocManaged(&x, sizeof(int) * size);
  cudaMallocManaged(&y, sizeof(int) * size);
  printf("%d\n", pixels_read);

  for (int i = 0; i < pixels_read; i++) {
    int pxIdx = i*sprite.bpp;
    x[pxIdx] = sprite.p[pxIdx];
    x[pxIdx + 1] = sprite.p[pxIdx+1];
    x[pxIdx + 2] = sprite.p[pxIdx+2]; 
//    printf("R: %d,   G: %d,   B: %d\n", sprite.p[pxIdx+2], sprite.p[pxIdx+1], sprite.p[pxIdx]);
  }
//  printf("\n");

  int blockNum = ceil(size/BLOCKSIZE);
  RGBToGreyscale<<<blockNum, BLOCKSIZE>>>(x, y, size);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaDeviceSynchronize();
  int count = 0;
  for (int i = 0; i < size; i++) {
    if (y[i] == 0){
       count += 1;
    }
    printf("%d ", y[i]);
  }
  printf("%d ", count);
  printf("\n");

  bool wrote = writeFile(&sprite, "greyscale_test.bmp");

  //printf("%d ", sprite.bpp);
//  printf("\n");
  free(sprite.p);
  cudaFree(x);
  cudaFree(y);

  return 0;
}
