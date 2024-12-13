#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#define BLOCKSIZE 1024

typedef struct {
  union { int width, w; }; // width of image
  union { int height, h; }; // height of image
  union { uint8_t *pixels, *p; }; // pixel bgr values (rgb, reversed)
  union { uint8_t bytesPerPixel, bpp; }; // number of bytes in each pixel (3 or 4)
} sprite;

/**
TODO insert __device__ method to alter greyscale method:
  submit a symbol (e.g. "NTSC") 
  use a switch to submit the greyscale calculation with the method selected
 */

// loads .bmp into provided sprite
extern "C" int loadFile(sprite *sprite, const char *filename);

// writes provided sprite into new bmp
extern "C" bool writeFile(sprite *sprite, const char *writeFile);

//NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
__global__ void RGBToGreyscale(uint8_t* pixels_rgb_arr, uint8_t* output, int size, int depth) { //takes in arr with rgb values
  int cur_index = depth * (blockIdx.x * blockDim.x +threadIdx.x);
  if (cur_index < size;){
    // math
    uint8_t greyVal = 0.114*pixels_rgb_arr[cur_index] // 0.114*b
      + 0.587 * pixels_rgb_arr[cur_index + 1] // 0.587*g
      + 0.299 * pixels_rgb_arr[cur_index + 2]; // 0.299*r

    // write
    for (int i = cur_index; i < cur_index + 3; i++){
      output[i] = greyVal; // inefficient writes to global but more efficient than a CPU still
     }
   }
}

int main(int argc, char *argv[]) {
  // initialize sprite
  static sprite sprite;
  int pixels_read = loadFile(&sprite, argv[1]); printf("%d\n", pixels_read);
  int size = pixels_read*sprite.bpp;

  // cuda malloc
  uint8_t *in_pixels, *out_pixels;
  cudaMalloc(&in_pixels, sizeof(uint8_t) * size);
  cudaMalloc(&out_pixels, sizeof(uint8_t) * size);
  cudaMemcpy(in_pixels, sprite.p, size, cudaMemcpyHostToDevice);

  // obsolete due to manual cudaMemcpy above. saved in case of bugs
  //for (int i = 0; i < pixels_read; i++) {
    //int pxIdx = i*sprite.bpp;
    //x[pxIdx] = sprite.p[pxIdx];
    //x[pxIdx + 1] = sprite.p[pxIdx+1];
    //x[pxIdx + 2] = sprite.p[pxIdx+2]; 
  //}

  // run kernel
  int blockNum = ceil(pixels_read/BLOCKSIZE);
  RGBToGreyscale<<<blockNum, BLOCKSIZE>>>(in_pixels, out_pixels, size, sprite.bpp);
  cudaDeviceSynchronize(); // IMMEDIATELY after opening the kernel >:|
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  int count = 0;
  for (int i = 0; i < size; i++) {
    if (y[i] == 0){ count += 1; } // count # of 0s
    printf("%d ", y[i]); // print output to make sure it looks right
  }
  printf("\n0s: %d\n", count);

  // write file out
  cudaMemcpy(sprite.p, out_pixels, size, cudaMemcpyDeviceToHost);
  bool wrote = writeFile(&sprite, "outputs/greyscale_test.bmp"); // TODO accept second CL arg

  // freedom!!
  free(sprite.p);
  cudaFree(in_d);
  cudaFree(out_d);

  return 0;
}
