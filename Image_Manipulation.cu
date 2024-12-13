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

/** loads .bmp into provided sprite */
extern "C" int loadFile(sprite *sprite, const char *filename);

/** writes provided sprite into new bmp */
extern "C" bool writeFile(sprite *sprite, const char *writeFile);

/**
TODO insert __device__ method to alter greyscale method:
  submit a symbol (e.g. "NTSC") 
  use a switch to submit the greyscale calculation with the method selected
 */

//NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
__global__ void RGBToGreyscale(uint8_t* pixels_rgb_arr, uint8_t* output, int size, int depth) { //takes in arr with rgb values
  int index = depth * (blockIdx.x*blockDim.x +threadIdx.x);

  if (index+depth <= size){
    
    // math
    uint8_t b = pixels_rgb_arr[index], g = pixels_rgb_arr[index+1], r = pixels_rgb_arr[index+2]; 
    uint8_t greyVal = 0.114*b + 0.587 * g + 0.299 * r;

    // write
    for (int i = index; i < index + 3; i++){
      output[i] = greyVal; // inefficient writes?
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
  int blockNum = ceil ( (1.*pixels_read) / BLOCKSIZE);
  RGBToGreyscale<<<blockNum, BLOCKSIZE>>>(in_pixels, out_pixels, size, sprite.bpp);
  cudaDeviceSynchronize(); // IMMEDIATELY after opening the kernel >:|
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  // write file out
  cudaMemcpy(sprite.p, out_pixels, size, cudaMemcpyDeviceToHost);
  bool wrote = writeFile(&sprite, "outputs/greyscale_test.bmp"); // TODO accept second CL arg

  for (int i = 0; i < size; i++) {
    printf("%d ", sprite.p[i]); // print output to make sure it looks right
  }

  // freedom!!
  free(sprite.p);
  cudaFree(in_pixels);
  cudaFree(out_pixels);

  return 0;
}
