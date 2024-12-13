#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#define BLOCKSIZE 1024
#define TILEWIDTH 32
#define RADIUS 1

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
Performs a blur operation on the input using the provided convolution kernel.
  Uses 1D symmetrical convolution, which gets applied horizontally and then vertically

@param input: The input image
@param output: Where to put the blurred image
@param w, @param h, @param depth: The sprite information for use in kernel
@param kernel, @param r: The 1D Gaussian kernel and its radius
*/
__global__ void gaussianBlur(uint8_t* input, uint8_t* output, // in- and out-puts
			     int w, int h, int depth, // width, height, and depth of the image
			     double* kernel, int r // kernel array (1D) and radius
			     ){
  // memory caches
  __shared__ uint8_t red[BLOCKSIZE + 2*r]; // shared arrays with halo.
  __shared__ uint8_t green[BLOCKSIZE + 2*r]; // greens
  __shared__ uint8_t blue[BLOCKSIZE + 2*r]; // blues
  
  int size = w * h * depth;
  int lindex = threadIdx.x;
  int gindex = depth * (blockIdx.x*blockDim.x +threadIdx.x);

  if (index+depth <= size){ // stay in range
    
    // math
    uint8_t b = input[index], g = input[index+1], r = input[index+2]; 
    //uint8_t greyVal = 0.114*b + 0.587 * g + 0.299 * r;

    // write
    for (int i = index; i < index + 3; i++){
      //output[i] = greyVal; // inefficient writes?
    }
  }
}

/** Performs a blur operation on the input using a flat 1/r^2 convolution kernel */
__global__ void flatBlur(uint8_t* input, uint8_t* output

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

  // run kernel
  int blockX = ceil ( (1.*sprite.w) / TILEWIDTH ), blockY = ceil ( (1.*sprite.h) / TILEWIDTH );
  int xWidth = TILEWIDTH < sprite.w ? TILEWIDTH : sprite.w, yWidth = TILEWIDTH < sprite.h ? TILEWIDTH : sprite.h;
  dim3 dimGrid( blockX, blockY, 1);
  dim3 dimBlock( xWidth, yWidth, 1);
  gaussianBlur<<<dimGrid, dimBlock>>>(in_pixels, out_pixels,
				      sprite.w, sprite.h, sprite.bpp,
				      /*kernel*/, /*r*/);
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
