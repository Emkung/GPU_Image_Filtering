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

/** provides the absolute value of an input */
__device__ int abs(int number) {
  int out = number << 1; // fancy bitfuckery erases sign bit
  return temp >> 1; // now sign bit is 0!!
}

/** 
Performs a blur operation on the input using the provided convolution kernel.
  Uses 1D symmetrical convolution, which gets applied horizontally ONLY in this step.
  The next step will calculate the identical convolution in the vertical direction.

@param input: The input image
@param output: Where to put the blurred image
@param w, @param h, @param depth: The sprite information for use in kernel
@param kernel, @param r: The 1D Gaussian kernel and its radius. 
  Kernel should be symmetrical and requires LAST r+1 elements to be submitted.
*/
__global__ void gaussianBlurWidth(uint8_t* input, uint8_t* output, // in- and out-puts
			     int w, int h, int depth, // width, height, and depth of the image
			     float* kernel, int r // kernel array (1D) and radius
			     ){
  // memory caches
  __shared__ uint8_t red[BLOCKSIZE + 2*r]; // shared arrays with halo.
  __shared__ uint8_t green[BLOCKSIZE + 2*r]; // greens
  __shared__ uint8_t blue[BLOCKSIZE + 2*r]; // blues
  
  int size = w * h * depth;
  int lindex = threadIdx.x + r;
  int gindex = depth * (blockIdx.x*blockDim.x +threadIdx.x);

  if (index+depth <= size){ // stay in range
    
    // load into shared memory
    uint8_t b = input[gindex], g = input[gindex+1], r = input[gindex+2];
    blue[lindex] = b; green[lindex] = g; red[lindex] = r;

    // halo handling
    if (lindex < 2*r) {
      int hindex = gindex - (r*depth); // halo index
      if (hindex < 0) { // use what we've already read
	int iout = 2*r - lindex - 1;
	blue[iout] = b; green[iout] = g; red[iout] = r;
      } else { // play fetch
	uint8_t bh = input[hindex], gh = input[hindex+1], rh = input[hindex+2]; // handle halos normally
	blue[lindex+mod] = bh; green[lindex+mod+1] = gh; red[lindex+mod+2] = rh;
      }
    }
    if (lindex >= blockDim.x) {
      int hindex = gindex + (r*depth); // halo index
      if (hindex >= w) { // swap to h on next kernel
	int mod = blockDim.x-1 - lindex+r; // in 0..(r-1), where highest possible = 0, lowest =(r-1)
	int iout = blockDim.x+r + mod;
	blue[iout] = b; green[iout] = g; red[iout] = r;
      } else {
	uint8_t bh = input[hindex], gh = input[hindex+1], rh = input[hindex+2]; // handle halos normally
	blue[lindex+mod] = bh; green[lindex+mod+1] = gh; red[lindex+mod+2] = rh;
      }
    }

    __syncthreads();

    // math
    float rSum = 0, gSum = 0, bSum = 0;
    for (int i = -r; i <= r; i++) {
      /** calculate stuff */
      float f = kernel[abs(i)];
      rSum += f*red[lindex+i];
      gSum += f*green[lindex+i];
      bSum += f*blue[lindex+i];
    }

    // write
    output[gindex] = (uint8_t) bSum; output[gindex+1] = (uint8_t) gSum; output[gindex+2] = (uint8_t) rSum;
  }
}

/** Performs a blur operation on the input using a flat 1/r^2 convolution kernel */
__global__ void flatBlur(uint8_t* input, uint8_t* output) {
  /** ok i lied not really */
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

  // run kernel
  if (sprite.w > BLOCKSIZE || sprite.h > BLOCKSIZE) {
    std::cerr << "Unable to blur images greater than " << BLOCKSIZE << " in a single dimension.\n";
    return -1;
  }
  //int blockX = ceil ( (1.*sprite.w) / TILEWIDTH ), blockY = ceil ( (1.*sprite.h) / TILEWIDTH );
  //int xWidth = TILEWIDTH < sprite.w ? TILEWIDTH : sprite.w, yWidth = TILEWIDTH < sprite.h ? TILEWIDTH : sprite.h;
  //dim3 dimGrid( blockX, blockY, 1);
  //dim3 dimBlock( xWidth, yWidth, 1);
  gaussianBlurWidth<<<h, w>>>(in_pixels, out_pixels,
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
