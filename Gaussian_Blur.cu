#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
using namespace std;

#define BLOCKSIZE 1024
#define TILEWIDTH 32
#define RADIUS 1
#define SIGMA 1.
__constant__ float MASK[RADIUS+1];

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
__device__ int pos(int number) {
  int out = number << 1; // fancy bitfuckery erases sign bit
  return out >> 1; // now sign bit is 0!!
}

/** 
Performs a blur operation on the input using the provided convolution kernel.
  Uses 1D symmetrical convolution, which gets applied horizontally ONLY in this step.
  The next step will calculate the identical convolution in the vertical direction.

@param input: The input image
@param output: Where to put the blurred image
@param w, @param h, @param depth: The sprite information for use in kernel
@param kernel, @param r: The 1D Gaussian kernel and its radius. 
  Kernel requires FIRST r+1 elements to be submitted. Because kernels must be symmetrical, we only use the first r+1.
*/
__global__ void gaussianBlurWidth(uint8_t* input, uint8_t* output, // in- and out-puts
				  const int w, const int h, const int depth, // width, height, and depth of the image
				  const int rad // kernel radius. kernel array is tied to constant memory at MASK.
				  ){
  // memory caches
  __shared__ uint8_t red[BLOCKSIZE + 2*RADIUS]; // shared arrays with halo. must be initialized to max possible
  __shared__ uint8_t green[BLOCKSIZE + 2*RADIUS]; // greens
  __shared__ uint8_t blue[BLOCKSIZE + 2*RADIUS]; // blues
  int lindex = threadIdx.x + rad;
  
  if (lindex-rad < w && blockIdx.x < h){ // stay in range
    int gindex = depth * (blockIdx.x*w + threadIdx.x); // pretend blockDim.x == w, because for now it does
    
    // load into shared memory
    uint8_t b = input[gindex], g = input[gindex+1], r = input[gindex+2];
    blue[lindex] = b; green[lindex] = g; red[lindex] = r;

    // halo handling
    if (lindex < 2*rad) {
      int hindex = gindex - (r*depth); // halo index
      if (hindex < 0) { // use what we've already read
	int iout = 2*rad - lindex - 1;
	blue[iout] = b; green[iout] = g; red[iout] = r; // no need to read new vals
      } else { // play fetch
	uint8_t bh = input[hindex], gh = input[hindex+1], rh = input[hindex+2]; // handle halos normally
	blue[lindex-rad] = bh; green[lindex-rad+1] = gh; red[lindex-rad+2] = rh;
      }
    }
    if (lindex >= blockDim.x) { // these are separated to handle r's greater than the block size.
      int hindex = gindex + (rad*depth); // halo index
      if (hindex >= w) { // swap to h on next kernel
	int mod = w-1 - lindex+rad; // in 0..(r-1), where highest possible = 0, lowest =(r-1)
	int iout = w+rad + mod;
	blue[iout] = b; green[iout] = g; red[iout] = r;
      } else {
	uint8_t bh = input[hindex], gh = input[hindex+1], rh = input[hindex+2]; // handle halos normally
	blue[lindex+rad] = bh; green[lindex+rad+1] = gh; red[lindex+rad+2] = rh;
      }
    }
    __syncthreads(); // patience

    // math
    float rSum = 0, gSum = 0, bSum = 0;
    for (int i = -rad; i <= rad; i++) {
      /** calculate stuff */
      float f = MASK[pos(pos(i)-rad)]; // {-r, r}->0, 0->r, correct order between
      rSum += f*red[lindex+i];
      gSum += f*green[lindex+i];
      bSum += f*blue[lindex+i];
    }

    // write
    output[gindex] = (uint8_t) bSum; output[gindex+1] = (uint8_t) gSum; output[gindex+2] = (uint8_t) rSum;
  }
}

__global__ void gaussianBlurHeight(uint8_t* input, uint8_t* output, // in- and out-puts
				   const int w, const int h, const int depth, // width, height, and depth of the image
				   const int rad // kernel radius
				   ){
  // memory caches
  __shared__ uint8_t red[BLOCKSIZE + 2*RADIUS]; // shared arrays with halo.
  __shared__ uint8_t green[BLOCKSIZE + 2*RADIUS]; // greens
  __shared__ uint8_t blue[BLOCKSIZE + 2*RADIUS]; // blues
  int lindex = threadIdx.x + rad;

  if (lindex-rad < h && blockIdx.x < w){

    int gindex = depth * (blockIdx.x + threadIdx.x*w); // for now, block size has to be at least h
    
    // load into shared memory
    uint8_t b = input[gindex], g = input[gindex+1], r = input[gindex+2];
    blue[lindex] = b; green[lindex] = g; red[lindex] = r;

    // halo handling
    if (lindex < 2*rad) {
      int hindex = gindex - (rad*w*depth); // halo index
      if (hindex < 0) { // use what we've already read
	int iout = 2*rad - lindex - 1;
	blue[iout] = b; green[iout] = g; red[iout] = r; // no need to read new vals
      } else { // play fetch
	uint8_t bh = input[hindex], gh = input[hindex+1], rh = input[hindex+2]; // handle halos normally
	blue[lindex-rad] = bh; green[lindex-rad+1] = gh; red[lindex-rad+2] = rh;
      }
    }
    if (lindex >= blockDim.x) { // these are separated to handle r's greater than the block size.
      int hindex = gindex + (rad*w*depth); // halo index
      if (hindex >= h) { // swap to h on next kernel
	int mod = h-1 - lindex+rad; // in 0..(r-1), where highest possible = 0, lowest =(r-1)
	int iout = h+rad + mod;
	blue[iout] = b; green[iout] = g; red[iout] = r;
      } else {
	uint8_t bh = input[hindex], gh = input[hindex+1], rh = input[hindex+2]; // handle halos normally
	blue[lindex+rad] = bh; green[lindex+rad+1] = gh; red[lindex+rad+2] = rh;
      }
    }
    __syncthreads(); // patience

    // math
    float rSum = 0, gSum = 0, bSum = 0;
    for (int i = -rad; i <= rad; i++) {
      /** calculate stuff */
      float f = MASK[pos(pos(i)-rad)]; // {-r, r}->0, 0->r, correct order in between.
      rSum += f*red[lindex+i];
      gSum += f*green[lindex+i];
      bSum += f*blue[lindex+i];
    }

    // write
    output[gindex] = (uint8_t) bSum; output[gindex+1] = (uint8_t) gSum; output[gindex+2] = (uint8_t) rSum;
  }
}

/**
Generates the first half +1 elements of a Gaussian kernel
Because Gaussian kernels are symmetric, this can be extrapolated to a full kernel, and will be later.
Done this way because there's fewer CPU math operations this way T-T
  and also i'm a massochist or smth idk
 */
__host__ float* gaussianKernel(const int r, const float sigma) {
  float* out = (float*) malloc ( (r+1)*sizeof(double) );
  float s = 2*sigma*sigma;
  
  float sum = 0.; // for normalizing
  for (int x = -r; x <= 0; x++) { // only use first half of kernel for calculations
    out[x+r] = exp(-(x*x) / s) / (M_PI * s);
    sum += x==0 ? out[x+r] : 2*out[x+r];
  }

  for (int i = 0; i <= r; i++) {
    out[i] /= sum; // normalize
  }

  return out;
}

/**
Runs a full Gaussian blur, start to finish, including copying out from cuda into the provided sprite's pixel list.
 */
__host__ bool gaussianBlur(sprite* sprite, const int r, const float sig) {
  if (sprite->w > BLOCKSIZE || sprite->h > BLOCKSIZE) {
    cerr << "Unable to blur images greater than " << BLOCKSIZE << " in a single dimension.\n";
    return false;
  } if (r >= sprite->h || r >= sprite->w) {
    cerr << "Unable to blur images with r > either image dimension.\n";
    return false;
  }
  
  // cuda malloc
  int size = sprite->w * sprite->h * sprite->bpp;
  uint8_t *in_pixels, *out_pixels;
  cudaMalloc(&in_pixels, sizeof(uint8_t) * size);
  cudaMalloc(&out_pixels, sizeof(uint8_t) * size);
  cudaMemcpy(in_pixels, sprite->p, size, cudaMemcpyHostToDevice);

  // run kernel
  float* mask = gaussianKernel(r, sig);
  cudaMemcpyToSymbol(MASK, mask, (r+1)*sizeof(float));
  //int blockX = ceil ( (1.*sprite.w) / TILEWIDTH ), blockY = ceil ( (1.*sprite.h) / TILEWIDTH );
  //int xWidth = TILEWIDTH < sprite.w ? TILEWIDTH : sprite.w, yWidth = TILEWIDTH < sprite.h ? TILEWIDTH : sprite.h;
  //dim3 dimGrid( blockX, blockY, 1);
  //dim3 dimBlock( xWidth, yWidth, 1);
  gaussianBlurWidth<<<sprite->h, sprite->w>>>(in_pixels, out_pixels,
  					      sprite->w, sprite->h, sprite->bpp,
  					      r);
  cudaDeviceSynchronize(); // IMMEDIATELY after opening the kernel >:|
  gaussianBlurHeight<<<sprite->w, sprite->h>>>(out_pixels, in_pixels, // swap so output goes back in
  					       sprite->w, sprite->h, sprite->bpp,
  					       r);
  cudaDeviceSynchronize();
  cerr << cudaGetErrorString(cudaGetLastError()) << "\n";

  // write file out
  cudaMemcpy(sprite->p, in_pixels, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i <= r; i++) {
    cout << mask[i] << " ";
  }
  cout << endl;

  // freedom!!
  free(mask);
  cudaFree(in_pixels);
  cudaFree(out_pixels);
  return true;
}

/** Performs a blur operation on the input using a flat 1/r^2 convolution kernel */
__host__ bool flatBlur(uint8_t* input, uint8_t* output) {
  /** ok i lied not really */
  return false;
}

int main(int argc, char *argv[]) {
  // initialize sprite
  static sprite sprite;
  int pixels_read = loadFile(&sprite, argv[1]); cout << pixels_read << "\n";
  int size = pixels_read * sprite.bpp;

  bool success = gaussianBlur(&sprite, RADIUS, SIGMA);
  if (!success) {
    cerr << "Blur kernel failed. See above for more detail error messaging.\n";
    return -1;
  }
  
  bool wrote = writeFile(&sprite, "outputs/blur_test.bmp"); // TODO accept second CL arg

  for (int i = 0; i < size; i++) {
    cout << (int) sprite.p[i] << " "; // print output to make sure it looks right
  }
  cout << "\n";

  // freedom!!
  free(sprite.p);

  return 0;
}
