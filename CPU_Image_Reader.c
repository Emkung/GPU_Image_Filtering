//#include <file.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#define PRINT_ERROR(a, args...) printf("ERROR %s() %s Line %d: " a "\n", __FUNCTION__, __FILE__, __LINE__, ##args);

typedef struct {
  uint8_t r, g, b;
} pixel;

typedef struct {
  union { int width, w; };
  union { int height, h; };
  union { pixel *pixels, *p; };
} sprite;

/**
@param *sprite The pointer to a sprite object that needs to be exported
@param *writefile The string name of the file to write out to
@param depth Bytes per pixel (3 for RGB pixels, 4 for RGBA pixels)
 */
boolean writeFile(sprite *sprite, const int depth, const char *writeFile) {
  int return_v = false;

  int spriteSize = sprite.w * sprite.h * depth;
  int fileSize = 2 + 13*sizeof(uint32_t) + spriteSize; // tag size + header size + sprite size
  char tag[] = { 'B', 'M' };
  uint32_t header[] = {
    fileSize, 0x00, 0x36, // size of file, irrelevant, location in file where pixels start
    0x28, // size of header beyond previous line
    sprite.w, sprite.h, // what it says
    0x180001, // 24 bits/pixel (0x18), and then 1 color plane (unsure what that means)
    0, // no compression mode
    0, // this would be the image size in bytes if we were compressing
    0x00002e23, 0x00002e23, // these are for display resolution of the image. used an arbitrary sample from a file
    0, 0 // no special color space, all colors are relevant
  };


  FILE *file = fopen(*writeFile, "w+");
  
  fclose(file);
}


int loadFile(sprite *sprite, const char *filename){
  int return_v = 0;
  FILE *file;
  file = fopen(filename, "rb");

  uint32_t image_data_address; // location of file where pixels begin
  int32_t width; // on the tag
  int32_t height; // on the tag
  uint32_t pixel_count; // HOW MANY THINGS TO READ
  uint16_t bit_depth; // some kind of error handling
  uint8_t byte_depth; // bytes in a pixel
  pixel *pixels; // what we actually wanna return

  if (file) {
    if ( fgetc(file) == 'B' && fgetc(file) == 'M' ) { // confirm bitmap file
      fseek(file, 8, SEEK_CUR);
      fread(&image_data_address, 4, 1, file);
      fseek(file, 4, SEEK_CUR);
      fread(&width, 4, 1, file);
      fread(&height, 4, 1, file);
      fseek(file, 2, SEEK_CUR);
      fread(&bit_depth, 2, 1, file);

      if ( bit_depth != 24 ) { // error here
	PRINT_ERROR("(%s) bit depth\tEXP: 32\tOUT: %d\n", filename, bit_depth);
      }
      else {
	pixel_count = width * height;
	byte_depth = bit_depth / 8;
	uint8_t *rgbs = malloc(pixel_count * byte_depth);
	pixels = malloc(pixel_count * sizeof(pixel));

	if(pixels && rgbs) {
	  fseek(file, image_data_address, SEEK_SET);

	  int pixels_read = fread(rgbs, byte_depth, pixel_count, file); // stores the number of pixels actually read

	  if (pixels_read == pixel_count) {

	    for(int i = 0; i < pixel_count; i++){
	      pixel px;
	      px.b = rgbs[3*i];
	      px.g = rgbs[3*i+1];
	      px.r = rgbs[3*i+2];
	      pixels[i] = px;
	    }

	    sprite->w = width;
	    sprite->h = height;
	    sprite->p = pixels;
	    return_v = pixels_read;
	  }

	  else { // error here
	    PRINT_ERROR("(%s) expected %d pixels, read %d\n", filename, pixel_count, pixels_read);
	    free(pixels);
	  }

	  free(rgbs);
	  // free(pixels);
	}

	  else { // error here
	    PRINT_ERROR("(%s) malloc failure for %d pixels\n", filename, pixel_count);
	    if (pixels) { free(pixels);}
	    if (rgbs) { free(rgbs);}
	  }
        }
      }
  
    else { // error here
      PRINT_ERROR("(%s) first two bytes not `BM`\n", filename);
    }
	
    fclose(file);
  }  
  else { // error here
    PRINT_ERROR("(%s) failed to open file\n", filename);
  }
  
  return return_v;
}

int main(int argc, char *argv[]){
  static sprite sprite;
  int pixels_read = loadFile(&sprite, argv[1]);
  printf("%d\n", pixels_read);

  for (int i = 0; i < pixels_read; i++) {
    pixel pixel = sprite.p[i];
    printf("R: %d,   G: %d,   B: %d\n", pixel.r, pixel.g, pixel.b);
  }
  free(sprite.p);
  return 0;
}
