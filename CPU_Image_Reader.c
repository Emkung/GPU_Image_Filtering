//#include <file.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#define PRINT_ERROR(a, args...) printf("ERROR %s() %s Line %d: " a "\n", __FUNCTION__, __FILE__, __LINE__, ##args);

/**
width The width, in pixels, of the image
height The height, in pixels, of the image
pixels The array of [a]bgr's
bytesPerPixel Does this bmp use 3- or 4-byte pixels? (3 for just RGB, 4 for RGBA)
 */
typedef struct {
  union { int width, w; };
  union { int height, h; };
  union { uint8_t *pixels, *p; };
  union { uint8_t bytesPerPixel, bpp; };
} sprite;

/**
@param *sprite The pointer to a sprite object that needs to be exported
@param *writefile The string name of the file to write out to
 */
bool writeFile(sprite *sprite, const char *writeFile) {
  bool return_v = true;

  int spriteSize = sprite->w * sprite->h * sprite->bpp;
  int fileSize = 2 + 13*sizeof(uint32_t) + spriteSize; // tag size + header size + sprite size

  char tag[] = { 'B', 'M' };
  uint32_t header[] = {
    fileSize, 0x00, 0x36, // size of file, irrelevant, location in file where pixels start
    0x28, // size of header beyond previous line
    sprite->w, sprite->h, // what it says
    0x180001, // 24 bits/pixel (0x18), and then 1 color plane (unsure what that means)
    0, // no compression mode
    0, // this would be the image size in bytes if we were compressing
    0x00002e23, 0x00002e23, // for display resolution of the image. used arbitrary sample from a file
    0, 0 // no special color space, all colors are relevant
  };

  FILE *file = fopen(writeFile, "wb");

  fwrite(tag, sizeof(char), 2, file);
  fwrite(header, sizeof(header), 1, file);
  int bytes_wrote = fwrite(sprite->p, sizeof(uint8_t), spriteSize, file);
  int pixels_wrote = bytes_wrote / sprite->bpp;

  if (bytes_wrote != spriteSize) {
    PRINT_ERROR("expected %d bytes, wrote %d (%d pixels)!\n", spriteSize, bytes_wrote, pixels_wrote);
    return_v = false;
  }
  
  fclose(file);
  return return_v;
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
  uint8_t *pixels; // every pixel has 3 8-bit ints

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
	PRINT_ERROR("(%s) bit depth\tEXP: 24\tOUT: %d\n", filename, bit_depth);
      }
      else {
	pixel_count = width * height;
	byte_depth = bit_depth / 8;
	pixels = malloc(pixel_count*byte_depth);

	if(pixels) {
	  fseek(file, image_data_address, SEEK_SET);

	  int pixels_read = fread(pixels, byte_depth, pixel_count, file); // stores the number of pixels actually read

	  if (pixels_read == pixel_count) {
	    sprite->w = width;
	    sprite->h = height;
	    sprite->p = pixels;
	    sprite->bpp = byte_depth;
	    return_v = pixels_read;
	  }

	  else { // error here
	    PRINT_ERROR("(%s) expected %d pixels, read %d\n", filename, pixel_count, pixels_read);
	    free(pixels);
	  }

	}

	  else { // error here
	    PRINT_ERROR("(%s) malloc failure for %d pixels\n", filename, pixel_count);
	    if (pixels) { free(pixels);}
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
