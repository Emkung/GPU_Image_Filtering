#include <file.h>
#include <stdio.h>

typedef struct {
  unsigned char a, r, g, b;
} pixel;

typedef struct {
  union { int width, w; };
  union { int height, h; };
  union { uint32_t *pixels, *p; };
} sprite;


bool loadFile(sprite *sprite, const char *filename){
  bool return_v = true;
  FILE *file;
  file = fopen(filename, "rb");

  uint32_t image_data_address; // location of file where pixels begin
  int32_t width; // on the tag
  int32_t height; // on the tag
  uint32_t pixel_count; // HOW MANY THINGS TO READ
  uint16_t bit_depth; // some kind of error handling
  uint8_t byte_depth; // bytes in a pixel
  uint32_t *pixels; // what we actually wanna return

  if (file) {
    if ( fgetc(file) == 'B' && fgetc(file) == 'M' ) { // confirm bitmap file
      fseek(file, 8, SEEK_CUR);
      fread(&image_data_address, 4, 1, file);
      fseek(file, 4, SEEK_CUR);
      fseek(&width, 4, 1, file);
      fread(&height, 4, 1, file);
      fseek(file, 2, SEEK_CUR);
      fread(&bit_depth, 2, 1, file);

      if ( bit_depth != 32 ) { // error here
	PRINT_ERROR("(%s) bit depth\tEXP: 32\tOUT: %d\n", filename, bit_depth);
	return_v = false;
      }
      else {
	pixel_count = width * height;
	byte_depth = bit_depth / 8;
	pixels = malloc(pixel_count * byte_depth);

	if(pixels) {
	  fseek(file, image_data_address, SEEK_SET);
	  int pixels_read = fread(pixels, byte_depth, pixel_count, file); // stores the number of pixels actually read

	  if (pixels_read == pixel_count) {
	    sprite->w = width;
	    sprite->h = height;
	    sprite->p = pixels;
	  }

	  else { // error here
	    PRINT_ERROR("(%s) expected %d pixels, read %d\n", filename, pixel_count, pixels_read);
	    free(pixels)
	    return_v = false;
	  }

	  free(pixels);
	}

	else { // error here
	  PRINT_ERROR("(%s) malloc failure for %d pixels\n", filename, pixel_count);
	  return_v = false;
	}
      }

      else { // error here
	PRINT_ERROR("(%s) first two bytes not `BM`\n", filename, pixel);
	return_v = false;
      }

      fclose(file);
    }
    
    else { // error here
      PRINT_ERROR("(%s) failed to open file\n", filename, pixel);
      return_v = false;
    }
    
  }
  
  return return_v;
}
