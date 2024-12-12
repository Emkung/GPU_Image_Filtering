#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

typedef struct {
  union { int width, w; };
  union { int height, h; };
  union { uint8_t *pixels, *p; };
  union { uint8_t bytesPerPixel, bpp; };
} sprite;

extern int loadFile(sprite *sprite, const char *filename);
extern bool writeFile(sprite *sprite, const char *writeFile);

int main(int argc, char *argv[]){
  static sprite sprite;
  int pixels_read = loadFile(&sprite, argv[1]);
  printf("%d\n", pixels_read);

  int spriteBytes = pixels_read*sprite.bpp;
  for (int i = 0; i < spriteBytes; i+=3) {
    printf("R: %d,   G: %d,   B: %d\n", sprite.p[i+2], sprite.p[i+1], sprite.p[i]);
  }

  bool wrote = writeFile(&sprite, "write_test.bmp");
  wrote ? printf("\nSuccess!\n") : printf("\nNeeds Debugging T-T\n");
  
  free(sprite.p);
  return 0;
}
