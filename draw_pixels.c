#include <stdio.h>
#include <stdbool.h>
#include <SDL2/SDL.h>
#include <math.h>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;

int main(int argc, char *argv[]) {

    if(SDL_Init(SDL_INIT_VIDEO)<0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("Colorful Mandelbrot Set", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);

    if(window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    SDL_Surface* screenSurface = SDL_GetWindowSurface(window);

    bool quit = false;

    SDL_Event e;

    double zoom = 1.0;
    double offsetX = 0.0;
    double offsetY = 0.0;

    while(!quit) {
        while(SDL_PollEvent(&e) != 0) {
            if(e.type == SDL_QUIT) {
                quit = true;
            }
        }

        // Lock the surface to directly manipulate the pixels
        SDL_LockSurface(screenSurface);

        Uint32* pixels = (Uint32*)screenSurface->pixels;

        // Draw a colorful mandelbrot set with zoom
        for(int y = 0; y < SCREEN_HEIGHT; y++) {
            for(int x = 0; x < SCREEN_WIDTH; x++) {

                // Calculate the mandelbrot set with zoom and offset
                double cr = (x - SCREEN_WIDTH/2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX;
                double ci = (y - SCREEN_HEIGHT/2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetY;
                double zr = 0, zi = 0;
                int i = 0;
                while(i < 1000 && zr*zr + zi*zi < 4.0) {
                    double temp = zr*zr - zi*zi + cr;
                    zi = 2.0 * zr * zi + ci;
                    zr = temp;
                    i++;
                }

                // Color mapping based on iteration count with a gradient of red, orange, and yellow
                if (i == 1000) {
                    // Points inside the Mandelbrot set are colored with a fiery gradient
                    double magnitude = sqrt(zr * zr + zi * zi);
                    int red = (int)(fabs(sin(magnitude)) * 255);
                    int green = (int)(fabs(sin(magnitude)) * 50); // Further reduce green for a more fiery look
                    int blue = 0; // Set blue to 0 for warm colors only
                    pixels[y * SCREEN_WIDTH + x] = SDL_MapRGB(screenSurface->format, red, green, blue);
                } else {
                    // Points outside the Mandelbrot set are black
                    pixels[y * SCREEN_WIDTH + x] = SDL_MapRGB(screenSurface->format, 0,0,0);
                }
            }
        }

        // Unlock the surface
        SDL_UnlockSurface(screenSurface);

        SDL_UpdateWindowSurface(window);

        // Zoom in gradually
        zoom *= 1.05;
    }

    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}