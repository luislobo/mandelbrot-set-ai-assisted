#include <stdio.h>
#include <stdbool.h>
#include <SDL2/SDL.h>
#include <math.h>
#include <omp.h> // For OpenMP parallelism

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
    int maxIterations = 1000; // Fixed iteration count for consistency

    // Buffer to store color values before rendering
    Uint32* colorBuffer = (Uint32*)malloc(SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));

    while(!quit) {
        while(SDL_PollEvent(&e) != 0) {
            if(e.type == SDL_QUIT) {
                quit = true;
            }
        }

        int skip = (int)(zoom > 5 ? zoom / 5 : 1); // Skip pixels at higher zoom levels for performance

        #pragma omp parallel for collapse(2) schedule(dynamic, 4)
        for(int y = 0; y < SCREEN_HEIGHT; y += skip) {
            for(int x = 0; x < SCREEN_WIDTH; x += skip) {

                // Calculate the mandelbrot set with zoom and offset
                double cr = (x - SCREEN_WIDTH/2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX;
                double ci = (y - SCREEN_HEIGHT/2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetY;
                double zr = 0, zi = 0;
                int i = 0;
                while(i < maxIterations && zr*zr + zi*zi < 4.0) {
                    double temp = zr*zr - zi*zi + cr;
                    zi = 2.0 * zr * zi + ci;
                    zr = temp;
                    i++;
                }

                // Color mapping based on iteration count with a gradient representing depth
                double t = (double)i / maxIterations;
                int red, green, blue;
                if (i == maxIterations) {
                    // Points inside the Mandelbrot set are colored with a deep red
                    red = 0;
                    green = 0;
                    blue = 0;
                } else {
                    // Points outside the Mandelbrot set are colored using a cycling gradient
                    red = (int)(255 * (0.5 * sin(zoom * 0.1 + t * 6.28) + 0.5));
                    green = (int)(150 * (0.5 * sin(zoom * 0.1 + t * 6.28 + 2.0) + 0.5));
                    blue = (int)(50 * (0.5 * sin(zoom * 0.1 + t * 6.28 + 4.0) + 0.5));
                }

                Uint32 color = SDL_MapRGB(screenSurface->format, red, green, blue);

                for (int sy = 0; sy < skip; sy++) {
                    for (int sx = 0; sx < skip; sx++) {
                        if (y + sy < SCREEN_HEIGHT && x + sx < SCREEN_WIDTH) {
                            colorBuffer[(y + sy) * SCREEN_WIDTH + (x + sx)] = color;
                        }
                    }
                }
            }
        }

        // Lock the surface to directly manipulate the pixels
        SDL_LockSurface(screenSurface);

        Uint32* pixels = (Uint32*)screenSurface->pixels;

        // Copy color buffer to the surface pixels
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < SCREEN_HEIGHT; y++) {
            for (int x = 0; x < SCREEN_WIDTH; x++) {
                pixels[y * SCREEN_WIDTH + x] = colorBuffer[y * SCREEN_WIDTH + x];
            }
        }

        // Unlock the surface
        SDL_UnlockSurface(screenSurface);

        SDL_UpdateWindowSurface(window);

        // Zoom in gradually
        zoom *= 1.02;
    }

    free(colorBuffer);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}