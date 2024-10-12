#include <stdio.h>
#include <stdbool.h>
#include <SDL2/SDL.h>
#include <math.h>
#include <omp.h> // For OpenMP parallelism
#include <immintrin.h> // For SIMD optimizations

// Constants for screen dimensions
const int SCREEN_WIDTH = 800;   // Width of the window in pixels
const int SCREEN_HEIGHT = 600;  // Height of the window in pixels

int main(int argc, char *argv[]) {

    if(SDL_Init(SDL_INIT_VIDEO)<0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Create a window to display the Mandelbrot set
    SDL_Window* window = SDL_CreateWindow("Colorful Mandelbrot Set", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);

    if(window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    // Get the surface contained by the window
    SDL_Surface* screenSurface = SDL_GetWindowSurface(window);

    bool quit = false;  // Flag to indicate when to exit the main loop

    SDL_Event e;  // Event handler to capture user events

    double zoom = 1.0;        // Zoom level of the Mandelbrot set visualization
    double offsetX = -0.75;     // Horizontal offset for panning the view
    double offsetY = 0.1;     // Vertical offset for panning the view
    int maxIterations = 1000; // Maximum number of iterations to determine if a point belongs to the Mandelbrot set

    // Buffer to store color values before rendering
    Uint32* colorBuffer = (Uint32*)malloc(SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));

    while(!quit) {
        // Handle user events, e.g., quit or key presses
        while(SDL_PollEvent(&e) != 0) {
            if(e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_UP:
                        offsetY -= 0.05 / zoom; // Pan up
                        break;
                    case SDLK_DOWN:
                        offsetY += 0.05 / zoom; // Pan down
                        break;
                    case SDLK_LEFT:
                        offsetX -= 0.05 / zoom; // Pan left
                        break;
                    case SDLK_RIGHT:
                        offsetX += 0.05 / zoom; // Pan right
                        break;
                }
            }
        }
        while(SDL_PollEvent(&e) != 0) {
            if(e.type == SDL_QUIT) {
                quit = true;
            }
        }

        // Skip factor to improve performance at higher zoom levels
        int skip = 1; // Always render all pixels to maintain resolution
        // Adjusted iteration count to maintain performance at higher zoom levels
        int adjustedIterations = (int)(maxIterations / sqrt(zoom));
        if (adjustedIterations < 100) {
            adjustedIterations = 100; // Ensure a minimum number of iterations
        }

        // Calculate the Mandelbrot set and store the color values in the buffer
        #pragma omp parallel for schedule(dynamic, 4)
        for(int y = 0; y < SCREEN_HEIGHT; y += skip) {
            for(int x = 0; x < SCREEN_WIDTH; x += skip) {

                // Calculate the real and imaginary components of the complex number
                double cr = (x - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX;
                double ci = (y - SCREEN_HEIGHT / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetY;

                // Early escape for points known to be inside the main cardioid or period-2 bulb
                double q = (cr - 0.25) * (cr - 0.25) + ci * ci; // Represents a value used to determine if the point is within the main cardioid or period-2 bulb
                // If 'q' meets specific conditions, it means the point is definitely inside the set, allowing us to skip further calculations
                // The conditions used below are derived from properties of the Mandelbrot set:
                // 1. If q * (q + (cr - 0.25)) < 0.25 * ci * ci, the point is inside the main cardioid.
                // 2. If (cr + 1) * (cr + 1) + ci * ci < 0.0625, the point is inside the period-2 bulb.
                if (q * (q + (cr - 0.25)) < 0.25 * ci * ci || (cr + 1) * (cr + 1) + ci * ci < 0.0625) {
                    // Set the color to black for points inside the set
                    for (int sy = 0; sy < skip; sy++) {
                        for (int sx = 0; sx < skip; sx++) {
                            if (y + sy < SCREEN_HEIGHT && x + sx < SCREEN_WIDTH) {
                                colorBuffer[(y + sy) * SCREEN_WIDTH + (x + sx)] = SDL_MapRGB(screenSurface->format, 0, 0, 0);
                            }
                        }
                    }
                    continue;
                }

                // Initialize real and imaginary parts of z to zero
                double zr = 0, zi = 0;
                int i = 0;
                // Iterate to determine if the point is in the Mandelbrot set
                while(i < adjustedIterations && zr * zr + zi * zi < 4.0) {
                    double temp = zr * zr - zi * zi + cr;
                    zi = 2.0 * zr * zi + ci;
                    zr = temp;
                    i++;
                }

                // Color mapping based on iteration count with a gradient representing depth
                double t = (double)i / adjustedIterations;
                int red, green, blue;
                if (i == adjustedIterations) {
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

                // Map the RGB color to the SDL color format
                Uint32 color = SDL_MapRGB(screenSurface->format, red, green, blue);

                // Set the color for the current pixel and its neighbors if skipping
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

        // Copy the color buffer to the screen surface
        memcpy(screenSurface->pixels, colorBuffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));

        // Unlock the surface
        SDL_UnlockSurface(screenSurface);

        // Update the window surface to show the new frame
        SDL_UpdateWindowSurface(window);

        // Gradually zoom in for animation effect
        zoom *= 1.02;
    }

    // Free the color buffer memory
    free(colorBuffer);
    // Destroy the window
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}