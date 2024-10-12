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
    double zoomSpeed = 1.02;  // Zoom speed factor
    double offsetX = -0.75;   // Horizontal offset for panning the view
    double offsetY = 0.1;     // Vertical offset for panning the view
    int maxIterations = 1000; // Maximum number of iterations to determine if a point belongs to the Mandelbrot set

    // Buffer to store color values before rendering
    Uint32* colorBuffer = (Uint32*)malloc(SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));

    // Variables to track the viewport
    double velocityX = 0.0;
    double velocityY = 0.0;
    double targetOffsetX = offsetX;
    double targetOffsetY = offsetY;
    const double acceleration = 0.005;
    const double maxSpeed = 0.05;
    const double deceleration = 0.9;
    const double clickSpeed = 0.1; // Speed factor for moving towards click target

    while(!quit) {
        // Handle user events, e.g., quit or key presses
        while(SDL_PollEvent(&e) != 0) {
            if(e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    int mouseX, mouseY;
                    SDL_GetMouseState(&mouseX, &mouseY);
                    // Set new target offsets to center the clicked point
                    targetOffsetX = (mouseX - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX;
                    targetOffsetY = (mouseY - SCREEN_HEIGHT / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetY;
                }
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_UP:
                        velocityY -= acceleration;
                        break;
                    case SDLK_DOWN:
                        velocityY += acceleration;
                        break;
                    case SDLK_LEFT:
                        velocityX -= acceleration;
                        break;
                    case SDLK_RIGHT:
                        velocityX += acceleration;
                        break;
                }
            } else if (e.type == SDL_MOUSEWHEEL) {
                if (e.wheel.y > 0) {
                    zoomSpeed *= 1.01; // Gradually increase zoom speed for zooming in
                    if (zoomSpeed > 1.1) zoomSpeed = 1.1; // Cap the zoom speed
                } else if (e.wheel.y < 0) {
                    zoomSpeed *= 0.99; // Gradually decrease zoom speed for zooming out
                    if (zoomSpeed < 0.9) zoomSpeed = 0.9; // Cap the zoom speed
                }
            }
        }

        // Move towards target offsets smoothly
        double diffX = targetOffsetX - offsetX;
        double diffY = targetOffsetY - offsetY;
        offsetX += diffX * clickSpeed;
        offsetY += diffY * clickSpeed;

        // Clamp velocities to max speed
        if (velocityX > maxSpeed) velocityX = maxSpeed;
        if (velocityX < -maxSpeed) velocityX = -maxSpeed;
        if (velocityY > maxSpeed) velocityY = maxSpeed;
        if (velocityY < -maxSpeed) velocityY = -maxSpeed;

        // Apply velocity to offsets
        offsetX += velocityX / zoom;
        offsetY += velocityY / zoom;

        // Apply deceleration when no key is pressed
        velocityX *= deceleration;
        velocityY *= deceleration;

        // Adjust iteration count based on zoom level to maintain performance
        int adjustedIterations = maxIterations / (1 + log2(zoom));
        if (adjustedIterations < 100) {
            adjustedIterations = 100; // Ensure a minimum number of iterations
        }

        // Calculate the Mandelbrot set and store the color values in the buffer
        #pragma omp parallel for schedule(dynamic, 4) collapse(2)
        for(int y = 0; y < SCREEN_HEIGHT; y++) {
            for(int x = 0; x < SCREEN_WIDTH; x++) {
                // Calculate the real and imaginary components of the complex number
                double cr = (x - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX;
                double ci = (y - SCREEN_HEIGHT / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetY;

                // Early escape for points known to be inside the main cardioid or period-2 bulb
                double q = (cr - 0.25) * (cr - 0.25) + ci * ci;
                if (q * (q + (cr - 0.25)) < 0.25 * ci * ci || (cr + 1) * (cr + 1) + ci * ci < 0.0625) {
                    colorBuffer[y * SCREEN_WIDTH + x] = SDL_MapRGB(screenSurface->format, 0, 0, 0);
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
                    red = green = blue = 0;
                } else {
                    red = (int)(200 * (0.5 * sin(0.1 + t * 3.14) + 0.5));
                    green = (int)(100 * (0.5 * sin(0.1 + t * 3.14 + 1.0) + 0.5));
                    blue = (int)(50 * (0.5 * sin(0.1 + t * 3.14 + 2.0) + 0.5));
                }

                // Map the RGB color to the SDL color format
                colorBuffer[y * SCREEN_WIDTH + x] = SDL_MapRGB(screenSurface->format, red, green, blue);
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

        // Apply automatic zoom with the current zoom speed
        zoom *= zoomSpeed;
    }

    // Free the color buffer memory
    free(colorBuffer);
    // Destroy the window
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}