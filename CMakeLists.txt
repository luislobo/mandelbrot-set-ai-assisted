cmake_minimum_required(VERSION 3.10)

# Set the project name
project(PixelDrawingApp)

# Set the C++ standard (you can omit this if you don't need C++ specific options)
set(CMAKE_C_STANDARD 11)

# Find the SDL2 library
find_package(SDL2 REQUIRED)

# Add the executable target
add_executable(PixelDrawingApp draw_pixels.c)

# Link SDL2 to the executable
target_link_libraries(PixelDrawingApp SDL2::SDL2 m)
