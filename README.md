# Parallel Image Convolution with MPI

A C++ program that performs image convolution (blurring, sharpening, edge detection, etc.) in parallel across multiple processors using MPI (Message Passing Interface) and OpenCV. The program runs both a serial and a parallel version of the convolution and verifies that their outputs are identical.

## Overview

The program reads an image and a convolution kernel (from a CSV file), then:

1. Runs a **serial convolution** on rank 0 as a reference implementation.
2. Splits the image into row-wise chunks and **scatters** them to all MPI processes, including the halo rows each process needs to compute a valid convolution at its boundaries.
3. Each process performs the convolution on its assigned (non-halo) rows.
4. **Gathers** the processed rows back to rank 0 to assemble the final parallel image.
5. **Compares** the serial and parallel outputs pixel-by-pixel and reports whether they match.

It also prints the parallel execution time and writes both output images to disk.

## Requirements

- A C++ compiler with C++11 support
- [OpenCV 4](https://opencv.org/) (headers expected under `opencv4/opencv2/`)
- An MPI implementation (e.g., OpenMPI or MPICH)

## Building

Assuming the source file is `convolution.cpp`:

```bash
mpic++ -std=c++11 convolution.cpp -o convolution `pkg-config --cflags --libs opencv4`
```

If `pkg-config` is not available, link OpenCV manually:

```bash
mpic++ -std=c++11 convolution.cpp -o convolution \
    -I/usr/include/opencv4 \
    -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
```

## Usage

```bash
mpirun -np <num_processes> ./convolution <image_file> <kernel_file>
```

**Arguments:**

- `<num_processes>` — number of MPI processes to use
- `<image_file>` — path to the input image (any format OpenCV can read: JPEG, PNG, BMP, etc.)
- `<kernel_file>` — path to a CSV file containing the convolution kernel

**Example:**

```bash
mpirun -np 4 ./convolution input.jpg blur_kernel.csv
```

## Kernel File Format

The kernel file is a comma-separated list of integers representing a square matrix, flattened in row-major order. The total number of values must be a perfect square (e.g., 9 for a 3×3 kernel, 25 for a 5×5 kernel).

**Example — 3×3 box blur (`blur_kernel.csv`):**

```
1,1,1
1,1,1
1,1,1
```

**Example — 3×3 edge detection:**

```
-1,-1,-1
-1,8,-1
-1,-1,-1
```

The program normalizes each pixel's accumulated value by the sum of absolute kernel weights actually used at that pixel, so kernels do not need to be pre-normalized.

## Output

The program produces the following:

- `Serial_final_Image_0.jpg` — the serially convolved image
- `Parallel_final_Image_0.jpg` — the parallel convolved image
- Display windows showing the original image, the serial result, the parallel result, and each process's received and modified image segment (note: `cv::waitKey` calls are commented out by default, so windows close immediately)
- Console output with the parallel execution time and a confirmation that the serial and parallel images are identical

## How It Works

### Domain Decomposition

The image is divided into roughly equal horizontal stripes using a `parallelRange` helper that distributes any remainder rows across the lowest-ranked processes.

### Halo Exchange

Each process receives `haloRows = floor(kernel_dim / 2)` extra rows above and below its assigned stripe so that the convolution at the stripe's boundary rows can be computed correctly. Rank 0 only needs halo rows below its stripe; the last rank only needs halo rows above; all other ranks need halos on both sides.

### Custom MPI Datatypes

Two derived datatypes are created to make scattering and gathering cleaner:

- `Vec3b` — a contiguous type wrapping three `unsigned char`s (one BGR pixel)
- `Row_Pixels` — a contiguous type wrapping one full row of `Vec3b` values

These let `MPI_Scatterv` and `MPI_Gatherv` operate naturally on rows of pixels.

### Communication Pattern

- `MPI_Bcast` distributes the kernel size, the kernel itself, and the image dimensions
- `MPI_Scatterv` distributes the image rows (with halos) to each process
- `MPI_Gatherv` collects the processed (non-halo) rows back on rank 0

### Verification

After the parallel result is assembled, rank 0 compares it pixel-by-pixel and channel-by-channel against the serial result and prints a success or mismatch message.

## Notes

- The image must be tall enough that each process gets at least one row plus its halo. With very small images and many processes, behavior may be incorrect.
- The `cv::imshow` / `cv::waitKey` calls open OpenCV display windows on every rank. On a cluster without a display, comment these out or use `MPI_Comm_rank == 0` guards.
- The program assumes the image is a 3-channel BGR image (`CV_8UC3`).
