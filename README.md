# Camera Calibration Tool

A Python tool for camera calibration using a checkerboard pattern. This tool helps remove lens distortion from images by calculating the camera's intrinsic parameters.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- A printed checkerboard pattern (9x6 corners)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NickHerrig/camera-calibration.git
   cd camera-calibration
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The tool provides four main operations:

### 1. Capture Calibration Images

Capture multiple images of a checkerboard pattern from different angles:

```bash
python calibrate.py capture
```

- Press 's' to save a frame
- Press 'q' to quit
- Images are saved in the `images/` directory

Tips for capturing good calibration images:
- Use a flat checkerboard pattern (9x6 corners)
- Capture 10-20 images from different angles
- Ensure the entire checkerboard is visible in each image
- Avoid motion blur
- Cover different areas of the camera view

### 2. Calibrate the Camera

Process the captured images to calculate camera parameters:

```bash
python calibrate.py calibrate
```

This will:
- Detect checkerboard corners in all images
- Calculate camera matrix and distortion coefficients
- Save calibration data to `camera_calibration.npz`
- Display re-projection error for quality assessment

### 3. Test Calibration

View the results by undistorting an image:

```bash
python calibrate.py show
```

This will:
- Load the calibration data
- Undistort the first image in the `images/` directory
- Save the undistorted image with prefix 'undistorted_'
- Display the result

### 4. Clean Up

Remove all captured images:

```bash
python calibrate.py delete
```

## Output Files

- `camera_calibration.npz`: Contains the camera matrix and distortion coefficients
- `images/`: Directory containing captured and undistorted images

## Troubleshooting

1. If the webcam doesn't open:
   - Check if another application is using the camera
   - Verify your webcam is properly connected

2. If checkerboard detection fails:
   - Ensure good lighting conditions
   - Keep the checkerboard pattern flat
   - Make sure the entire pattern is visible

3. If calibration results are poor:
   - Capture more images from different angles
   - Ensure the checkerboard pattern is clearly visible
   - Check the re-projection error (lower is better)

## License
Apache 2.0