import cv2
import argparse
import os
import numpy as np

calibration_data_path = "camera_calibration.npz"

def capture_webcam():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
            
        print("Press 's' to save a frame, 'q' to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            cv2.imshow('Webcam Feed', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                if not os.path.exists('images'):
                    os.makedirs('images')
                frame_num = 1
                while os.path.exists(f'images/frame_{frame_num}.png'):
                    frame_num += 1
                cv2.imwrite(f'images/frame_{frame_num}.png', frame)
                print(f"Frame saved as 'images/frame_{frame_num}.png'")

            if key & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during capture: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def delete_images():
    if os.path.exists('images'):
        for filename in os.listdir('images'):
            file_path = os.path.join('images', filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print("No images directory found.")

def calibrate_camera():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = os.listdir('images') if os.path.exists('images') else []
    if not images:
        print("No images found in the 'images' directory.")
        return

    gray = None
    for fname in images:
        img = cv2.imread(os.path.join('images', fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Check if we have any object/image points before calibrating
    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("No valid chessboard corners found. Make sure your images contain a visible 9x6 chessboard pattern.")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Camera calibrated successfully.")
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
            print(f"Re-projection error for image {i+1}: {error}")
        
        total_error = mean_error/len(objpoints)
        print(f"Average re-projection error: {total_error}")

        np.savez(calibration_data_path, camera_matrix=mtx, dist_coeffs=dist)
        print(f"Calibration data saved to '{calibration_data_path}'.")
    else:
        print("Camera calibration failed.")

def undistort_image():
    if not os.path.exists(calibration_data_path):
        print(f"Calibration data file '{calibration_data_path}' not found. Please calibrate first.")
        return

    with np.load(calibration_data_path) as data:
        mtx = data['camera_matrix']
        dist = data['dist_coeffs']
    
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:", dist)
    print("Max distortion coefficient magnitude:", np.max(np.abs(dist)))
    
    images = os.listdir('images') if os.path.exists('images') else []
    if not images:
        print("No images found in the 'images' directory.")
        return

    img_path = os.path.join('images', images[0])
    img = cv2.imread(img_path)

    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    roi_x, roi_y, roi_w, roi_h = roi
    dst = dst[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    output_path = os.path.join('images', 'undistorted_' + os.path.basename(img_path))
    cv2.imwrite(output_path, dst)
    print(f"Saved undistorted image to: {output_path}")

    cv2.imshow('Undistorted Image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam Capture Tool")
    parser.add_argument(
        'mode', 
        choices=['capture', 'delete', 'calibrate', 'show'], 
        help="Mode to run the tool in: 'capture' to start capturing webcam frames, 'delete' to remove all saved images, 'calibrate' to calibrate the camera using checkerboard images, 'show' to undistort an image"
    )

    args = parser.parse_args()

    if args.mode == 'capture':
        capture_webcam()
    elif args.mode == 'delete':
        delete_images()
    elif args.mode == 'calibrate':
        calibrate_camera()
    elif args.mode == 'show':
        undistort_image()
