import numpy as np
import cv2 as cv
import glob
import pickle


def get_params(cam_nr):
    print(f'=== calibrate cam nr {cam_nr}')
    
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    objp *= 26.3 # mm size of one square

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(f'cam{cam_nr}*.png')
    i = 0

    print('--- detecting corners')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8,6), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # # Draw and display the corners
            # cv.drawChessboardCorners(img, (8,6), corners2, ret)
            # cv.imwrite(f'corners{i}.png', img)
            i += 1
            print(f'image {i} analysed')

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('--- calculating params')

    print(ret)
    print(mtx)
    print(dist)
    # print(rvecs)
    # print(tvecs)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )
    
    return mtx, dist, mean_error/len(objpoints)


if __name__ == '__main__':
    cam_numbers = [0, 2]
    parameters = {}
    
    for i in cam_numbers:
        mtx, dist, error = get_params(i)
        parameters[i] = [mtx, dist, error]
    
    print('=== result dict')
    print(parameters)
    
    pickle.dump(parameters, open('cam_parameters.pkl', 'wb'))





# cam 0
# 0.809450589425117
# [[1.42371135e+03 0.00000000e+00 1.10132488e+03]
#  [0.00000000e+00 1.42231476e+03 8.00078952e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# [[-0.00826651  0.04511153  0.00160556 -0.00140585 -0.29227446]]
# total error: 0.06995445680889006

# cam 2
# 0.7765351939151545
# [[1.42884643e+03 0.00000000e+00 1.05587343e+03]
#  [0.00000000e+00 1.43447580e+03 7.34000939e+02]
# #  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# [[-0.04246852  0.12470708 -0.00186713  0.00116219 -0.18077664]]
# total error: 0.08342191127542727