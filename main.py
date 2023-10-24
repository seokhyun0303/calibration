import numpy as np
import cv2
import sys
import glob

#galaxy s20 camera

def calibration():
    CHECKERBOARD = (5, 4)  # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    frames = []
    # 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
    objpoints = []
    # 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
    imgpoints = []
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= 5
    print(objp)


    cap = cv2.VideoCapture('checkerboard3.mp4')
    if not cap.isOpened():
        print('File open failed!')
        cap.release()

        sys.exit()

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    for i in range(len(frames)):
        if (i%10==0):

            img = frames[i]
            # 그레이 스케일로 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 체커보드 코너 찾기
            # 이미지에서 원하는 개수의 코너가 발견되면 ret = true
            ret, corners = cv2.findChessboardCorners(gray,
                                                     CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # 원하는 개수의 코너가 감지되면,
            # 픽셀 좌표 미세조정 -> 체커보드 이미지 표시
            if ret == True:
                objpoints.append(objp)
                # 주어진 2D 점에 대한 픽셀 좌표 미세조정
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # 코너 그리기 및 표시
    print(objpoints[0])
    print(imgpoints[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez('cdata', mtx=mtx, dist=dist, objp=objp)
    cap.release()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    calibration()


