import numpy as np
import cv2
import sys
import json
import math
import glob

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    return img
def checkerboardshow():
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter('output.avi', fourcc, 30, (720, 764))
    CHECKERBOARD = (5, 4)  # 체커보드 행과 열당 내부 코너 수
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    axis = np.float32([[15, 0, 0], [0, 15, 0], [0, 0, 15]]).reshape(-1, 3)

    cdata = np.load('cdata.npz')
    objp = cdata['objp']
    for i in objp:
        temp = i[1]
        i[1] = i[2]
        i[2] = temp

    mtx = cdata['mtx']
    dist = cdata['dist']
    CPdata =[]

    frames = []
    cap = cv2.VideoCapture('checkerboard.mp4')
    if not cap.isOpened():
        print('File open failed!')
        cap.release()
        sys.exit()

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    idx = 0
    for f in frames:
        img = f
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img, corners2.astype(int), imgpts.astype(int))
            cvec = np.array([[1., 0., 0., 0.],
                            [0., -1., 0., 0.],
                            [0., 0., -1., 0.],
                            [0., 0., 0., 1.]])
            rmat, r = cv2.Rodrigues(rvecs)
            rmat = np.linalg.inv(rmat)
            tvecs = -rmat@tvecs
            rt = np.array([[rmat[0][0], rmat[0][1], rmat[0][2], tvecs[0][0]],
                          [rmat[1][0], rmat[1][1], rmat[1][2], tvecs[1][0]],
                          [rmat[2][0], rmat[2][1], rmat[2][2], tvecs[2][0]],
                          [0., 0., 0., 1.]])

            newrt = cvec @ rt
            newrvecs = newrt[:3, :3]
            newtvecs = newrt[:3, 3:]

            tr = newrvecs[0][0] + newrvecs[1][1] + newrvecs[2][2]

            if (tr > 0):
                S = math.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (newrvecs[2][1] - newrvecs[1][2]) / S
                qy = (newrvecs[0][2] - newrvecs[2][0]) / S
                qz = (newrvecs[1][0] - newrvecs[0][1]) / S
            elif ((newrvecs[0][0] > newrvecs[1][1]) and (newrvecs[0][0] > newrvecs[2][2])):
                S = math.sqrt(1.0 + newrvecs[0][0] - newrvecs[1][1] - newrvecs[2][2]) * 2
                qw = (newrvecs[2][1] - newrvecs[1][2]) / S
                qx = 0.25 * S
                qy = (newrvecs[0][1] + newrvecs[1][0]) / S
                qz = (newrvecs[0][2] + newrvecs[2][0]) / S
            elif (newrvecs[1][1] > newrvecs[2][2]):
                S = math.sqrt(1.0 + newrvecs[1][1] - newrvecs[0][0] - newrvecs[2][2]) * 2
                qw = (newrvecs[0][2] - newrvecs[2][0]) /S
                qx = (newrvecs[0][1] + newrvecs[1][0]) /S
                qy = 0.25 * S
                qz = (newrvecs[1][2] + newrvecs[2][1]) /S
            else:
                S = math.sqrt(1.0 + newrvecs[2][2] - newrvecs[0][0] - newrvecs[1][1]) * 2
                qw = (newrvecs[1][0] - newrvecs[0][1]) /S
                qx = (newrvecs[0][2] + newrvecs[2][0]) /S
                qy = (newrvecs[1][2] + newrvecs[2][1]) /S
                qz = 0.25 * S


            rdata = [qx, qy, qz, qw]
            data = {"frame": idx, "rvecs": rdata, "tvecs": newtvecs.reshape(-1).tolist()}
            CPdata.append(data)
            idx += 1
        cv2.imshow('img', img)
        #out.write(img)
        cv2.waitKey(5)

    cpd = {"data": CPdata}
    with open("CPdata.json", 'w') as outfile:
        json.dump(cpd, outfile, indent=4)
    cdata.close()
    cap.release()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    checkerboardshow()


