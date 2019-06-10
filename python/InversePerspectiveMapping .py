import numpy as np
import cv2
class InversePerspectiveMapping():
    '''需要导入cv2,numpy两个库'''
    '''InputMat 代表输入图片
       CameraInfo 代表相机参数'''

    def __init__(self,InputMat,CameraInfo,VpPortion ):
        '''CameraInfo=[cameraInfo_focalLengthX,cameraInfo_focalLengthY,cameraInfo_opticalCenterX,cameraInfo_opticalCenterY,cameraInfo_cameraHeight,
        cameraInfo_pitch,cameraInfo_yaw,cameraInfo_imageWidth,cameraInfo_imageHeight]'''

        '''impInfo=[ipmInfo_ipmWidth, ipmInfo_ipmHeight, ipmInfo_ipmLeft,ipmInfo_ipmRight,
        ipmInfo_ipmTop,ipmInfo_ipmBottom,ipmInfo_ipmInterpolation,ipmInfo_ipmVpPortion]'''
        self.InputMat = InputMat
        self.CameraInfo = CameraInfo
        self.IpmInfo = [0,0,0,0,0,0,0,0]
        self.IpmInfo[7] = VpPortion


    def GetVanishingPoint(self):
        """该函数的功能是计算消失点"""

        '''cameraInfo[0]= focalLengthX; cameraInfo[1]=focalLengthY;  cameraInfo[5]= pitch; cameraInfo[6]=yaw'''
        cameraInfo = self.CameraInfo
        A_1 = cameraInfo[6] * np.pi / 180
        A_2 = cameraInfo[5] * np.pi / 180
        vpp = np.array([np.sin(cameraInfo[6] * np.pi / 180) / np.cos(cameraInfo[5] * np.pi / 180),
                        np.cos(cameraInfo[6] * np.pi / 180) / np.cos(cameraInfo[5] * np.pi / 180), 0])

        '''绕Z轴旋转'''
        tyawp = np.array([[np.cos(cameraInfo[6] * np.pi / 180), -np.sin(cameraInfo[6] * np.pi / 180), 0],
                          [np.sin(cameraInfo[6] * np.pi / 180), np.cos(cameraInfo[6] * np.pi / 180), 0],
                          [0, 0, 1]])

        '''绕X轴旋转'''
        tpitchp = np.array([[1, 0, 0],
                            [0, -np.sin(cameraInfo[5] * np.pi / 180), -np.cos(cameraInfo[5] * np.pi / 180)],
                            [0, np.cos(cameraInfo[5] * np.pi / 180), -np.sin(cameraInfo[5] * np.pi / 180)]])

        '''旋转矩阵'''
        transform = np.dot(tyawp, tpitchp)

        t1p = np.array([[cameraInfo[0], 0, cameraInfo[0]],
                        [0, cameraInfo[1], cameraInfo[1]],
                        [0, 0, 1]])

        transform = np.dot(t1p, transform)

        vp = np.dot(transform, vpp)
        print(vp)

        return vp

    def TransformImage2Ground(self,uvLimitsp):
        '''cameraInfo[0]= focalLengthX; cameraInfo[1]=focalLengthY;  cameraInfo[2]=opticalCenterX;  cameraInfo[3]=opticalCenterY
        cameraInfo[4]=cameraHeight; cameraInfo[5]= pitch; cameraInfo[6]=yaw'''
        cameraInfo = self.CameraInfo
        row = uvLimitsp.shape[0]
        col = uvLimitsp.shape[1]
        inPoints4 = np.zeros((row + 2, col))
        inPoints4[0] = uvLimitsp[0]
        inPoints4[1] = uvLimitsp[1]
        inPoints4[2] = [1, 1, 1, 1]
        inPoints3 = inPoints4[0:3, :]

        c1 = np.cos(cameraInfo[5] * np.pi / 180)
        s1 = np.sin(cameraInfo[5] * np.pi / 180)
        c2 = np.cos(cameraInfo[6] * np.pi / 180)
        s2 = np.sin(cameraInfo[6] * np.pi / 180)
        '''
        matp = np.array([[-cameraInfo[4] * c2 / cameraInfo[0], cameraInfo[4] * s1 * s2 / cameraInfo[1],
                          (cameraInfo[4] * c2 * cameraInfo[2] / cameraInfo[0]) -
                          (cameraInfo[4] * s1 * s2 * cameraInfo[3] / cameraInfo[1]) - cameraInfo[4] * c1 * s2],
                         [cameraInfo[4] * s2 / cameraInfo[0], cameraInfo[4] * s1 * c2 / cameraInfo[1],
                          (-cameraInfo[4] * s2 * cameraInfo[2] / cameraInfo[0]) - (cameraInfo[4] * s1 * c2*cameraInfo[3] / cameraInfo[1]) -
                          cameraInfo[4] * c1 * c2],
                         [0, cameraInfo[4] * c1 / cameraInfo[1],
                          (-cameraInfo[4] * c1 * cameraInfo[3] / cameraInfo[1]) + cameraInfo[4] * s1],
                         [0, -c1 / cameraInfo[1], (c1 * cameraInfo[3] / cameraInfo[1]) - s1]])
        '''
        matp = np.array([[-cameraInfo[4] * c2 / cameraInfo[0], cameraInfo[4] * s1 * s2 / cameraInfo[1],
                          (np.dot(cameraInfo[4] * c2, cameraInfo[2]) / cameraInfo[0]) - (
                                      np.dot(cameraInfo[4] * s1 * s2, cameraInfo[3]) / cameraInfo[1]) - cameraInfo[
                              4] * c1 * s2],
                         [cameraInfo[4] * s2 / cameraInfo[0], cameraInfo[4] * s1 * c2 / cameraInfo[1],
                          (np.dot(-cameraInfo[4] * s2, cameraInfo[2]) / cameraInfo[0]) - (
                                      np.dot(cameraInfo[4] * s1 * c2, cameraInfo[3]) / cameraInfo[1]) - cameraInfo[
                              4] * c1 * c2],
                         [0, cameraInfo[4] * c1 / cameraInfo[1],
                          (np.dot(-cameraInfo[4] * c1, cameraInfo[3]) / cameraInfo[1]) + cameraInfo[4] * s1],
                         [0, -c1 / cameraInfo[1], (c1 * cameraInfo[3] / cameraInfo[1]) - s1]])

        inPoints4 = np.dot(matp, inPoints3)
        inPointsr4 = inPoints4[3, :]
        div = inPointsr4
        inPoints4[0, :] = inPoints4[0, :] / div
        inPoints4[1, :] = inPoints4[1, :] / div
        inPoints2 = inPoints4[0:2, :]
        xyLimits = inPoints2

        return xyLimits

    def TransformGround2Image(self,xyGrid):
        cameraInfo = self.CameraInfo
        inPoints2 = xyGrid[0:2]
        col_1 = np.shape(xyGrid)[1]
        inPointsr3 = np.ones((1, col_1)) * (-cameraInfo[4])
        inPoints3 = np.zeros((3, col_1))
        inPoints3[0:2] = inPoints2
        inPoints3[2] = inPointsr3

        c1 = np.cos(cameraInfo[5] * np.pi / 180)
        s1 = np.sin(cameraInfo[5] * np.pi / 180)
        c2 = np.cos(cameraInfo[6] * np.pi / 180)
        s2 = np.sin(cameraInfo[6] * np.pi / 180)
        '''
        matp = np.array([[cameraInfo[0] * c2 + 0 * cameraInfo[2],-cameraInfo[0] * c1*s2 + s1* cameraInfo[2],cameraInfo[0] * s1*s2 + c1 * cameraInfo[2]],
                         [s2*cameraInfo[1],c1*c2*cameraInfo[1]+s1*cameraInfo[3],-s1*c2*cameraInfo[1]+c1*cameraInfo[3]],
                         [0, s1, c1]])
        print(np.linalg.inv(matp))
        '''
        matp = np.array([[cameraInfo[0] * c2 + c1 * s2 * cameraInfo[2], -cameraInfo[0] * s2 + c1 * c2 * cameraInfo[2],
                          - s1 * cameraInfo[2]],
                         [s2 * (-cameraInfo[1] * s1 + c1 * cameraInfo[3]),
                          c2 * (-cameraInfo[1] * s1 + c1 * cameraInfo[3]), -cameraInfo[1] * c1 - s1 * cameraInfo[3]],
                         [c1 * s2, c1 * c2, -s1]])

        inPoints3 = np.dot(matp, inPoints3)
        inPointsr3 = inPoints3[2]
        div = inPointsr3
        inPoints3[0] = inPoints3[0] / div
        inPoints3[1] = inPoints3[1] / div
        inPoints2 = inPoints3[0:2]
        uvGrid = inPoints2

        return uvGrid

    def main(self):

        outImage = np.zeros((self.CameraInfo[8], self.CameraInfo[7]))
        R = self.InputMat[:, :, 1]  # 其作用类似于cvtColor(I.R,CV_BGR2GRAY)

        u = R.shape[0]
        v = R.shape[1]

        vpp = self.GetVanishingPoint()
        vp_x = vpp[0]
        vp_y = vpp[1]
        print("vp_y" ,vp_y )

        '''IpmInfo赋值'''
        self.IpmInfo[0]=int(self.CameraInfo[7])
        self.IpmInfo[1] = int(self.CameraInfo[8])
        self.IpmInfo[2] = 0
        self.IpmInfo[3] = int(self.CameraInfo[7] - 1)
        eps = self.IpmInfo[7] * v
        print(eps)
        self.IpmInfo[4] = vp_y + eps
        self.IpmInfo[5] = int(self.CameraInfo[8] - 1)
        self.IpmInfo[6] = 0

        print(self.IpmInfo)

        uvLimitsp = np.array([[vp_x, self.IpmInfo[3],self.IpmInfo[2], vp_x], [self.IpmInfo[4],self.IpmInfo[4], self.IpmInfo[4],self.IpmInfo[5]]])

        xyLimits = self.TransformImage2Ground(uvLimitsp)

        row1 = xyLimits[0, :]
        row2 = xyLimits[1, :]
        xfMin = min(row1)
        xfMax = max(row1)
        yfMin = min(row2)
        yfMax = max(row2)

        outRow = outImage.shape[0]
        outCol = outImage.shape[1]
        stepRow = (yfMax - yfMin) / outRow
        stepCol = (xfMax - xfMin) / outCol
        xyGrid = np.zeros((2, outRow * outCol))
        y = yfMax - 0.5 * stepRow

        for i in range(outRow):
            x = xfMin + 0.5 * stepCol
            for j in range(outCol):
                xyGrid[0, i * outCol + j] = x
                xyGrid[1, i * outCol + j] = y
                x = x + stepCol
            y = y - stepRow

        uvGrid = self.TransformGround2Image(xyGrid)

        means_1 = 0
        RR = np.zeros((u, v))
        for i in range(u):
            for j in range(v):
                means_1 = means_1 + R[i, j]
                RR[i, j] = float(R[i, j]) / 255
        means_2 = means_1 / (u * v)
        means = means_2 / 255

        for i in range(outRow):
            for j in range(outCol):
                ui = uvGrid[0, i * outCol + j]
                vi = uvGrid[1, i * outCol + j]
                if (ui < self.IpmInfo[2] or ui > self.IpmInfo[3] or vi < self.IpmInfo[4] or vi >self.IpmInfo[5]):
                    outImage[i, j] = means
                else:
                    x1 = int(ui)
                    x2 = int(ui + 1)
                    y1 = int(vi)
                    y2 = int(vi + 1)
                    x = ui - float(x1)
                    y = vi - float(y1)
                    val = float(RR[y1, x1]) * (1 - x) * (1 - y) + float(RR[y1, x2]) * x * (1 - y) + float(RR[y2, x1]) * (1 - x) * y + float(RR[y2, x2]) * x * y
                    outImage[i, j] = val

        return outImage







I = cv2.imread("Images/02.jpg")
CameraInfo =[309.4362,344.2161,605.3490,500,1310,7.4,0.0,1280,720]

k = InversePerspectiveMapping(I,CameraInfo,0.18)
m = k.main()

cv2.imshow("result2",m)
cv2.waitKey(0)


