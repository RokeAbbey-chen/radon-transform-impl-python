import cv2
import numpy as np


class RadonAlgor(object):

    @staticmethod
    def __transLine(img, sin, cos, rho):
        """
        针对sin, cos, rho所代表的直线进行变换
        :param img:
        :param sin:
        :param cos:
        :return:
        """
        minY, maxY = 0, img.shape[0] - 1
        _sum = 0
        # img = img.copy().astype(float)
        if np.abs(cos) > 1e-4:
            for y in range(minY, maxY):
                x = (rho - y * sin) / cos  # cos为0的问题
                floorX = int(np.floor(x))
                if 0 < floorX + 1 < img.shape[1]:
                    _sum += img[y, floorX + 1] * (x - floorX)
                if 0 <= floorX < img.shape[1]:
                    _sum += img[y, floorX] * (floorX + 1 - x)
        else:
            """直线横着的时候"""
            for x in range(0, img.shape[1]):
                y = rho
                floorY = int(np.floor(y))
                if 0 <= floorY + 1 < img.shape[0]:
                    _sum += img[floorY + 1, x] * (y - floorY)
                if 0 <= floorY < img.shape[0]:
                    _sum += img[floorY, x] * (floorY + 1 - y)
        return _sum

    @staticmethod
    def radonTransform(img, rhoStep: [float, int], thetaStep: [float, int]):
        cls = RadonAlgor
        maxRho = np.ceil(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
        rhoRange, thetaRange = list(cls.floatRange(0, maxRho, rhoStep)), list(cls.floatRange(0, 180, thetaStep))
        dst = cls.createRhoThetaMat(rhoRange, thetaRange)

        for ti, theta in enumerate(thetaRange):
            """theta先算，省的重复计算sin, cos"""
            sin = np.sin(theta * np.pi / 180)
            cos = np.cos(theta * np.pi / 180)
            if 45 == ti:
                print("ti = 45")
            for ri, rho in enumerate(rhoRange):
                # dst[ti, ri] =
                dst[ti, ri] = cls.__transLine(img, sin, cos, rho)
                print(f"ti:{ti}, theta:{theta}, ri:{ri}, rho:{rho}, dst[ti,ri]:{dst[ti, ri]}")
        return dst

    @staticmethod
    def createRhoThetaMatWithImg(img, rhoStep: [float, int], thetaStep: [float, int]):
        """
        返回纵坐标为theta, 横坐标为rho的矩阵
        :param img:
        :param rhoStep:
        :param thetaStep:
        :return:
        """
        cls = RadonAlgor
        maxRho = np.ceil(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
        return cls.createRhoThetaMat(cls.floatRange(0, maxRho, rhoStep), cls.floatRange(0, 360, thetaStep))

    @staticmethod
    def createRhoThetaMat(rhoRange, thetaRange):
        """
        返回纵坐标为theta, 横坐标为rho的矩阵, 注意theta是第一维, rho是第二维
        :param rhoRange:
        :param thetaRange:
        :return:
        """

        def getLength(rg):
            if type(rg) == int:
                return rg
            elif type(rg) == list:
                return len(rg)
            else:
                return len(list(rhoRange))

        width = getLength(rhoRange)
        height = getLength(thetaRange)

        # if type(thetaRange) != list:
        #     thetaRange = list(thetaRange)
        # lengthRHO, lengthTheta = len(rhoRange), len(thetaRange)
        lengthRHO, lengthTheta = width, height
        dst = np.zeros((lengthTheta, lengthRHO), dtype=float)
        return dst

    @staticmethod
    def floatRange(start: [float, int], end: [float, int], step: [float, int]):
        if type(start) == type(end) == type(step) == int:
            for i in range(start, end, step):
                yield i
        else:
            if (end - start) * step <= 0:
                """递增/递减方向必须一致"""
                return
            while start < end:
                yield start
                start += step


class RadonAlgor2(RadonAlgor):

    @staticmethod
    def radonTransform(img, thetaStep: [float, int]):
        """
        :param img: 输入必须是灰度图
        :param thetaStep:
        :return:
        """
        spCls = RadonAlgor
        cls = RadonAlgor2
        thetaRange = spCls.floatRange(0, 180, thetaStep)
        # pMat, ((pt, pb), (pl, pr)) = cls.getSuitableMat(img.shape, img.dtype)
        diagImg, _ = cls.padding(img)
        center = (diagImg.shape[1] >> 1, diagImg.shape[0] >> 1)
        dst = spCls.createRhoThetaMat(diagImg.shape[0], int(np.floor(180 / thetaStep)))
        for ti, theta in enumerate(thetaRange):
            rotateMat = cv2.getRotationMatrix2D(center, theta, 1.0)
            rotated = cv2.warpAffine(diagImg, rotateMat, diagImg.shape)
            """如果axis=0，则theta代表的就是原点与直线垂足的连线关于x正半轴的夹角
            如果axis=1，则theta代表的是直线与x正半轴的夹角"""
            dst[ti, :] = rotated.sum(axis=0)
        return dst

    @staticmethod
    def getSuitableMat(shape, dtype):
        """
        获取一个正方形矩阵，矩阵的长于宽等于等于max(shape[:2])
        :param shape:
        :param dtype:
        :return: 返回目标矩阵并且返回长宽的增长值 ((t, b), (l, r))代表原矩阵到目标矩阵所需要的padding数值
        """
        diagLen = int(np.ceil(max(shape[:2]) * np.sqrt(2)))
        bg = np.zeros((diagLen, diagLen), dtype=dtype)
        t, l = (diagLen - shape[0]) >> 1, (diagLen - shape[1]) >> 1
        b, r = diagLen - t - shape[0], diagLen - l - shape[1]
        return bg, ((t, b), (l, r))

    @staticmethod
    def padding(img):
        cls = RadonAlgor2
        suitResult = cls.getSuitableMat(img.shape, img.dtype)
        diagMat, ((pt, pb), (pl, pr)) = suitResult
        diagMat[pt: -pb, pl: -pr] = img
        return suitResult


def main0():
    img = cv2.imread(r"D:\mywork\workspace\radon\radon-transform-impl-py\imgs\6.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayFloat = gray.copy().astype(float) / 255.0
    radonMat = RadonAlgor2.radonTransform(grayFloat, 1)
    radonImg = (radonMat - radonMat.min()) / max(1, radonMat.max() - radonMat.min())
    cv2.imshow("gray:", gray)
    cv2.imshow("radonImg:", radonImg)
    cv2.waitKey(0)


def main1():
    path = r"D:\mywork\workspace\radon\radon-transform-impl-py\imgs\3.png"
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrix = cv2.getRotationMatrix2D((600, 300), -60, 1.0)
    warped = cv2.warpAffine(gray, matrix, (1200, 600))
    cv2.imshow("warped", warped)
    cv2.waitKey(0)
    print(matrix)
    pass

if __name__ == '__main__':
    main0()
