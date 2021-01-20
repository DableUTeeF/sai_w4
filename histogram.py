import cv2
import numpy as np


if __name__ == '__main__':
    im1 = cv2.imread('/media/palm/BiggerData/denso/testdata/subs/20200415074224_f.jpg')
    im2 = cv2.imread('/media/palm/BiggerData/denso/testdata/subs/20200415074533_f.jpg')
    cv2.imshow('1', im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)[:30, 470:]
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)[:30, 470:]
    cv2.imshow('2', im1)
    cv2.waitKey()

    hist1 = cv2.calcHist([im1], [0], None, [64], [0, 255])
    # cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    max1 = np.argsort(hist1, 0)[::-1]
    sum1 = np.sum(max1[:5])

    hist2 = cv2.calcHist([im2], [0], None, [64], [0, 255])
    # cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    max2 = np.argsort(hist2, 0)[::-1]
    sum2 = np.sum(max2[:5])

    compare1 = cv2.compareHist(hist1, hist2, 0)
    print(sum1, sum2, sum1/sum2)
