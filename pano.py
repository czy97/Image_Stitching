import numpy as np
import cv2
import sys
from matchers import matchers
import time


class Stitch:
    def __init__(self, args,detector = 'SURF',descriptor = 'BRIEF'):
        self.path = args
        fp = open(self.path, 'r')
        filenames = [each.rstrip('\r\n') for each in fp.readlines()]


        self.images = [cv2.resize(cv2.imread(each), (480, 320)) for each in filenames]
        # self.images = [cv2.imread(each) for each in filenames]
        self.matcher_obj = matchers(detector = detector,descriptor = descriptor)




    def leftshift(self):

        a = self.images[0]
        b = self.images[1]

        H = self.matcher_obj.match(a, b, 'left')
        xh = np.linalg.inv(H)
        ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
        ds = ds / ds[-1]
        f1 = np.dot(xh, np.array([0, 0, 1]))
        f1 = f1 / f1[-1]
        xh[0][-1] += abs(f1[0])
        xh[1][-1] += abs(f1[1])
        ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
        offsety = abs(int(f1[1]))
        offsetx = abs(int(f1[0]))
        dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
        tmp = cv2.warpPerspective(a, xh, dsize)

        tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b

        self.leftImage = tmp
    def drawMatch(self,filepath = 'match.jpg'):
        self.matcher_obj.drawMatch(filepath)




if __name__ == '__main__':
    try:
        args = sys.argv[1]
    except:
        args = "txtlists/files1.txt"
    finally:
        print("Parameters : ", args)

    fp = open(args, 'r')
    filenames = [each.rstrip('\r\n') for each in fp.readlines()]
    prefix = 'resImage/'

    prefix1 = 'resImage/'
    prefix2 = 'resMatch/'
    storeName = prefix1 + filenames[0].split('/')[-1].split('.')[0] + '_' + filenames[1].split('/')[-1].split('.')[0]
    STORE_MATCH = prefix2 + filenames[0].split('/')[-1].split('.')[0] + '_' + filenames[1].split('/')[-1].split('.')[0]
    # ORB based on FAST
    # ORB SIFT SURF BRIEF
    detector = 'SURF'
    descriptor = 'SURF'

    storeName = storeName + '_' + detector + '_' + descriptor + '.jpg'
    STORE_MATCH = STORE_MATCH + '_' + detector + '_' + descriptor + '.jpg'

    s = Stitch(args, detector=detector, descriptor=descriptor)
    s.leftshift()
    s.drawMatch(filepath=STORE_MATCH)

    print("done")
    cv2.imwrite(storeName, s.leftImage)
    print("image written")
    cv2.destroyAllWindows()

