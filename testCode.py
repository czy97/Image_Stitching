from pano import *

imageList = 'ImageList/imageList4.txt'
fp = open(imageList, 'r')
filenames = [each.rstrip('\r\n') for each in fp.readlines()]
prefix1 = 'resImage/'
prefix2 = 'resMatch/'

storeName = prefix1 + filenames[0].split('/')[-1].split('.')[0] + '_' + filenames[1].split('/')[-1].split('.')[0]
STORE_MATCH = prefix2 + filenames[0].split('/')[-1].split('.')[0] + '_' + filenames[1].split('/')[-1].split('.')[0]
#ORB based on FAST
#ORB SIFT SURF BRIEF
detector = 'SURF'
descriptor = 'SURF'

storeName = storeName + '_' + detector+ '_' + descriptor + '.jpg'
STORE_MATCH = STORE_MATCH + '_' + detector+ '_' + descriptor + '.jpg'

s = Stitch(imageList,detector = detector,descriptor = descriptor)
s.leftshift()
s.drawMatch(filepath = STORE_MATCH)

print("done")
cv2.imwrite(storeName, s.leftImage)
print("image written")
cv2.destroyAllWindows()

