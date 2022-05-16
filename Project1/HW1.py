import cv2
import numpy as np
img = cv2.imread('doc_db/2_noise.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#median filter,it's better against salt/pepper noise
def median(imggr): #manual implementation of median filter
    imgf=imggr
    for x in range(0,(imggr.shape[0]-1)):
        for y in range(0,(imggr.shape[1]-1)):
            window = [imggr[0, 0]] * 9
            window[0] = imggr[x-1,y-1]#analytic implementation of window,a 3rd for loop would increase the run time
            window[1] = imggr[x,y-1]
            window[2] = imggr[x+1,y-1]
            window[3] = imggr[x-1,y]
            window[4] = imggr[x,y]
            window[5] = imggr[x+1,y]
            window[6] = imggr[x-1,y+1]
            window[7] = imggr[x,y+1]
            window[8] = imggr[x+1,y+1]
            window.sort()#sorting and picking the median value from all the possible values in the next line
            imgf[x,y]=window[4]
    return imgf
filtered = median(gray)
_, bin2 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
str_ele = np.ones((40, 40))
morph1 = cv2.morphologyEx(bin2, cv2.MORPH_DILATE, str_ele)#to find sub-areas
morph1 = cv2.morphologyEx(morph1, cv2.MORPH_OPEN, str_ele)#to remove unwanted items
str_el = np.ones((7, 7))#to find words
morph2 = cv2.morphologyEx(bin2, cv2.MORPH_DILATE, str_el)#2nd to words
num, _, stats, _ = cv2.connectedComponentsWithStats(morph1)
x = stats[1:, cv2.CC_STAT_LEFT]# 1: to exclude the background,which is labeled 0
y = stats[1:, cv2.CC_STAT_TOP]
w = stats[1:, cv2.CC_STAT_WIDTH]
h = stats[1:, cv2.CC_STAT_HEIGHT]
bdn_area = []
bdn_boxes = []
area = []
endpoint_x = []
endpoint_y = []
words = []
gray_box = []
total_gray = []
mean_gray = []
sums = cv2.integral(gray)#summed area
for i in range(len(x)):
    cv2.rectangle(img, (x[i], y[i]), (x[i]+w[i], y[i]+h[i]), (0,0,0), 5)
    cv2.putText(img, '' + str(i+1), (x[i], y[i] + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    endpoint_x.insert(i, x[i]+w[i])
    endpoint_y.insert(i, y[i]+h[i])
    bdn_boxes.insert(i, bin2[y[i]:endpoint_y[i], x[i]:endpoint_x[i]])#crop binary img according to boundary boxes
    nums, _, _, _ = cv2.connectedComponentsWithStats(morph2[y[i]:endpoint_y[i], x[i]:endpoint_x[i]])#connectedComp to find num of labels,i.e. words
    words.insert(i, nums-1)#minus-1 cause background gets a label too
    area.insert(i, cv2.countNonZero(bdn_boxes[i]))#finding which pixels have non zero value,due to bin inv
    bdn_area.insert(i, w[i] * h[i])#total box area
    gray_box.insert(i,gray[y[i]:endpoint_y[i], x[i]:endpoint_x[i]])#crop grayscale img according to boundary boxes,to find mean gray value
    total_gray.insert(i,(w[i],h[i]))
    mean_gray.insert(i,(sums[total_gray[i]]/(bdn_area[i])))#mean gray for each sub-area
    print('Square for area ' + str(i+1), ' is ' + str(bdn_area[i]),' pixels, while text area is '+str(area[i]),
          'pixels,word approximated count is ' +str(words[i]),'and mean gray value is '+str(mean_gray[i]))
cv2.imwrite('bounds_x.png',img)
#########        END       ####################