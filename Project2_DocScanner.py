import cv2
import numpy as np
#membuat document scanner
#first, membuat preprocessing image
widthImg = 640
heightImg = 480

cap = cv2.VideoCapture(0)
cap.set(3,widthImg)
cap.set(4,heightImg)
cap.set(10,100)

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5) , 1)
    imgCanny = cv2.Canny(imgBlur, 200,200)
    kernel = np.ones((5,5))
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDilation, kernel, iterations=1)
    
    #kalo ingin mengetahui setiap isi dari variabel diatas, kita bisa 
    #menuliskannya ke return, agar ditampilkan. Contoh: return imgCanny
    return imgThreshold

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1,(255,0,0), 3)
            peri = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    #kita ingin memberikan penanda pada setiap ujung doc yg discan
    cv2.drawContours(imgContour, biggest, -1,(255,0,0), 15)
    return biggest
        
            # print (approx)
            # #mendefinsikan banyak corner
            # objCor = len(approx)
            # #kita membuat kotak utk mendeteksi shape
            # x, y , w, h = cv2.boundingRect(approx)
            
#fungsi utk mendapatkan nilai contour
def reorder (myPoints):
    #jadi kita ingin mendapatkan titik sudut dari doc yg kita scan
    myPoints = myPoints.reshape((4,2))
    #4,1,2 didapat dari hasil scan dri getWarp
    myPointsNew = np.zeros((4,1,2), np.int32)
    
    #kita ingin jumlahkan matrix arraynya, agar menjadi satu baris (bkn matrix)
    add = myPoints.sum(1)
    #print("add", add)
    
    #kita ingin mencari nilai dari [0,0]
    myPointsNew[0] = myPoints[np.argmin(add)]
    #kita ingin mencari nilai dari [widthimg, heightimg]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis = 1)
    #kita ingin mencari nilai dari [widthImg,0]
    myPointsNew[1] = myPoints[np.argmin(diff)]
    #kita ingin mencari nilai dari [0,heightimg]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #maksud dari nilai 0 ini adalah, nilai yg paling rendah pada 1 matrix
    #print("New Point", myPointsNew)
    return myPointsNew
    
#mengambil gambar dalam gambar
def getWarp(img, biggest):
    biggest = reorder(biggest)
    #print(biggest)
    point1 = np.float32(biggest)
    point2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg, heightImg]])
    #sampe sini, ketika nilai warpnya !=0, maka warp nya akan salah, karena
    #kita mengambil scara real time dgn berbagai sudut
    matrix = cv2.getPerspectiveTransform(point1, point2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    
    #membuat cropping img, agar lebih rapi
    imgCrop = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgCrop = cv2.resize(imgCrop, (widthImg, heightImg))
    
    return imgCrop

#memberikan stack images
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg,heightImg))
    #mencopy value img ke imgContour
    imgContour = img.copy()
    #menuliskan variabel imgThreshold kembali di while true
    imgThreshold = preProcessing(img)
    biggest = getContours(imgThreshold)
    #print (biggest)
    if biggest.size !=0:
        
        imgWarped = getWarp(img, biggest)

        imageArray = ([imgContour, imgWarped])
        cv2.imshow("Video Zoomed", imgWarped)
    else:
        imageArray = ([img, imgContour])
    
    stackedImages = stackImages(0.6, imageArray)
    cv2.imshow("Doc Scanner", stackedImages)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.release
cv2.destroyAllWindows()
