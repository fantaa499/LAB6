import cv2
import numpy as np
from enum import Enum

whiteBW = 255
blackBW = 0

class Rotation(Enum):
    up = 0
    right = 1
    down = 2
    left = 3

class MatrP:
    # размерность матрицы улитки
    n = 3
    matr = np.zeros(n**2)

    def __init__ (self, matr, n = 3):
        self.n = n
        # устанавливаем начальное положение в середину

        # self.matr[8] = matr[0][0]
        # self.matr[1] = matr[0][1]
        # self.matr[2] = matr[0][2]
        #
        # self.matr[7] = matr[1][0]
        # self.matr[0] = matr[1][2]
        # self.matr[3] = matr[1][2]
        #
        # self.matr[6] = matr[2][0]
        # self.matr[5] = matr[2][1]
        # self.matr[4] = matr[2][2]

        i = int(matr.shape[0]/2)
        j = i
        # счетчик для прохода по вектору matr
        p = 0
        # текущее направление (вверх)
        mot = Rotation.up
        while p < self.n**2:
            self.matr[p] = matr[i][j]
            i, j = self.motion(mot, i, j)
            # смена направления движения если по какой-либо оси дошли до конца матрицы
            if (mot == Rotation.up or mot == Rotation.down) and \
               (i == n - 1 or i == 0):
                mot = Rotation((mot.value + 1) % 4)
            elif (mot == Rotation.left or mot == Rotation.right) and \
               (j == n - 1 or j == 0):
                mot = Rotation((mot.value + 1) % 4)

            p+=1



    def motion (self, k, i, j):
        if k == Rotation.up:
            return i-1, j+0
        if k == Rotation.right:
            return i+0, j+1
        if k == Rotation.down:
            return i+1, j+0
        if k == Rotation.left:
            return i+0, j-1

    def f (self, k):
        if k == 1:
            return 1
        else:
            return 8 * (k - 1)

    # Считаем количество переходов от черного к белому
    def A (self):
        previous = whiteBW
        k = 0
        # поиск переходов в последовательности 2,3,4..9
        for el in self.matr[2:]:
            if el == whiteBW and previous == blackBW:
                k += 1
            previous = el
        # поиск перехода в последовательности 9,2
        if self.matr[2] == whiteBW and previous == blackBW:
            k += 1
        return k

    def B(self):
        return int(sum(self.matr[2:])/whiteBW)

    def __getitem__ (self, ind):
        return self.matr[ind-1]






def skeletization (image):
    temple = image.copy()
    while ((not (temple == 1).all())):
        for k in range(2):
            temple = temple * 0 + 1
            for i in range(image.shape[0]-2):
                for j in range (image.shape[1]-2):
                    A = image[0+i:3+i, 0+j:3+j]
                    if A[1][1] == whiteBW:
                        P = MatrP(A)
                        # координаты элемента P[1]
                        Pi = i + 1
                        Pj = j + 1
                        counter = 0
                        if (2 <= P.B() <= 6):
                            counter+=1
                        if P.A() == 1:
                            counter+=1
                        if P[2]*P[4]*P[6] == 0 and k == 0:
                            counter+=1
                        if P[4]*P[6]*P[8] == 0 and k == 0 :
                            counter+=1
                        if P[2]*P[4]*P[8] == 0 and k == 1:
                            counter+=1
                        if P[2]*P[6]*P[8] == 0 and k == 1:
                            counter+=1
                        if counter == 4:
                            temple[Pi, Pj] = 0
            image = cv2.bitwise_and(image,
                                    image,
                                    mask=temple)
            # cv2.imshow("test3", image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    return image


##############
#   Задание 1
##############
# picture = cv2.imread("img/A.jpg", 0)
# _, pictureBW = cv2.threshold(picture, 240, 255, cv2.THRESH_BINARY)
# templePic = pictureBW.copy()
#
# skeletonPic = skeletization(templePic)
# cv2.imshow("test",skeletonPic)
# cv2.imshow("test2",pictureBW)
# cv2.waitKey()
# cv2.destroyAllWindows()
#


##############
#   Задание 2
##############



# Создание трэкбаров
def create_trackbar(win_name, max_value, H1, S1, V1, H2, S2, V2):
    cv2.createTrackbar("H1", win_name, H1, max_value, callback)
    cv2.createTrackbar("S1", win_name, S1, max_value, callback)
    cv2.createTrackbar("V1", win_name, V1, max_value, callback)
    cv2.createTrackbar("H2", win_name, H2, max_value, callback)
    cv2.createTrackbar("S2", win_name, S2, max_value, callback)
    cv2.createTrackbar("V2", win_name, V2, max_value, callback)


# Пороговая фильтрация
def threshold_filter(frame):
    min_color = np.array([H1, S1, V1], np.uint8)
    max_color = np.array([H2, S2, V2], np.uint8)
    frame = cv2.inRange(frame,
                       min_color,
                       max_color)
    return frame



def calibration_frame(calibration, calibration_hsv, win_name):
    global H1, H2, S1, S2, V1, V2

    H1 = cv2.getTrackbarPos("H1", win_name)
    S1 = cv2.getTrackbarPos("S1", win_name)
    V1 = cv2.getTrackbarPos("V1", win_name)
    H2 = cv2.getTrackbarPos("H2", win_name)
    S2 = cv2.getTrackbarPos("S2", win_name)
    V2 = cv2.getTrackbarPos("V2", win_name)
    mask = threshold_filter(calibration_hsv)
    calibration_with_mask = cv2.bitwise_and(calibration,
                                            calibration,
                                            mask=mask)
    return mask, calibration_with_mask

# Вызов для трэкбара
def callback(x):
    pass

# Срабатывает при нажатии на экран
# Меняет состояние - калибровано-некалибровано
def buttCallback(event, x, y, flags, userdata):
    if (event == cv2.EVENT_LBUTTONDOWN):
        global isCalibrate
        isCalibrate = 1



# Создаем окно
win_name = "Calibration"
cv2.namedWindow(win_name)
calibration = cv2.imread("img/2.jpg")
calibration_hsv = cv2.cvtColor(calibration,cv2.COLOR_BGR2HSV)

# Инициализация параметров для пороговой фильтрации
H1 = 91
S1 = 3
V1 = 107
H2 = 124
S2 = 145
V2 = 164


isCalibrate = 0

create_trackbar(win_name, 255, H1, S1, V1, H2, S2, V2)

# Калибровка по порогового фильтра
while (not isCalibrate):
    mask, c_with_mask = calibration_frame(calibration, calibration_hsv, win_name)
    cv2.setMouseCallback(win_name,
                         buttCallback,
                         isCalibrate)
    cv2.imshow(win_name, c_with_mask)
    cv2.waitKey(10)


# gray = cv2.cvtColor(calibration,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# skeleton = skeletization(mask)
# cv2.imwrite("skeleton.jpg", skeleton)

skeleton = cv2.imread("skeleton.jpg", 0)
lines = cv2.HoughLines(skeleton,1,np.pi/180,70)
# lines = cv2.HoughLines(edges,
for i in range(len(lines)):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    cv2.line(calibration,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('houghlines3.jpg',calibration)
cv2.waitKey()
cv2.destroyAllWindows()


