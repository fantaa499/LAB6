import cv2
import numpy as np
from matplotlib import pyplot as plt

ni = cv2.imread("img/ni.png")
cu = cv2.imread("img/cu.png")
ni_hsv = cv2.cvtColor(ni, cv2.COLOR_BGR2HSV)
cu_hsv = cv2.cvtColor(cu, cv2.COLOR_BGR2HSV)

cv2.namedWindow("test")

img_origin = cv2.imread("img/3.jpg")
img_hsv = cv2.cvtColor(img_origin, cv2.COLOR_BGR2HSV)
img = cv2.cvtColor(img_origin,cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)
# ret, img = cv2.threshold(img,0,200,cv2.THRESH_BINARY)
# img = cv2.bitwise_not(img,
#                       img,
#                       mask=img)
cv2.imshow("test",img)
cv2.waitKey()
cv2.destroyAllWindows()
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                           param1=49,param2=53,minRadius=0,maxRadius=0)

blackC = (0, 0, 0)

circles = np.uint16(np.around(circles))
squares_hsv = []
squares = []
for circl in circles[0,:]:
    square = np.zeros((circl[2] * 2, circl[2] * 2, 3), np.uint8)
    square_hsv = np.zeros((circl[2] * 2, circl[2] * 2, 3), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Для круга
            # if (i-circl[1])**2 + (j-circl[0])**2 < circl[2]**2:
            #     img_origin[i][j] = (255,0,0)
            # Для квадрата
            if (circl[1] - circl[2] < i < circl[1] + circl[2]) and \
               (circl[0] - circl[2] < j < circl[0] + circl[2]):
                # Для круга
                if (i-circl[1])**2 + (j-circl[0])**2 < circl[2]**2:
                    square[i - circl[1] + circl[2] - 1][j - circl[0] + circl[2] - 1] = img_origin[i][j]
                    square_hsv[i - circl[1] + circl[2] - 1][j - circl[0] + circl[2] - 1] = img_hsv[i][j]
                else:
                    square[i - circl[1] + circl[2] - 1][j - circl[0] + circl[2] - 1] = blackC
                    square_hsv[i - circl[1] + circl[2] - 1][j - circl[0] + circl[2] - 1] = blackC
    # draw the outer circle
    squares.append(square)
    squares_hsv.append(square_hsv)

    cv2.circle(img_origin,(circl[0],circl[1]),circl[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img_origin,(circl[0],circl[1]),2,(0,0,255),3)
squares = np.array(squares)
squares_hsv = np.array(squares_hsv)

cv2.imshow("test",img_origin)
cv2.waitKey()
cv2.destroyAllWindows()
def makeBorder (el, square):
    if el.shape[0] > square.shape[0]:
        square = cv2.copyMakeBorder(square,
                                    0,
                                    (el.shape[0] - square.shape[0]),
                                    0,
                                    (el.shape[1] - square.shape[1]),
                                    cv2.BORDER_CONSTANT)
    else:
        el = cv2.copyMakeBorder(el,
                                0,
                                (square.shape[0] - el.shape[0]),
                                0,
                                (square.shape[1] - el.shape[1]),
                                cv2.BORDER_CONSTANT)
    return el, square

def normalize (a, b):
    a, b = makeBorder(a, b)
    ba, ga, ra = cv2.split(a)
    bb, gb, rb = cv2.split(b)
    ba_norm = ba / np.sqrt(np.sum(ba ** 2))
    ga_norm = ga / np.sqrt(np.sum(ga ** 2))
    ra_norm = ra / np.sqrt(np.sum(ra ** 2))
    bb_norm = bb / np.sqrt(np.sum(bb ** 2))
    gb_norm = gb / np.sqrt(np.sum(gb ** 2))
    rb_norm = rb / np.sqrt(np.sum(rb ** 2))
    return ba_norm, ga_norm, ra_norm, bb_norm, gb_norm, rb_norm



def normaL2 (a, b):
    global blackC
    a, b = makeBorder(a, b)
    ba, ga, ra = cv2.split(a)
    bb, gb, rb = cv2.split(b)
    aBGR = np.array([ba,ga,ra])
    bBGR = np.array([bb, gb, rb])
    normb = 0
    normg = 0
    normr = 0
    counter = 1000
    for i in range(a.shape[1]):
        for j in range(a.shape[0]):
            if (a[i][j] != 0).any() and (b[i][j] != 0).any() and counter > 0:
                normb = normb + (aBGR[0][i][j] - bBGR[0][i][j]) ** 2
                normg = normg + (aBGR[1][i][j] - bBGR[1][i][j]) ** 2
                normr = normr + (aBGR[2][i][j] - bBGR[2][i][j]) ** 2
                counter -= 1
    normb = np.sqrt(normb)
    normg = np.sqrt(normg)
    normr = np.sqrt(normr)
    area = a.shape[0]**2
    return normb/area , normg/area, normr/area

# for i, square in enumerate(squares):
#     ba_norm, ga_norm, ra_norm, bb_norm, gb_norm, rb_norm = normalize(ni, square)
#     print("степень схожести с никелем монеты ", i, " равна - ", np.sum(ba_norm * bb_norm) * 100, "%", " по B")
#     print("степень схожести с никелем монеты ", i, " равна - ", np.sum(ga_norm * gb_norm) * 100, "%", " по G")
#     print("степень схожести с никелем монеты ", i, " равна - ", np.sum(ra_norm * rb_norm) * 100, "%", " по R")
#     ba_norm, ga_norm, ra_norm, bb_norm, gb_norm, rb_norm = normalize(cu, square)
#     print("степень схожести с медью монеты ", i, " равна - ", np.sum(ba_norm * bb_norm) * 100, "%", " по B")
#     print("степень схожести с медью монеты ", i, " равна - ", np.sum(ga_norm * gb_norm) * 100, "%", " по G")
#     print("степень схожести с медью монеты ", i, " равна - ", np.sum(ra_norm * rb_norm) * 100, "%", " по R")

def dftCenter (image):
    # нахождение оптимальных значений для ДПФ
    dft_width = cv2.getOptimalDFTSize(image.shape[1])
    dft_height = cv2.getOptimalDFTSize(image.shape[0])
    # заполнение недостающих пикселей черным цветом (оптимальное изображение может быть больше)
    image = cv2.copyMakeBorder(image,
                              0,
                              dft_width - image.shape[1],
                              0,
                              dft_height - image.shape[0],
                              cv2.BORDER_CONSTANT)
    # dft - двухканальное изображение, ДПФ
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # переносим низкие частоты в центр
    dft_shift = np.fft.fftshift(dft)
    # находим длины векторов и переводим в логарфимический масштаб
    magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

def antiDFT (img):
    img = cv2.dft(np.float32(img), flags=(cv2.DFT_INVERSE | cv2.DFT_REAL_OUTPUT))
    return img


# let = cv2.imread("pic/let2.jpg", 0)
# num = cv2.imread("pic/num1.jpg", 0)
# let = cv2.bitwise_not(let)
# num = cv2.bitwise_not(num)
#
# new_img_mul = cv2.mulSpectrums(num_dft_origin, let_dft_origin, 0, conjB=1)
# anti_new_img = cv2.dft((new_img_mul), flags=(cv2.DFT_INVERSE | cv2.DFT_REAL_OUTPUT))
#
# match_img = anti_new_img.copy()
# # находим максимальное значение элемента
# min, max, coord_min, coord_max = cv2.minMaxLoc(match_img)
# newmax = max-59504415232
# # newmax = 12075304415232
# match_img = filtr(match_img, newmax)


# for square in squares:
#     b, g, r = normaL2(square, ni)
#     print(b, "  ", g, "  ", r)
#     print((b + g + r) / 3, "\n")
#     b, g, r = normaL2(square, cu)
#     print(b, "  ", g, "  ", r)
#     print((b+g+r) /3, "\n")
#     print("#########################################")


    # errorL2a = cv2.norm(square-ni, cv2.NORM_L2)


    # sim = errorL2a / square.shape[0]**2
    # print(sim, "\n")
    # square, cu = makeBorder(square, cu)
    # errorL2b = cv2.norm(square-cu,cv2.NORM_L2)
    # sim = errorL2b / square.shape[0]**2
    # print(sim, "\n")
#
# cv2.imshow('detected circles',squares[0])
# cv2.waitKey(0)
# cv2.imshow('detected circles',cu)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
#


# plt.imshow(squares[5])
# plt.title("name")
# plt.xticks([])
# plt.yticks([])
# plt.show()
# cv2.imshow("test", squares_hsv[2])
# cv2.waitKey()

#
# for i, square in enumerate(squares_hsv):
#     template, square = makeBorder(cu_hsv, square)
#     hist = cv2.calcHist(template, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hist1 = cv2.normalize(hist, hist).flatten()
#     hist = cv2.calcHist(square, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hist2 = cv2.normalize(hist, hist).flatten()
#     d1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
#
#     template, square = makeBorder(ni_hsv, square)
#     hist = cv2.calcHist(template, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hist1 = cv2.normalize(hist, hist).flatten()
#     hist = cv2.calcHist(square, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hist2 = cv2.normalize(hist, hist).flatten()
#     d2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
#     if (d1 > d2):
#         string = "латунное гальванопокрытие"
#         string = "latun"
#     else:
#         string = "никелевое гальванопокрытие"
#         string = "nikel"
#     print(d1)
#     print(d2)
#     print(string)
#     print("#########################################")
#     cv2.imshow(string, squares_hsv[i])
#     cv2.waitKey()





# cv2.imshow("test", img_hsv)
# cv2.waitKey()
# cv2.imshow("test", ni_hsv)
# cv2.waitKey()
# cv2.imshow("test", square)
# cv2.waitKey()

