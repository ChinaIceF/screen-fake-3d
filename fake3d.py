from lib.getfileinfo import *
from lib.get_image_grade import *

import os
import sys
import matplotlib.image as mpimg
import math
import numpy

#  常量
_d = 0.9    #  使用者距离屏幕的距离，单位 m
_screen_yx = [2160, 3840]  #  屏幕的分辨率
_screen_dpm = 3840 / 1.23  #  像素个数每米

_eye_pos = [int(_screen_yx[0]/2), int(_screen_yx[1]/2)]
_a = numpy.arctan((1.23 / (2 * _d)))

def direction_to_pixel(d_):
    return int(d_ * _screen_dpm)

def get_angle(p):
    return numpy.arctan(p / _d)

if len(sys.argv) > 1 :
    filename = sys.argv[1]
else :
    filename = "test.png"

#  读取文件
image_loaded = mpimg.imread(filename)
image_y, image_x, image_depth = image_loaded.shape

# Fcolor(xf) = color(xp * arctan(d/xf) / a)
final_image = numpy.zeros(image_loaded.shape)
final_image_y, final_image_x, final_image_depth = final_image.shape

for yf in range(- int(_screen_yx[0]/2), int(_screen_yx[0]/2), 1):
    for xf in range(- int(_screen_yx[1]/2), int(_screen_yx[1]/2), 1):
        index_xf = int((xf + _screen_yx[1]/2) * final_image_x / image_x)
        index_yf = int((yf + _screen_yx[0]/2) * final_image_y / image_y)
        
        modified_x = int(image_x / 2 * (get_angle(xf / _screen_dpm)) / _a + image_x / 2)
        modified_y = int(image_y / 2 * (get_angle(yf / _screen_dpm)) / _a + image_y / 2)

        final_image[index_yf, index_xf, :] = image_loaded[modified_y, modified_x, :]
        print(index_yf / _screen_yx[0])

#  输出
generate_image(final_image, "output.png")