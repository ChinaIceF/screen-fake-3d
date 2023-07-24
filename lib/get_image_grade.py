
import numpy
import numba
import sys
import matplotlib.pyplot as plt
from numba import *
from PIL import Image


'''
class Kernel(object):
  ##  Kernel 卷积核类
  def __init__(self, arg):
    self.arg = numpy.array(arg)
    self.y, self.x = self.arg.shape
    self.total_size = self.y * self.x
'''

def generate_image(image, name, k=255):
  ##  把图片输出成文件
  pil_image_calculated = Image.fromarray(numpy.uint8(image * k))
  pil_image_calculated.save(name,"png")

@jit(nopython = True, parallel = True)
def CNN(a_channel ,kernel):

  ##  逐个图层进行卷积，输入的 a_channel 是某个通道（二维数组，取值 0 - 1 ）
  ##  kernel 是一个二维数组
  
  channel_y, channel_x = a_channel.shape
  limit_y, limit_x = [channel_y - int(kernel.shape[0] / 2) - 1, channel_x -  int(kernel.shape[1] / 2) - 1]
  
  result = numpy.ones((limit_y, limit_x))
  
  for y in prange(1, limit_y, 1):
    #print("\r","__________"[0:int(y/limit_y*10)],end = "")
    for x in range(1, limit_x, 1):
      temp_piece = a_channel[y-1:y+2, x-1:x+2]
      result[y,x] = numpy.mean(temp_piece * kernel)

  return result



if __name__ == "__main__":
  
  _ALL_CHANNEL_NAME = ['R','G','B','A']
  
  if len(sys.argv) > 1 :
    filename = sys.argv[1]
  else :
    filename = "test.png"

  get_img_info(filename)
  
  image_loaded = mpimg.imread(filename)
  image_y, image_x, image_depth = image_loaded.shape
  
  kernel_test = numpy.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
  
  
  #  卷积核控制向量方向 ↓ → ↙ ↘
  kernel_directly_y_plus = numpy.array([[-1,-2,-1],[0,0,0],[1,2,1]])
  kernel_directly_x_plus = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  kernel_left_down = numpy.array([[0,-1,-2],[1,0,-1],[2,1,0]])
  kernel_right_down = numpy.array([[-2,-1,0],[-1,0,1],[0,1,2]])
  
  hr_2 = 0.7071067811865476
  all_kernels = [kernel_directly_y_plus, kernel_directly_x_plus, kernel_left_down, kernel_right_down]
  all_basis_vector = [(0,1),(1,0),(-hr_2,hr_2),(hr_2,hr_2)]   
  """  这些向量是(x, y)"""
  
  image_grade_vector = numpy.zeros([image_y,image_x,3,image_depth]) #  定义向量 array
  
  for c in range(image_depth):
    this_output_image = numpy.zeros([image_y,image_x,3]) #  定义 RGB 通道
    
    #  R B 通道分别为沿 y x 方向的向量
    for i in range(4):
      CNNed = CNN(image_loaded[:,:,c], all_kernels[i])
      this_output_image[:-2,:-2,1] = this_output_image[:-2,:-2,1] + CNNed * all_basis_vector[i][1]
      this_output_image[:-2,:-2,2] = this_output_image[:-2,:-2,2] + CNNed * all_basis_vector[i][0]
    
    image_grade_vector[:,:,:,c] = numpy.array(this_output_image)
    
    ##  输出为图片的部分，只是为了可视化，做的处理不进行运算
    this_output_image = this_output_image * 0.125  #  防止溢出，乘一个比例，大小随便，只要最大值小于1就行
    this_output_image = this_output_image / 0.5 + 0.5
    this_output_image[:,:,0] = image_loaded[:,:,c] * 1  #  R  通道是原图的值
    generate_image(this_output_image, "image_output/image_grade_texture_"+_ALL_CHANNEL_NAME[c]+".png", 255)
  
  
  plt.axis([0, image_y, image_x, 0])
  plt.axis('on')
  #plt.grid(True)
  '''
  plt.arrow(0,0, 20,0, head_length = 4, head_width = 3, color = 'k')
  plt.arrow(0,0, 0,20, head_length = 4, head_width = 3, color = 'k')
  '''
  ax = plt.gca()
  ax.set_aspect(1)
  
  ##  按 R 的向量图生成坐标
  _ALL_ARROW_COLOR = ['#FF000030','#00FF0030','#0000FF30','#7F7F7F00']
  resize_level = 2
  for c in range(image_depth):
    _y = image_y - 2
    _x = image_x - 2
    for y in range(1,_y):
      print("\r",y/_y,end = "")
      for x in range(1,_x):
        vector_y = image_grade_vector[y,x][1][c] * resize_level
        vector_x = image_grade_vector[y,x][2][c] * resize_level
        if not (vector_y == 0 and vector_x == 0):
          plt.arrow(x,y,vector_x,vector_y, head_length = 0.1, head_width = 0.1, color = _ALL_ARROW_COLOR[c])
  
  plt.axis('off')
  plt.savefig("image_output/color_grade_vector.png", dpi=1200, bbox_inches='tight')
  #plt.show()