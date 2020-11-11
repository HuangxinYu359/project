from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#加载图片
image = Image.open('str.jpg')
A = np.array(image)
#A.shape (417,500,3)
#灰度化：转成二维数据
image_gray = A.max(axis =2)
#查看图片，image_gray.shape为（417，500）
print(image_gray.shape)
plt.imshow(image_gray)
plt.show()
#对矩阵进行分解，得到p,s,q
p,s,q = svd(image_gray,full_matrices=False)

def get_image_feature(s,k):
    s_temp = np.zeros(s.shape[0])
    s_temp[0:k] = s[0:k]
    s = s_temp*np.identity(s.shape[0])
    #重构A
    temp_a = np.dot(p,s)
    temp_a = np.dot(temp_a,q)
    plt.imshow(temp_a)
    plt.show()



get_image_feature(s,4)
get_image_feature(s,40)
get_image_feature(s,200)
