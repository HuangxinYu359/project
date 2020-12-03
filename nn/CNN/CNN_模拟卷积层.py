import torch
import numpy as np

image = np.array([[1,1,1,0,0],
         [0,1,1,1,0],
         [0,0,1,1,1],
         [0,0,1,1,0],
         [0,1,1,0,0]])
filter_1 = np.array([[1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1]])
filters=np.array([filter_1])
image = image.astype('float32')
#from_numpy 将numpy数据放到tensor
#batch,chanel,width,height
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(1)
#print(image.shape)

weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
#使用卷积对四维数据进行处理
import torch.nn as nn
conv = nn.Conv2d(1,1,kernel_size=(3,3),bias = False)
conv.weight = torch.nn.Parameter(weight)
conv_output = conv(image)
print(conv_output)

