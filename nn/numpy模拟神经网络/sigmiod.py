import numpy as np
import matplotlib.pyplot as plt
class Sigmiod():
    """
    def __init__(self):
        pass
    """
    def sigmiod(self):
        y = 1/(1+np.exp(-self))
        return y

    def plot_sigmoid(self,low,up,pace):
        x = np.arange(low,up,pace)
        y = Sigmiod.sigmiod(x)
        plt.plot(x,y)
        plt.show()

if __name__ == '__main__':
    Sigmiod().plot_sigmoid(-5,5,0.1)