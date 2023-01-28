import matplotlib.pyplot as plt
import os

def show_evolution(self): # dir: 'show_evolution/子文件夹名/'
      if not os.path.exists(self.dir):
                  os.mkdir(self.dir)
      for k in range(len(self.preturb)):
                  fig, ax = plt.subplots(1, 3)
                  fig.set_size_inches(9, 3)
                  ax[0].imshow(self.input_fig[0])# [h, w, 3]
                  ax[0].set_axis_off()
                  ax[1].imshow(0.5 + self.preturb[k])
                  ax[1].set_axis_off()
                  ax[2].imshow(self.noised_inputs[k])
                  ax[2].set_axis_off()
                  plt.savefig(self.dir + f'{k}.jpg')
                  plt.close()