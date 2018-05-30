import numpy as np

import matplotlib as mpl
mpl.use('TkAgg') # use TkAgg backend to avoid segmentation fault
import matplotlib.pyplot as plt

xx, yy = np.mgrid[:75, :75]
c = [20, 20]
r = 3.1


circle = (xx-c[0])**2 + (yy-c[1])**2
circle = circle<r**2
print(np.nonzero(circle))

plt.imshow(circle)
plt.show()