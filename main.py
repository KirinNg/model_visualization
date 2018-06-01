import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import lucid.modelzoo.vision_models as model
from lucid.misc.io import show
from lucid.misc.io import save
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

model = model.InceptionV1()
model.load_graphdef()

_ = render.render_vis(model,"mixed4a_pre_relu:492")
img = np.reshape(_,[128,128,3])
plt.imshow(img)
plt.show()