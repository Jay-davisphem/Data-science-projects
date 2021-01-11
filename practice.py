import numpy as np
import pandas as pd

def z(u, v, w, x, y):
    return (u*x - v*w)**y
u = np.arange(100, 600)
v = np.arange(0, 500)
w = np.arange(200, 700)
x = np.arange(300, 800)
y = np.remainder(np.arange(200, 700), 10) + 1
z = z(u, v, w, x, y)

df = pd.DataFrame({'u': u, 'v': v, 'w': w, 'x': x, 'y': y, 'z': z})
