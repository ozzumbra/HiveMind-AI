
---

## **3. Файл `hex_utils.py`**
Вспомогательные функции для работы с гексагональными координатами:
```python
import numpy as np

def axial_to_cube(q, r):
    x = q
    z = r
    y = -x - z
    return np.array([x, y, z])

def cube_distance(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]), abs(a[2]-b[2]))

def generate_hex_kernel(radius=1):
    kernel = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if abs(s) <= radius:
                kernel.append((q, r))
    return kernel

