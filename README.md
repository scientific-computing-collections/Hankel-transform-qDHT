# Quais-discrete Hankel Transform  

python3 algorithm that perform zero-order Hankel Transform for an 1D array (float or real valued). 
Such QDHT algorithm is especially suit for iterative codes, where energy-preserving is important -- as discrete form of Parseval theorem is  guaranteed.

For more details, see: 
[Li Yu, et.al, Quais-discrete Hankel transform, Optics Letters, 23, 409, 1998](https://www.osapublishing.org/ol/abstract.cfm?&uri=ol-23-6-409). 


## description 

----
## usage
```python
import numpy as np 
import matplotlib.pyplot as plt 

N = 4096 # points 
ht = Hankel_qDHT(N) 
r_axis = ht.r   # the generated ratial axis, r is almost equally spaced, 
                # corresponding to the positive-zeros of 0-order Bessel 
                # function J_0(x) 
# field before 0-order Hankel Transform: 
field =  np.exp(-np.pi * r_axis**2 / (1.0 + 1.0J * 0.2)) 
# field after the transform, lives still on ht.r axis: 
field_H0 = ht.transform(field) # 

plt.plot(r_axis, np.abs(field), "b-") 
plt.plot(r_axis, np.abs(field_H0), "r--") 
plt.xlabel("r") 
plt.ylabel("abs. field") 
plt.show() 
```

### example
qDHT close to the precise of machine error: 
![compare with the direct-sum result](https://github.com/scientific-computing-collections/Hankel-transform-qDHT/blob/main/qDHT-compare-with-direct-sum.png)

### requiremets 

`python>=3.0` 
site-packages: `numpy`, `scipy`, `matplotlib` 
