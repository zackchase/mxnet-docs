# _MXNet_ Basics - NDArray

In _MXNet_, we primarily work with data via `NDArray` objects. Each `NDArray` consists of
a multidimensional array of homogenous data. Put simply, this just means that all elements 
of a given NDArray must be of the same type most often, we'll be working with floating point numbers.

As concrete examples, we could store a single _vector_ as a 1D `NDArray`, 
or a table of values, formally known as a _matrix_, as a 2D `NDArray`. 
In deep learning, we often have to work with higher-dimensional arrays. 
Images, for example consist of 3D `NDArrays`,
with separate dimensions corresponding to the height, width, and channel (red, green and blue). 
And when we have a dataset of multiple images, we might store this object in a 4D array.
When the number of dimensions exceeds 2, we call these objects _tensors_.

Of course, storing data isn't very exciting if you can't do anything interesting with it. 
The `NDArray` module supports a large number of optimized functions
for manipulating and computing upon their data. These include funamdental mathematical operations 
and also some specialized functions motivated by common use cases in neural networks. 

You might notice that _MXNet_'s `NDArray` is similar to `numpy.ndarray`.
However, _MXNet_ additionally provides the following important features:

* Multiple device support: `NDArray` runs operations on various devices, including CPU and GPU cards.
* Automatic parallelization: These operations executed in parallel with each other automatically.

In short, _MXNet_'s `NDArray` supports the familiar imperative programming interface of NumPy 
while providing the blistering speed required to execute state-of-the-art deep learning algorithms.  

## Creation and Initialization

We can create an `NDArray` on either a CPU or a GPU:

```python
    >>> import mxnet as mx
    >>> a = mx.nd.empty((2, 3)) # create a 2-by-3 matrix on cpu
    >>> b = mx.nd.empty((2, 3), mx.gpu()) # create a 2-by-3 matrix on gpu 0
    >>> c = mx.nd.empty((2, 3), mx.gpu(2)) # create a 2-by-3 matrix on gpu 2
    >>> c.shape # get shape
    (2L, 3L)
    >>> c.context # get device info
    gpu(2)
```

They can be initialized in various ways:

```python
    >>> a = mx.nd.zeros((2, 3)) # create a 2-by-3 matrix filled with 0
    >>> b = mx.nd.ones((2, 3))  # create a 2-by-3 matrix filled with 1
    >>> b[:] = 2 # set all elements of b to 2
```

We can copy the value from one NDArray to another, even if they are located on different devices:

```python
    >>> a = mx.nd.ones((2, 3))
    >>> b = mx.nd.zeros((2, 3), mx.gpu())
    >>> a.copyto(b) # copy data from cpu to gpu
```

We can also convert `NDArray` to `numpy.ndarray`:

```python
    >>> a = mx.nd.ones((2, 3))
    >>> b = a.asnumpy()
    >>> type(b)
    <type 'numpy.ndarray'>
    >>> print b
    [[ 1.  1.  1.]
    [ 1.  1.  1.]]
```

and vice versa:

```python
    >>> import numpy as np
    >>> a = mx.nd.empty((2, 3))
    >>> a[:] = np.random.uniform(-0.1, 0.1, a.shape)
    >>> print a.asnumpy()
    [[-0.06821112 -0.03704893  0.06688045]
     [ 0.09947646 -0.07700162  0.07681718]]
```

## Basic Element-wise Operations

By default, `NDArray` performs element-wise operations:

```python
    >>> a = mx.nd.ones((2, 3)) * 2
    >>> b = mx.nd.ones((2, 3)) * 4
    >>> print b.asnumpy()
    [[ 4.  4.  4.]
     [ 4.  4.  4.]]
    >>> c = a + b
    >>> print c.asnumpy()
    [[ 6.  6.  6.]
     [ 6.  6.  6.]]
    >>> d = a * b
    >>> print d.asnumpy()
    [[ 8.  8.  8.]
     [ 8.  8.  8.]]
```

If two `NDArray`s are located on different devices, we need to explicitly move them onto the same device. The following example performs computations on GPU 0:

```python
    >>> a = mx.nd.ones((2, 3)) * 2
    >>> b = mx.nd.ones((2, 3), mx.gpu()) * 3
    >>> c = a.copyto(mx.gpu()) * b
    >>> print c.asnumpy()
    [[ 6.  6.  6.]
     [ 6.  6.  6.]]
```

## Load and Save

There are two ways to save data to (or load it from) disks easily. The first way uses
`pickle`.  `NDArray` is pickle compatible, which means that you can simply pickle the
`NDArray` as you do with `numpy.ndarray`:

 ```python
    >>> import mxnet as mx
    >>> import pickle as pkl

    >>> a = mx.nd.ones((2, 3)) * 2
    >>> data = pkl.dumps(a)
    >>> b = pkl.loads(data)
    >>> print b.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
 ```

The second way is to directly dump a list of `NDArray` to disk in binary format:

 ```python
    >>> a = mx.nd.ones((2,3))*2
    >>> b = mx.nd.ones((2,3))*3
    >>> mx.nd.save('mydata.bin', [a, b])
    >>> c = mx.nd.load('mydata.bin')
    >>> print c[0].asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> print c[1].asnumpy()
    [[ 3.  3.  3.]
     [ 3.  3.  3.]]
 ```

We can also dump a dict:

 ```python
    >>> mx.nd.save('mydata.bin', {'a':a, 'b':b})
    >>> c = mx.nd.load('mydata.bin')
    >>> print c['a'].asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> print c['b'].asnumpy()
    [[ 3.  3.  3.]
     [ 3.  3.  3.]]
 ```

In addition, if we have set up distributed file systems, such as Amazon S3 and HDFS, we
can directly save to and load from them. For example:

 ```python
    >>> mx.nd.save('s3://mybucket/mydata.bin', [a,b])
    >>> mx.nd.save('hdfs///users/myname/mydata.bin', [a,b])
 ```

## Automatic Parallelization
`NDArray` can automatically execute operations in parallel. This is desirable when you
use multiple resources, such as CPU and GPU cards, and CPU-to-GPU memory bandwidth.

For example, if we write `a += 1` followed by `b += 1`, and `a` is on a CPU card while
`b` is on a GPU card, then we will want to execute them in parallel to improve the
efficiency. Furthermore, data copies between CPU and GPU are expensive, so we
want to run them in parallel with other computations.

However, finding statements that can be executed in parallel by eye is hard. In the
following example, `a+=1` and `c*=3` can be executed in parallel, but `a+=1` and
`b*=3` have to be sequentially executed.

 ```python
    a = mx.nd.ones((2,3))
    b = a
    c = a.copyto(mx.cpu())
    a += 1
    b *= 3
    c *= 3
 ```

This where _MXNet_'s power lies: it automatically resolves dependencies and
execute operationss in parallel while guaranteeding correctness. 
In other words, you can write a program without explicitly addressing threading.
_MXNet_ will take the program, intelligently analyze the computation, 
and strategically dispatch tasks to multiple devices (GPUs, multiple machines, etc).

_MXNet_ achieves this by lazy evaluation. Any operation we write down is issued to a
internal engine, and then returned. For example, if we run `a += 1`, it
returns immediately after pushing the plus operation to the engine. This
asynchronism allows us to push more operations to the engine, so it can determine
the read and write dependency and find the best way to execute operations in
parallel.

The actual computations are finished when we copy the results someplace else, such as `print a.asnumpy()` or `mx.nd.save([a])`. Therefore, to write highly parallelized code, we only need to postpone asking for
the results.

##  Next Steps
* [Symbol](symbol.md)
* [KVStore](kvstore.md)
