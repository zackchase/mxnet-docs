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

Luckily, MXNet can automatically resolve the dependencies and
execute operations in parallel with correctness guaranteed. In other words, we
can write a program as if it is using only a single thread, and MXNet will
automatically dispatch it to multiple devices, such as multiple GPU cards or multiple
computers.

MXNet achieves this by lazy evaluation. Any operation we write down is issued to a
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
