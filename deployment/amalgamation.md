# Deploying MXNet

If you are reading this page, you are likely familiar with the basics of 
_deep learning_ on MXNet and may have already trained a few models using the framework. 
You are now probably wondering how to put these models into production by deploying 
them to run inference in your existing applications that may live on servers, mobile phones, 
PCs, browsers or even IoT devices. Because of the lightweight size of MXNet models and the 
engine's cross platform linear algebra acceleration it is possible to deploy trained networks 
across all of these device classes using MXNet's inference-optimized engine called Amalgmation.

## What is Amalgamation?

Amalgmation is a build of MXNet that contains only the predict (forward) API along with the corresponding
hardware backend needed to accerelate inference computations on the device it is being built for. By only 
including the parts of the code needed for inference of pre-trained models, Amalgmation significantly
reduces the computational overhead of the engine, making it optimial for memory and compute constrained 
devices like phones or IoT deployments. Amalgamation can also help reduce the overhead of the MXNet runtime 
on server deployments, limiting the compute needed for high traffic models.

## Platforms Supported:

MXNet currenly supports Amalgmated build on the following platforms:

* __iOS__: [Build Instructions For iOS Devices](https://github.com/dmlc/mxnet/tree/master/amalgamation#ios)
* __Android__: [Build Instructions For Android Devices](https://github.com/dmlc/mxnet/tree/master/amalgamation#android)
* __Raspberry Pi__: [Build Instructions For Raspbian](http://mxnet.io/get_started/platforms/raspbian_setup.html)
* __All Modern Browsers__: [Build Instructions For Javascript](https://github.com/dmlc/mxnet/tree/master/amalgamation#javascript)
* __x86 Linux__: [Build Instructions For x86 Linux Variants](https://github.com/dmlc/mxnet/tree/master/amalgamation#how-to-generate-the-amalgamation)
* __ARM Linux__: [Build Instructions For ARM Linux Variants](https://github.com/dmlc/mxnet/tree/master/amalgamation#how-to-generate-the-amalgamation)

## Why Amalgamation?

You may be asking why there is a whole seperate build process needed to support Amalgmation. Often in a
production setting, the engine running model inference will have to be heavily modified to integrate tightly
with the rest of the application code, and this is rather hard to do with a large amount of code superfluous to
inference hanging around. The distinct and highly modular build process offered by Amalgamation is a way to ensure
that you have full control over what goes into the binary you are deploying. This also ensures that you can easily
edit the pieces of the engine that may need to be tweaked for your specific deployment.

## Customizing Amalgmation

One of the strongest selling points of Amalgmation and MXNet more generally is the ease with which it can be modified and extended.
Because Amlagmation direcly pulls in the hardware backend's for execution from [MShadow](https://github.com/dmlc/mshadow/), 
the compute graph execution optimization engine [NNVM](https://github.com/dmlc/nnvm/) and all the operators from 
[MXNet](https://github.com/dmlc/mxnet/tree/master/src/operator), augmenting the Amalgmated engine's functionality is as simple as editing
those parts of the project and re-building Amalagmation.

You may want to do this if you are adding new operators to MXNet or introducing custom hardware backends to support accelerated inference
on specific hardware, like mobile phone GPUs. Additionally, you may want to edit these file to remove operators and backends that are 
not used in the networks you plan on deploying, slimming down the size of the Amalgamted library and runtime even further.

## Conclusions
Given its combination of modifiability as well as computational and memory efficiency, MXNet's Amalgmation build is a unique and highly versatile
deployment option for your trained models on almost any platform imaginable.