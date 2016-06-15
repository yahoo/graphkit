.. graphkit documentation master file, created by
   sphinx-quickstart on Tue Jun 16 19:10:27 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Foreword
========

Read this document first to get a sense of the design goals and motivations for Graphkit


Motivation
----------



Design goals
------------

**Portability**

    This library should function seamlessly in Python 2 and Python 3 with minimal setup across linux, mac, and windows.

**Efficient, small, stable, well-documented core**

    When adding new functionality, we will take careful consideration for whether we can achieve the same functionality using the existing APIs. If this is the case, we will prefer writing documentation with concrete examples over adding new functionality.

    Once we reach 1.0, we will gaurantee backwards compatibility for 100% of the documented APIs provided by this library.

    100% of the public apis provided by this library will be documented and showcased with practical code examples.

**Adaptability**

    The value of this framework is in its ability to unify the interface to all predictive models. To this end, this framework should strive to provide concrete examples for adapting caffe, theano, scikit-learn, and many other machine learning frameworks.

**Serialization**

    Users of this library should not have to worry about what format to serialize their models as. If they adhere to the convention provided by this library, the library will provided the necessary tools to serialize the model parameters.

**Open source**

    We will aim to eventually open source this repository

Non goals
---------
    
**Model training**

    A major use case for this library is in the representation of predictive models as directed acyclic graphs. Naturally, training a model is one of the core functions that one often performs with predictive models. However, training often brings with it many dependencies that may or may not be needed during inference time. 
      
    We plan to provide in a separate library the necessary tools to access model parameters for required training. 
      
    We think of training as an computation performed **on** a network and its parameters as opposed one performed **by** the network. With this in mind, we believe that separate tools can be developed to address the training requirement without introducing additional complexity & depedencies into this framework.

**Operation zoo**

    Contributions for custom operations should not live in this repository. Instead, we ask that custom operations be contributed into vision-features so that we can keep this repository minimal and portable across a wide variety of platforms.
