API Reference
=============

This part of the documentation covers all the interfaces of GraphKit. 


Operation Interface
-------------------

.. autoclass:: flickr.vision.graphkit.Operation
   :members: __init__, compute, _after_init, provides, needs, params, name

Network Object
--------------

.. autoclass:: flickr.vision.graphkit.network.Network
   :members: __init__, compute, add_op, compile
