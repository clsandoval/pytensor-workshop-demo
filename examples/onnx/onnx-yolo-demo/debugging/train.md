ify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.        

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/train.py", line 548, in main   
    trainer.train()
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/train.py", line 463, in train  
    avg_loss, avg_box_loss, avg_cls_loss = self.train_epoch(dataloader)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/train.py", line 306, in train_epoch
    loss, box_loss, cls_loss = self.train_fn(images)
                               ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/compile/function/types.py", line 1038, in __call__ 
    outputs = vm() if output_subset is None else vm(output_subset=output_subset)
              ^^^^
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/basic.py", line 669, in thunk
    raise_with_op(self.fgraph, output_nodes[0], thunk)
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/utils.py", line 526, in raise_with_op
    raise exc_value.with_traceback(exc_trace)       
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/basic.py", line 665, in thunk
    outputs = fgraph_jit(*(x[0] for x in thunk_inputs))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/tmpin84sjdj", line 1123, in jax_funcified_fgraph
    tensor_variable_556 = alloc_1(tensor_variable_555, tensor_variable_400, tensor_variable_412, tensor_variable_410, tensor_variable_406, tensor_constant_6)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/tensor_basic.py", line 46, in alloc
    res = jnp.broadcast_to(x, shape)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 3070, in broadcast_to
    return util._broadcast_to(array, shape, sharding=out_sharding)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/venv/lib/python3.11/site-packages/jax/_src/numpy/util.py", line 278, in _broadcast_to
    shape = core.canonicalize_shape(shape)  # type: ignore[arg-type]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Shapes must be 1D sequences of concrete values of integer type, got (16, JitTracer<~int32[]>, JitTracer<~int32[]>, JitTracer<~int32[]>, 2).      
If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
Apply node that caused the error: Sub(True_div.0, mean)
Toposort index: 2983
Inputs types: [TensorType(float32, shape=()), TensorType(float32, shape=())]
Inputs shapes: [(16, 3, 320, 320), (6,), (128,), (64,), (64,), (64,), (32,), (128,), (64,), (256,), (128,), (256,), (128,), (256,), (128,), (256,), (128,), (64,), (128,), (64,), (32,), (64,), (32,), (16,), (32,), (16,), (16, 3, 3, 3), (16,), (16,), (16,), (32, 16, 3, 3), (32,), (32,), (32,), (16, 32, 1, 1), (16,), (16,), (16,), (16,), (16,), (16, 16, 3, 3), (16,), (16,), (16,), (16, 16, 3, 3), (16,), (16,), (16,), (32, 32, 1, 1), (32,), (32,), (32,), (64, 32, 3, 3), (64,), (64,), (64,), (32, 64, 1, 1), (32,), (32,), (32,), (32,), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (64, 64, 1, 1), (64,), (64,), (64,), (128, 64, 3, 3), (128,), (128,), (128,), (64, 128, 1, 1), (64,), (64,), (64,), (64,), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (128,), (256, 128, 3, 3), (256,), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (128,), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128,), (256, 256, 1, 1), (256,), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (128,), (256, 512, 1, 1), (256,), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128,), (256, 256, 1, 1), (256,), (256,), (256,), (64, 384, 1, 1), (64,), (64,), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (128,), (32, 192, 1, 1), (32,), (32,), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (64, 64, 1, 1), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 192, 1, 1), (64,), (64,), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (128,), (6, 128, 1, 1), (6,), (6,), (6,), (16, 3, 3, 3), (16,), (16,), (32, 16, 3, 3), (32,), (32,), (16, 32, 1, 1), (16,), (16,), (32, 32, 1, 1), (32,), (32,), (16, 16, 3, 3), (16,), (16,), (16, 16, 3, 3), (16,), (16,), (64, 32, 3, 3), (64,), (64,), (32, 64, 1, 1), (32,), (32,), (64, 64, 1, 1), (64,), (64,), (32, 32, 3, 3), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (128, 64, 3, 3), (128,), (128,), (64, 128, 1, 1), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (256, 128, 3, 3), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (256, 256, 1, 1), (256,), (256,), (128, 128, 3, 3), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128, 256, 1, 1), (128,), (128,), (256, 512, 1, 1), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (256, 256, 1, 1), (256,), (256,), (64, 384, 1, 1), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (32, 192, 1, 1), (32,), (32,), (64, 64, 1, 1), (64,), (64,), (32, 32, 3, 3), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (64, 64, 3, 3), (64,), (64,), (64, 192, 1, 1), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (128, 128, 3, 3), (128, 128, 3, 3), (128,), (128,), (128,), (128,), (128, 384, 1, 1), (128, 384, 1, 1), (128,), (128,), (128,), (128,), (256, 256, 1, 1), (256, 256, 1, 1), (256,), (256,), (256,), (256,), (128, 128, 3, 3), (128, 128, 3, 3), (128,), (128,), (128,), (128,), (128, 128, 3, 3), (128, 128, 3, 3), (128,), (128,), (128,), (128,), (6, 64, 1, 1), (6, 64, 1, 1), (6,), (6,), (6,), (6,), (6, 128, 1, 1), (6,), (6,), (6, 256, 1, 1), (6, 256, 1, 1), (6,), (6,), (6,), (6,)]
Inputs strides: [(1228800, 4, 3840, 12), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (108, 36, 12, 4), (4,), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (4,), (128, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (4,), (128, 4, 4, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (2048, 4, 4, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (1536, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (768, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (768, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (108, 36, 12, 4), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (128, 4, 4, 4), (4,), (4,), (128, 4, 4, 4), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (2048, 4, 4, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (1536, 4, 4, 4), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (768, 4, 4, 4), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (768, 4, 4, 4), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4608, 36, 12, 4), (4608, 36, 12, 4), (4,), (4,), (4,), (4,), (1536, 4, 4, 4), (1536, 4, 4, 4), (4,), (4,), (4,), (4,), (1024, 4, 4, 4), (1024, 4, 4, 4), (4,), (4,), (4,), (4,), (4608, 36, 12, 4), (4608, 36, 12, 4), (4,), (4,), (4,), (4,), (4608, 36, 12, 4), (4608, 36, 12, 4), (4,), (4,), (4,), (4,), (256, 4, 4, 4), (256, 4, 4, 4), (4,), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (1024, 4, 4, 4), (1024, 4, 4, 4), (4,), (4,), (4,), (4,)]
Inputs values: ['not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown']
Outputs clients: [[output[0](Sub.0)]]

HINT: Re-running with most PyTensor optimizations disabled could provide a back-trace showing when this node was created. This can be done by setting the PyTensor flag 'optimizer=fast_compile'. If that does not work, PyTensor optimizations can be disabled with 'optimizer=None'.
HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.

Saving emergency checkpoint...
✓ Checkpoint saved: checkpoints/error_checkpoint.npz
Traceback (most recent call last):
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/basic.py", line 665, in thunk
    outputs = fgraph_jit(*(x[0] for x in thunk_inputs))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/tmpin84sjdj", line 1123, in jax_funcified_fgraph
    tensor_variable_556 = alloc_1(tensor_variable_555, tensor_variable_400, tensor_variable_412, tensor_variable_410, tensor_variable_406, tensor_constant_6)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/tensor_basic.py", line 46, in alloc
    res = jnp.broadcast_to(x, shape)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 3070, in broadcast_to
    return util._broadcast_to(array, shape, sharding=out_sharding)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/venv/lib/python3.11/site-packages/jax/_src/numpy/util.py", line 278, in _broadcast_to
    shape = core.canonicalize_shape(shape)  # type: ignore[arg-type]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Shapes must be 1D sequences of concrete values of integer type, got (16, JitTracer<~int32[]>, JitTracer<~int32[]>, JitTracer<~int32[]>, 2).      
If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.        

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/train.py", line 568, in <module>
    main()
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/train.py", line 548, in main   
    trainer.train()
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/train.py", line 463, in train  
    avg_loss, avg_box_loss, avg_cls_loss = self.train_epoch(dataloader)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/train.py", line 306, in train_epoch
    loss, box_loss, cls_loss = self.train_fn(images)
                               ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/compile/function/types.py", line 1038, in __call__ 
    outputs = vm() if output_subset is None else vm(output_subset=output_subset)
              ^^^^
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/basic.py", line 669, in thunk
    raise_with_op(self.fgraph, output_nodes[0], thunk)
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/utils.py", line 526, in raise_with_op
    raise exc_value.with_traceback(exc_trace)       
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/basic.py", line 665, in thunk
    outputs = fgraph_jit(*(x[0] for x in thunk_inputs))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/tmpin84sjdj", line 1123, in jax_funcified_fgraph
    tensor_variable_556 = alloc_1(tensor_variable_555, tensor_variable_400, tensor_variable_412, tensor_variable_410, tensor_variable_406, tensor_constant_6)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/tensor_basic.py", line 46, in alloc
    res = jnp.broadcast_to(x, shape)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/venv/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 3070, in broadcast_to
    return util._broadcast_to(array, shape, sharding=out_sharding)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo/venv/lib/python3.11/site-packages/jax/_src/numpy/util.py", line 278, in _broadcast_to
    shape = core.canonicalize_shape(shape)  # type: ignore[arg-type]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Shapes must be 1D sequences of concrete values of integer type, got (16, JitTracer<~int32[]>, JitTracer<~int32[]>, JitTracer<~int32[]>, 2).      
If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
The error occurred while tracing the function jax_funcified_fgraph at /tmp/tmpin84sjdj:1 for jit. This value became a tracer due to JAX operations on these lines:

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:bool[] = eq 256:i32[] 256:i32[]       
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

  operation a:i32[] = max -819200:i32[] 1:i32[]     
    from line /home/ubuntu/pytensor-workshop-demo/pytensor/link/jax/dispatch/elemwise.py:18:15 (jax_funcify_Elemwise.<locals>.elemwise_fn)

(Additional originating lines are not shown.)       
Apply node that caused the error: Sub(True_div.0, mean)
Toposort index: 2983
Inputs types: [TensorType(float32, shape=()), TensorType(float32, shape=())]
Inputs shapes: [(16, 3, 320, 320), (6,), (128,), (64,), (64,), (64,), (32,), (128,), (64,), (256,), (128,), (256,), (128,), (256,), (128,), (256,), (128,), (64,), (128,), (64,), (32,), (64,), (32,), (16,), (32,), (16,), (16, 3, 3, 3), (16,), (16,), (16,), (32, 16, 3, 3), (32,), (32,), (32,), (16, 32, 1, 1), (16,), (16,), (16,), (16,), (16,), (16, 16, 3, 3), (16,), (16,), (16,), (16, 16, 3, 3), (16,), (16,), (16,), (32, 32, 1, 1), (32,), (32,), (32,), (64, 32, 3, 3), (64,), (64,), (64,), (32, 64, 1, 1), (32,), (32,), (32,), (32,), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (64, 64, 1, 1), (64,), (64,), (64,), (128, 64, 3, 3), (128,), (128,), (128,), (64, 128, 1, 1), (64,), (64,), (64,), (64,), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (128,), (256, 128, 3, 3), (256,), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (128,), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128,), (256, 256, 1, 1), (256,), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (128,), (256, 512, 1, 1), (256,), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128,), (256, 256, 1, 1), (256,), (256,), (256,), (64, 384, 1, 1), (64,), (64,), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (128,), (32, 192, 1, 1), (32,), (32,), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32,), (64, 64, 1, 1), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 192, 1, 1), (64,), (64,), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (128,), (6, 128, 1, 1), (6,), (6,), (6,), (16, 3, 3, 3), (16,), (16,), (32, 16, 3, 3), (32,), (32,), (16, 32, 1, 1), (16,), (16,), (32, 32, 1, 1), (32,), (32,), (16, 16, 3, 3), (16,), (16,), (16, 16, 3, 3), (16,), (16,), (64, 32, 3, 3), (64,), (64,), (32, 64, 1, 1), (32,), (32,), (64, 64, 1, 1), (64,), (64,), (32, 32, 3, 3), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (128, 64, 3, 3), (128,), (128,), (64, 128, 1, 1), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (256, 128, 3, 3), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (256, 256, 1, 1), (256,), (256,), (128, 128, 3, 3), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (128, 256, 1, 1), (128,), (128,), (256, 512, 1, 1), (256,), (256,), (128, 256, 1, 1), (128,), (128,), (128, 128, 3, 3), (128,), (128,), (256, 256, 1, 1), (256,), (256,), (64, 384, 1, 1), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (32, 192, 1, 1), (32,), (32,), (64, 64, 1, 1), (64,), (64,), (32, 32, 3, 3), (32,), (32,), (32, 32, 3, 3), (32,), (32,), (64, 64, 3, 3), (64,), (64,), (64, 192, 1, 1), (64,), (64,), (128, 128, 1, 1), (128,), (128,), (64, 64, 3, 3), (64,), (64,), (64, 64, 3, 3), (64,), (64,), (128, 128, 3, 3), (128, 128, 3, 3), (128,), (128,), (128,), (128,), (128, 384, 1, 1), (128, 384, 1, 1), (128,), (128,), (128,), (128,), (256, 256, 1, 1), (256, 256, 1, 1), (256,), (256,), (256,), (256,), (128, 128, 3, 3), (128, 128, 3, 3), (128,), (128,), (128,), (128,), (128, 128, 3, 3), (128, 128, 3, 3), (128,), (128,), (128,), (128,), (6, 64, 1, 1), (6, 64, 1, 1), (6,), (6,), (6,), (6,), (6, 128, 1, 1), (6,), (6,), (6, 256, 1, 1), (6, 256, 1, 1), (6,), (6,), (6,), (6,)]
Inputs strides: [(1228800, 4, 3840, 12), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (108, 36, 12, 4), (4,), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (4,), (128, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (4,), (128, 4, 4, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (2048, 4, 4, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4,), (1536, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (768, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (768, 4, 4, 4), (4,), (4,), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (4,), (108, 36, 12, 4), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (128, 4, 4, 4), (4,), (4,), (128, 4, 4, 4), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (576, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (2048, 4, 4, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (4608, 36, 12, 4), (4,), (4,), (1024, 4, 4, 4), (4,), (4,), (1536, 4, 4, 4), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (768, 4, 4, 4), (4,), (4,), (256, 4, 4, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (1152, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (768, 4, 4, 4), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (2304, 36, 12, 4), (4,), (4,), (4608, 36, 12, 4), (4608, 36, 12, 4), (4,), (4,), (4,), (4,), (1536, 4, 4, 4), (1536, 4, 4, 4), (4,), (4,), (4,), (4,), (1024, 4, 4, 4), (1024, 4, 4, 4), (4,), (4,), (4,), (4,), (4608, 36, 12, 4), (4608, 36, 12, 4), (4,), (4,), (4,), (4,), (4608, 36, 12, 4), (4608, 36, 12, 4), (4,), (4,), (4,), (4,), (256, 4, 4, 4), (256, 4, 4, 4), (4,), (4,), (4,), (4,), (512, 4, 4, 4), (4,), (4,), (1024, 4, 4, 4), (1024, 4, 4, 4), (4,), (4,), (4,), (4,)]
Inputs values: ['not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown', 'not shown']
Outputs clients: [[output[0](Sub.0)]]

HINT: Re-running with most PyTensor optimizations disabled could provide a back-trace showing when this node was created. This can be done by setting the PyTensor flag 'optimizer=fast_compile'. If that does not work, PyTensor optimizations can be disabled with 'optimizer=None'.
HINT: Use the PyTensor flag `exception_verbosity=high` for a debug print-out and storage map footprint of this Apply node.
(venv) ubuntu@192-222-58-23:~/pytensor-workshop-demo/examples/onnx/onnx-yolo-demo$