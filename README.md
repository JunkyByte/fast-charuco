# fast-charuco
Refer to https://github.com/JunkyByte/deepcharuco for the models / setup.  
This is a slightly improved inference setup for the deepcharuco models, using onnx + (slightly) parallel inference for performance gain. 

Performance comparison on GTX1080ti
on an image with 8/16 kpts found (deepc + refinenet).
```
8 images batch
Vanilla deepc -> 25fps
Parallel + onnx -> 44fps
Gain: +56%

4 images batch
Vanilla deepc -> 49fps
Parallel + onnx -> 92fps
Gain: +53%

2 images batch
Vanilla deepc -> 99fps
Parallel + onnx -> 165fps
Gain: +60%

1 image batch
Vanilla deepc -> 195fps
Parallel + onnx -> 300fps
Gain: +65%
```

Code used for conversion
(adapt it for `deepc.onnx`)
```python
filepath = "refinenet.onnx"
dynamic_axes = {'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}}
input_sample = torch.randn((1, 1, 24, 24))

refinenet.to_onnx(filepath, input_sample, export_params=True,
		  input_names = ['input'], output_names = ['output'],
		  dynamic_axes=dynamic_axes)
```
