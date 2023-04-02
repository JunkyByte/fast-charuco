# fast-charuco
Parallel charuco based pose estimation in python, using deepcharuco.

Performance comparison on Gtx1080ti
on an image with 8/16 kpts found.
```
8 images batch
Vanilla deepc -> ~25fps
Parallel + onnx -> ~44fps

4 images batch
Vanilla deepc -> 49fps
Parallel + onnx -> 92fps

2 images batch
Vanilla deepc -> 99fps
Parallel + onnx -> 165fps

1 image batch
Vanilla deepc -> 195fps
Parallel + onnx -> 300fps
```

Code used for conversion
(adapt it for `deepc.onnx`)
```
filepath = "refinenet.onnx"
dynamic_axes = {'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}}
input_sample = torch.randn((1, 1, 24, 24))

refinenet.to_onnx(filepath, input_sample, export_params=True,
		  input_names = ['input'], output_names = ['output'],
		  dynamic_axes=dynamic_axes)
```
