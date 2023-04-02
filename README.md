# fast-charuco
Parallel charuco based pose estimation in python, using opencv and deepcharuco.


```
filepath = "refinenet.onnx"
dynamic_axes = {'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}}
input_sample = torch.randn((1, 1, 24, 24))

refinenet.to_onnx(filepath, input_sample, export_params=True,
		  input_names = ['input'], output_names = ['output'],
		  dynamic_axes=dynamic_axes)
```
