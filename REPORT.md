# Model Compress Report
## 1. Baseline Model
- Model: VGG (2014) is a ConvNet introduced in paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition". VGG was the first network to use the concept of convolutional blocks, repeating architectures including Conv2d, BatchNorm, etc. VGG19 has 19 layers and use conv 3x3 on the whole network. Compare to ResNet or Inception, VGG19 is quite heavy (39M params). So I want to try compressing this model.

- Train: We use the dataset `cifar-100` and train for 100 epochs. The init learning rate is 0.1 and divide by 5 at 30th, 60th, 90th epochs.
    - `top1err = 0.323`
    - `top5err = 0.119`

## 2. Methods
### Quantization
Quantization is a technique to reduce the numerical precision (bits) of the weights and/or acitvations map in the model. For example, instead of using Float32 to represent the value of a number, quantization just use Int8 (8 bits) to represent. This helps to reduce model size and speed up runtime.

### Pruning
Pruning is a technique to remove unimportant weights from the model (weights close to 0). This helps to reduce model size and speed up runtime and allows fine-tune with sample data to keep the accuracy.

## 3. Experiments
### Implementation Details
Hardware:
- CPU: `Intel(R) Xeon(R) CPU @ 2.20GHz`
- GPU: `NVIDIA T4 GPU`
- CUDA version: `11.8`
- cuDNN: `8.9.0`
- Pytorch: `2.0.1`
- ONNX: `1.14.0`
- ONNXRUNTIME: `1.15.1`

I use pytorch and measure the speed on Intel(R) Xeon(R) CPU. You can see the detailed source code at [compress.ipynb](https://github.com/zero-nnkn/pytorch-compression-cifar100/blob/master/compress.ipynb)
- Quantization (`torch.quantization`): I apply two techniques of post-training dynamic quantization and post-training static quantization. I also perform experiments on two different data types, float16 and int8.Specifically, With static quantization, I test the effect of fusing the Conv2d and BatchNorm.
- Pruning (`torch.prune`): I apply two different pruning techniques: local pruning and global pruning. Local pruning is to prune the parameters module by module. Local pruning is pruning of separate modules, without affecting each other. Global pruning prune over the entire model at a given scale. We apply the pruning rate of 60% for experiments.
- ONNX: I export model to ONNX format and measure inference time.

### Results

| Method | Type | Top1err | Top5err | Runtime (ms) | Model size (MB) | 
|-|-|:-:|:-:|:-:|:-:|
| Baseline | float32 | 0.323 | 0.119 | 21.05 | 157.38 |
| Post-Training Dynamic | float16 | **0.323** | **0.119** | 20.80 | 157.38 |
| Post-Training Dynamic | int8 | **0.323** | **0.119** | 20.23 | 99.53 |
| Post-Training Static | int8 | 0.326 | 0.122 | **10.19** | 39.72 |
| Post-Training Static Fuse | int8 | 0.324 | 0.120 | 10.51 | **39.60** |
| Pytorch Global pruning (0.6) | float32 | 0.363 | 0.142 | 22.44 | 157.38 |
| Pytorch Local pruning (0.6) | float32 | 0.492 | 0.233 | 21.30 | 157.38 |
| ONNX | ONNX |  0.323 | 0.119 |  17.32 | 157.47 | 

### Analysis
The quantization methods show good results. Of these, the best is post-training static quantization. Specifically, the model after quantization (fuse), has increased top1 error 0.3% compared to baseline method, while increasing speed 2 times and and reducing memory usage by 3.97 times.

The torch pruning functions are currently at research phase. They just use the mask to control pruning params. Therefore, the model size and speed are not reduced, but even increased.

Converting the model to onnx format also increases the speed of inference significantly (21.05 -> 17.32).

### Checkpoints
- Baseline: [here](https://drive.google.com/file/d/1KT8Ww6t24V92SfkC56tbvCGyG_LMPQ-0/view?usp=sharing)
- ONNX model: [here](https://drive.google.com/file/d/134-4Pje5awnFxqflo9n-qOgEfMdx1tjK/view?usp=sharing)
- Best compressed model: [here](https://drive.google.com/file/d/1TjCNN2PNXwrPK27bYqtuypwSRwl8Gbpg/view?usp=sharing)

## 4. Future works
- Try other tools to quantize, because torch currently supports quantize with some layers and only for CPU execution.
- Try "real" pruning method.
- Try weight clustering (torch not supported).

## 5. References
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
- [Pytorch Quantization in Practice](https://pytorch.org/blog/quantization-in-practice/)
- [Pytorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [PyTorch Pruning](https://leimao.github.io/blog/PyTorch-Pruning/)
