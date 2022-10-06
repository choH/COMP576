# Assignment 1

> Shaochen (Henry) Zhong
> COMP 576, Fall 2022, by Prof. Patel

---

## Note to grader on late submission

I had some trouble running TensorFlow on Google Colab. The code that works will throw off some unreadable errors upon another execution — which made me spend time that is way more than necessary to debug my no-bug code, as I thought it was my code that was buggy but not the environment. 

After I realized it was an environment issue, I found out it has something to do with the execution mechanism of TensorFlow and can be avoided with some eager execution settings. Unfortunately, those settings are better supported in TF v2, where the provided skeleton code & tutorial are written in TF v1. Thus, I opted to restart my runtime environment before every execution manually.

Long story short, this is why I am turning in a late submission, and I will use one of my two no-penalty late submission days for this assignment. Thank you.

## 1. Backpropagation in a Simple Neural Network

### 1.a. Dataset

![figure_1a](media/16648828391542/figure_1a.png)

### 1.b. Activation Function

#### 1.b.1.
Please refer to `three_layer_neural_network.py`.

#### 1.b.2.


For 'Tanh':
\\[
\begin{align*}
f(z) &= \tanh(z) = \frac{\sinh(z)}{\cosh(z)} \newline
f'(z) &= \frac{d}{dz} \frac{\sinh(z)}{\cosh(z)} = \frac{\cosh^2(z) - \sinh^2(z)}{\cosh^2(z)} \newline
&= 1 - \frac{\sinh^2(z)}{\cosh^2(z)} = 1 - \tanh^2(z)
\end{align*}
\\]

For 'Sigmoid':
\\[
\begin{align*}
f(z) &= \sigma(z) = \frac{1}{1+e^{-z}} \newline
f'(z) &= \frac{d}{dz} (1+e^{-z})^{-1} = -(1+e^{-z})^{-2} \cdot (\frac{d}{dz} 1+e^{-z}) \newline
&= -(1+e^{-z})^{-2}  \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^{2} } \newline
&= \frac{(1 + e^{-z}) - 1}{(1+e^{-z})^{2} } = \frac{1}{1+e^{-z}} - \frac{1}{(1+e^{-z})^{2} } \newline
&= \sigma(z) - \sigma(z)^2 = \sigma(z)(1 - \sigma(z)
\end{align*}
\\]



For 'ReLU':
\\[
f(x) = \text{ReLU}(z) = \begin{cases}
z & z \ge 0 \newline
0 & \text{otherwise}
\end{cases}
\\]

\\[
f'(x) =  \frac{d}{dz} \text{ReLU}(z) = \begin{cases}
\frac{dz}{dz} = 1 & z > 0 \newline
\text{undefined} & z = 0 \newline
\frac{d0}{dz} = 0 & z > 0
\end{cases}
\\]


#### 1.b.3.
Please refer to `three_layer_neural_network.py`.


### 1.c. Build the Neural Network

#### 1.c.1.
Please refer to `three_layer_neural_network.py`.

#### 1.c.2.
Please refer to `three_layer_neural_network.py`.

### 1.d. Backward Pass

#### 1.d.1.

Let

\\[
\frac{dL}{dZ_{2}} = \frac{dL}{d\hat{y}} \frac{d\hat{y}}{dz_2} = \frac{1}{N} \sum^{N}_{i = 1} (\hat{y} - y)
\\]

\\[
\frac{dL}{dW_2} = \frac{dL}{dZ_{2}} \frac{dZ_{2}}{dW_{2}} = a_1^T \cdot \frac{dL}{dZ_{2}}
\\]

\\[
\frac{dL}{db_2} = \frac{dL}{dZ_{2}} \frac{dZ_{2}}{db_2} = \sum \frac{dL}{dZ_{2}}
\\]

\\[
\frac{dL}{dZ_{1}} = \frac{dL}{dZ_{2}} \frac{dZ_{2}}{da_1} \frac{da_1}{dZ_1} = \frac{1}{N} \sum^{N}_{i = 1} (\hat{y} - y) \cdot W_2^T \cdot da_1
\\]

\\[
\frac{dL}{dW_1} = X^T \cdot \frac{dL}{dZ_{1}}
\\]

\\[
\frac{dL}{db_1} =r \sum \frac{dL}{dZ_{1}}
\\]

#### 1.d.2.
Please refer to `three_layer_neural_network.py`.


### 1.e. Training

#### 1.e.1

![figure_1e1_tanh](media/16649039472344/figure_1e1_tanh.png)
`actFun_type = 'Tanh'` final loss: 0.070758

![figure_1e1_sigmoid](media/16649039472344/figure_1e1_sigmoid.png)
`actFun_type = 'Sigmoid'` final loss: 0.078155

![figure_1e1_relu](media/16649039472344/figure_1e1_relu.png)
`actFun_type = 'ReLU'` final loss: 0.071219

The three activation functions seem to achieve similar result with the exception that `ReLU`'s decision boundary is a lot less smooth. This might due to the piecewise nature of it.


#### 1.e.2


![figure_1e2](media/16649039472344/figure_1e2.png)
`actFun_type = 'Tanh'` and `nn_hidden_dim = 100`; final loss: 0.032493

![figure_1e2_loss](media/16649039472344/figure_1e2_loss.png)

It is not surprising that a MLP with larger amount of hidden units may fit the data better — which is both reflected on the visualization of decision boundary and the loss. The loss of `nn_hidden_dim = 100` is always lower than its `nn_hidden_dim = 3` counterparts after 1000 iterations.

### 1.f Training a Deeper Network


#### On `make_moon`:

I implemented my `DeepNeurralNetwork` with the `Layer` helper class as suggested by the instruction. Please refer to `n_layer_neural_network.py` for details.

The instruction of this task is vague as it did not specify what (and how many) *different network configurations* are expected. So I will compare network that's wider, denser, and wider+denser among three activation functions.

![figure_1e2_tanh_15x3](media/16649039472344/figure_1e2_tanh_15x3.png)
`Tanh` with `15x3` layers; final loss: 0.021276

![figure_1e2_sigmoid_15x3](media/16649039472344/figure_1e2_sigmoid_15x3.png)
`Sigmoid` with `15x3` layers; final loss: 0.091873

![figure_1e2_relu_15x3](media/16649039472344/figure_1e2_relu_15x3.png)
`ReLU` with `15x3` layers; final loss: 0.693148



![figure_1e2_sigmoid_3x3](media/16649039472344/figure_1e2_sigmoid_3x3.png)
`Sigmoid` with `3x3` layers; final loss: 0.092737

![figure_1e2_relu_15x3_zeros](media/16649039472344/figure_1e2_relu_15x3.png)
`ReLU` with `3x3` layers; final loss: 0.693148


![figure_1e2_tanh_15x6](media/16649039472344/figure_1e2_tanh_15x6-1.png)
`Tanh` with `15x6` layers; final loss: 0.030329

![figure_1e2_relu_15x6](media/16649039472344/figure_1e2_relu_15x3.png)
`ReLU` with `15x6` layers; final loss: 0.693148

![figure_1e2_sigmoid_15x6](media/16649039472344/figure_1e2_relu_15x3.png)
`Sigmoid` with `15x6` layers; final loss: 0.693161


Takeaways:
* `Tanh` seems to perform the best among the conducted experiments.
* With a very large dimension/layers, we will encounter invalid value/division errors.
* Increasing the capacity of the network doesn't always seem to help, at least not in a network this vanilla.


#### On `make_gaussian_quantiles`:

For other datasets, I implemented support for `make_gaussian_quantiles` and `make_circles`. In this report, I will focus on the previous dataset at it is non-binary (set `n_classes = 3`), and therefore potentially more interesting.

![figure_1e2_gq_tanh_15x3](media/16649039472344/figure_1e2_gq_tanh_15x3.png)

`Tanh` with `15x3` layers; final loss: 0.042408

![figure_1e2_gq_sigmoid_15x3](media/16649039472344/figure_1e2_gq_sigmoid_15x3.png)
`Sigmoid` with `15x3` layers; final loss: 0.130318

![figure_1e2_gq_reluy_15x3](media/16649039472344/figure_1e2_gq_reluy_15x3.png)
`ReLU` with `15x3` layers; final loss: 0.005850

It seems `ReLU` may offer the best fitting with the 'intermedia' network configuration we tried on `make_moons`. That being said, it still can't take full advantage of scaling — for example, in the following setup, the network is less fitted to the dataset even being deeper and wider.

![figure_1e2_gq_relu_20x4](media/16649039472344/figure_1e2_gq_relu_20x4.png)

`ReLU` with `20x4` layers; final loss: 0.785045


---

## 2. Training a Simple Deep Convolutional Network on MNIST


### 2.a. Build and Train a 4-layer DCN

#### 2.a.1-4. 

Please refer to `dcn_mnist.py` for the code completion.

#### 2.a.5.

The final test accuracy is `0.985`.

#### 2.a.6.

![fig_2a6](media/16649039472344/fig_2a6.png)
Train loss visualization.

### 2.b. More on Visualizing Your Training

#### First Layer

![figure_2b_l1_hpool1](media/16649039472344/figure_2b_l1_hpool1.png)
![figure_2b_l1_hconv1](media/16649039472344/figure_2b_l1_hconv1.png)
![figure_2b_l1_bconv1](media/16649039472344/figure_2b_l1_bconv1.png)
![figure_2b_l1_wconv1](media/16649039472344/figure_2b_l1_wconv1.png)

![figure_2b_l1_hpool1_hist](media/16649039472344/figure_2b_l1_hpool1_hist.png)
![figure_2b_l1_hconv1_hist](media/16649039472344/figure_2b_l1_hconv1_hist.png)
![figure_2b_l1_bconv1_hist](media/16649039472344/figure_2b_l1_bconv1_hist.png)
![figure_2b_l1_wconv1_hist](media/16649039472344/figure_2b_l1_wconv1_hist.png)

#### Second Layer

![figure_2b_l2_hpool2](media/16649039472344/figure_2b_l2_hpool2.png)
![figure_2b_l2_hconv2](media/16649039472344/figure_2b_l2_hconv2.png)
![figure_2b_l2_bconv2](media/16649039472344/figure_2b_l2_bconv2.png)
![figure_2b_l2_wconv2](media/16649039472344/figure_2b_l2_wconv2.png)

![figure_2b_l2_hpool2_hist](media/16649039472344/figure_2b_l2_hpool2_hist.png)
![figure_2b_l2_hconv2_hist](media/16649039472344/figure_2b_l2_hconv2_hist.png)
![figure_2b_l2_bconv2_hist](media/16649039472344/figure_2b_l2_bconv2_hist.png)
![figure_2b_l2_wconv2_hist](media/16649039472344/figure_2b_l2_wconv2_hist.png)


#### Dense Layer


![figure_2b_dl_hpool2](media/16649039472344/figure_2b_dl_hpool2.png)
![figure_2b_dl_hfc1](media/16649039472344/figure_2b_dl_hfc1.png)
![figure_2b_dl_bfc1](media/16649039472344/figure_2b_dl_bfc1.png)
![figure_2b_dl_wfc1](media/16649039472344/figure_2b_dl_wfc1.png)

![figure_2b_dl_hpool2_hist](media/16649039472344/figure_2b_dl_hpool2_hist.png)
![figure_2b_dl_hfc1_hist](media/16649039472344/figure_2b_dl_hfc1_hist.png)
![figure_2b_dl_bfc1_hist](media/16649039472344/figure_2b_dl_bfc1_hist.png)
![figure_2b_dl_wfc1_hist](media/16649039472344/figure_2b_dl_wfc1_hist.png)

#### Dropout

![figure_2b_do_kp](media/16649039472344/figure_2b_do_kp.png)
![figure_2b_do_hfc1](media/16649039472344/figure_2b_do_hfc1.png)

![figure_2b_do_kp_hist](media/16649039472344/figure_2b_do_kp_hist.png)
![figure_2b_do_hfc1_hist](media/16649039472344/figure_2b_do_hfc1_hist.png)


#### Softmax

![figure_2b_sm_yconv](media/16649039472344/figure_2b_sm_yconv.png)
![figure_2b_sm_bfc2](media/16649039472344/figure_2b_sm_bfc2.png)
![figure_2b_sm_wfc2](media/16649039472344/figure_2b_sm_wfc2.png)


![figure_2b_sm_yconv_hist](media/16649039472344/figure_2b_sm_yconv_hist.png)
![figure_2b_sm_bfc2_hist](media/16649039472344/figure_2b_sm_bfc2_hist.png)
![figure_2b_sm_wfc2_hist](media/16649039472344/figure_2b_sm_wfc2_hist.png)


### 2.c. Time for More Fun!!!

I used Kaiming He's normal initialization, leaky ReLU with `alpha = 0.2`, and vanilla SGD with `lr = 0.01` to give it some (leaky) ResNet flavor. The final test accuracy achieved is 0.9859, which is slightly higher the given setting (0.985).

Before I proceed with a ton of screenshots, the takeaways I observed are:

* This setting's plots are a lot more unstable, which suggest an adaptive learning rate might help.
* This setting's scaler plots look a lot smoother than the given setting in the first two layers.
* This setting's second layer converges visibly slower, where the given setting almost converges immediately — which suggests this setting is exploring more.

#### Loss
![figure_2c_loss](media/16649039472344/figure_2c_loss.png)

#### First Layer

![figure_2c_l1_hpool1](media/16649039472344/figure_2c_l1_hpool1.png)
![figure_2c_l1_hconv1](media/16649039472344/figure_2c_l1_hconv1.png)
![figure_2c_l1_bconv1](media/16649039472344/figure_2c_l1_bconv1.png)
![figure_2c_l1_wconv1](media/16649039472344/figure_2c_l1_wconv1.png)
![figure_2c_l1_hpool1_hist](media/16649039472344/figure_2c_l1_hpool1_hist.png)
![figure_2c_l1_hconv1_hist](media/16649039472344/figure_2c_l1_hconv1_hist.png)
![figure_2c_l1_bconv1_hist](media/16649039472344/figure_2c_l1_bconv1_hist.png)
![figure_2c_l1_wconv1_hist](media/16649039472344/figure_2c_l1_wconv1_hist.png)


#### Second Layer

![figure_2c_l2_wconv2](media/16649039472344/figure_2c_l2_wconv2.png)
![figure_2c_l2_hpool2](media/16649039472344/figure_2c_l2_hpool2.png)
![figure_2c_l2_hconv2](media/16649039472344/figure_2c_l2_hconv2.png)
![figure_2c_l2_bconv2](media/16649039472344/figure_2c_l2_bconv2.png)
![figure_2c_l2_hpool2_hist](media/16649039472344/figure_2c_l2_hpool2_hist.png)
![figure_2c_l2_hconv2_hist](media/16649039472344/figure_2c_l2_hconv2_hist.png)
![figure_2c_l2_bconv2_hist](media/16649039472344/figure_2c_l2_bconv2_hist.png)
![figure_2c_l2_wconv2_hist](media/16649039472344/figure_2c_l2_wconv2_hist.png)


#### Dense Layer

![figure_2c_dl_hpool2f](media/16649039472344/figure_2c_dl_hpool2f.png)
![figure_2c_dl_hfc1](media/16649039472344/figure_2c_dl_hfc1.png)
![figure_2c_dl_bfc1](media/16649039472344/figure_2c_dl_bfc1.png)
![figure_2c_dl_wfc1](media/16649039472344/figure_2c_dl_wfc1.png)

![figure_2c_dl_hpool2f_hist](media/16649039472344/figure_2c_dl_hpool2f_hist.png)
![figure_2c_dl_hfc1_hist](media/16649039472344/figure_2c_dl_hfc1_hist.png)
![figure_2c_dl_bfc1_hist](media/16649039472344/figure_2c_dl_bfc1_hist.png)
![figure_2c_dl_wfc1_hist](media/16649039472344/figure_2c_dl_wfc1_hist.png)

#### Dropout


![figure_2c_do_kp](media/16649039472344/figure_2c_do_kp.png)
![figure_2c_do_hfc1](media/16649039472344/figure_2c_do_hfc1.png)

![figure_2c_do_kp_hist](media/16649039472344/figure_2c_do_kp_hist.png)
![figure_2c_do_hfc1_hist](media/16649039472344/figure_2c_do_hfc1_hist.png)


#### Softmax

![figure_2c_sm_yconv](media/16649039472344/figure_2c_sm_yconv.png)
![figure_2c_sm_bfc2](media/16649039472344/figure_2c_sm_bfc2.png)
![figure_2c_sm_wfc2](media/16649039472344/figure_2c_sm_wfc2.png)

![figure_2c_sm_yconv_hist](media/16649039472344/figure_2c_sm_yconv_hist.png)
![figure_2c_sm_bfc2_hist](media/16649039472344/figure_2c_sm_bfc2_hist.png)
![figure_2c_sm_wfc2_hist](media/16649039472344/figure_2c_sm_wfc2_hist.png)

---

## Reference

I have referred to [this repository](https://github.com/MingzLiu/ELEC-576-intro2DL/blob/main/Assignment1/dcn_mnist-1.py) for TensorBoard summaries.