# 识别猫

## 一、导入包

1、numpy、matpoltlib、h5py、scipy

2、导入数据集

test_catvnoncat.h5

train_catvnoncat.h5

3、lr_utils.py处理数据集

## 二、预处理数据集

预处理数据集的常见步骤是：

- 找出数据的尺寸和维度（m_train，m_test，num_px等）
- 重塑数据集，以使每个示例都是大小为（num_px  *num_px \* 3，1）的向量
- “标准化”数据

## 三、逻辑回归算法的一般架构

![image-20210430193110260](C:\Users\13570\Desktop\image-20210430193110260.png)

![image-20210430193152541](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210430193152541.png)

-   初始化模型参数
-    通过最小化损失来学习模型的参数
-    使用学习到的参数进行预测（在测试集上）
-    分析结果并得出结论

## 四、构建算法的各个部分

建立神经网络的主要步骤是：
①.定义模型结构（例如输入特征的数量）
②.初始化模型的参数
③.循环：

-    计算当前损失（正向传播）
-    计算当前梯度（向后传播）
-    更新参数（梯度下降）

#### 1、辅助函数

实现sigmoid（）函数。

计算![image-20210430193606800](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210430193606800.png)

```python
def sigmoid(z):
```

#### 2、初始化参数

将w初始化为零的向量。

```python
def initialize_with_zeros(dim):
```

#### 3、前向和后向传播

实现函数propagate（）来计算损失函数及其梯度。

正向传播：

- 得到X
- 计算![image-20210430193812214](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210430193812214.png)
- 计算损失函数：![image-20210430193821027](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210430193821027.png)

```python
def propagate(w, b, X, Y):
```

#### 4、优化函数

- 初始化参数。
- 计算损失函数及其梯度。
- 使用梯度下降来更新参数。

目标是通过最小化损失函数 J 来学习 w 和 b。 对于参数θ，更新规则为θ=θ−α dθ，其中α是学习率。

```python
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
```

#### 5、预测分类

上一个函数将输出学习到的w和b。 我们能够使用w和b来预测数据集X的标签。实现`predict（）`函数。 预测分类有两个步骤：
1.计算![image-20210430194224852](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210430194224852.png)
2.将a的项转换为0（如果激活<= 0.5）或1（如果激活> 0.5），并将预测结果存储在向量“ Y_prediction”中

```python
def predict(w, b, X):
```

#### 6、合并到model中

将所有构件（在上一部分中实现的功能）以正确的顺序放在一起，从而得到整体的模型结构。

-    Y_prediction对测试集的预测
-    Y_prediction_train对训练集的预测
-    w，损失，optimize（）输出的梯度

```python
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
```

训练model

```python
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
```

**预期输出**:
train accuracy: 99.04306220095694 %
test accuracy: 70.0 %

**评价**：训练准确性接近100％。 这是一个很好的情况：模型正在运行，并且具有足够的容量来适合训练数据。 测试误差为68％。 考虑到我们使用的数据集很小，并且逻辑回归是线性分类器，对于这个简单的模型来说，这实际上还不错。

此外，该模型明显适合训练数据。 

## 五、分析

#### 1、绘制损失函数-梯度图像

```python
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
```

![image-20210430194701654](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210430194701654.png)

**解释**：
损失下降表明正在学习参数。 但是，你看到可以在训练集上训练更多模型。 尝试增加上面单元格中的迭代次数，然后重新运行这些单元格。 你可能会看到训练集准确性提高了，但是测试集准确性却降低了。 这称为过度拟合。

#### 2、学习速率的选择

进行进一步分析，并研究如何选择学习率α。

学习率的选择

提醒：
为了使梯度下降起作用，必须明智地选择学习率。 学习率α决定我们更新参数的速度。 如果学习率太大，我们可能会“超出”最佳值。 同样，如果太小，将需要更多的迭代才能收敛到最佳值。 这也是为什么调整好学习率至关重要。

```python
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```

![image-20210430194846398](C:\Users\13570\AppData\Roaming\Typora\typora-user-images\image-20210430194846398.png)

**解释**：

- 不同的学习率会带来不同的损失，因此会有不同的预测结果。
- 如果学习率太大（0.01），则成本可能会上下波动。 它甚至可能会发散（尽管在此示例中，使用0.01最终仍会以较高的损失值获得收益）。
- 较低的损失并不意味着模型效果很好。当训练精度比测试精度高很多时，就会发生过拟合情况。
- 在深度学习中，我们通常建议你：
  -    选择好能最小化损失函数的学习率。
  -    如果模型过度拟合，请使用其他方法来减少过度拟合。 

## 六、测试

```python
my_image = "cat_in_iran.jpg"   

fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```

## 七总结

1. 预处理数据集很重要。
2. 如何实现每个函数：initialize（），propagation（），optimize（），并用此构建一个model（）。
3. 调整学习速率（这是“超参数”的一个示例）可以对算法产生很大的影响。 
4. 由于数据集太小，测试准确率不是很高。