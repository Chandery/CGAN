# CGAN的探索和学习

## CGAN

CGAN（Conditional Generative Adversarial Network）模型是一种 深度学习模型，属于生成对抗网络（GAN）的一种变体。相比于GAN，它引入了条件信息（y），使得生成器可以生成与给定条件相匹配的合成数据，从而提高了生成数据的可控性和针对性。

## 相关探索和学习

### 初步探索

常规的GAN之所以不能控制条件，是因为进入生成器的噪音是随机生成的，并不包含条件信息，因此生成模型在生成的时候我们无法进行条件控制。CGAN的主要思想就是在在随机噪声中加入条件信息

#### **第一个想法**

在手写数字的条件生成中，100维的噪音向量，保留90维的随机向量，剩下的10维直接用独热的条件码，以达到控制条件的效果。

![image-20241101160625238](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/image-20241101160625238.png)

#### **具体实现：**

```python
def make_noise(self,...):
    num_classes = cond.max().item() + 1  # 假设cond中的标签是从0开始的
    noise_1 = torch.randn(batch_size, nz-num_classes, 1, 1, device=self.device) #随机噪声
    noise_2 = F.one_hot(cond, num_classes).float().unsqueeze(2).unsqueeze(3).to(self.device) #独热编码
    noise = torch.cat((noise_1, noise_2), 1) #连接起来 [bs, 100, 1, 1]
    return noise
```

#### **效果**

![image-20241101161952194](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/image-20241101161952194.png)

**200->250?**

### 使用Embedding

显然上面的做法效果很差

原因？

1. 条件占比少
2. 使用独热的条件是给人看的，机器要识别这种分布过于困难，特别在对抗网络中很容易还没有学到就崩了

我们需要一个Embedding来实现把条件嵌入噪声中。



#### nn.Embedding

![image-20241101162547075](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/image-20241101162547075.png)一般用到前两个参数：

1. 类别
2. 嵌入向量的维度

在这个问题中类别是10，嵌入向量维度沿用100

这样做的话，噪声向量100维都包含条件信息。

#### 代码处理

```python
# Generator
self.embedding = nn.Embedding(opt_class, nz)

def forward(self, input):
        x = self.embedding(input)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.main(x)
```

在进入模型之前先使用embedding层进行编码，这种办法模型对条件分布到噪声向量上的映射关系也是可以学习的

#### 效果

![image-20241101164145057](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/image-20241101164145057.png)

> 明显开始有分类了
>
> 效果还是不好
>
> 问题在哪？

**Discriminator没有关于条件的信息！**

### 对Discriminator进行修改

没有沿用DCGAN的卷积判别器

```python
def forward(self, input, label):
        x = torch.cat((input.view(input.size(0), -1), self.label_embedding(label.long())), -1)
        return self.main(x)
```

把label embedding以后和图像压平以后的序列concat起来

#### 效果

![image-20241101191627664](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/image-20241101191627664.png)

![image_9](https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/image_9.png)

可能因为epoch不足的原因，或者是判别器摒弃了卷积模式，最后的结果不算太好，但是已经很明显有0～9的分别了。

### 后续

如何改进？
