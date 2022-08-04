# MMD Loss in PyTorch
An implementation of Maximum Mean Discrepancy (MMD) as a differentiable loss in PyTorch.

Based on [ZongxianLee's popular repository](https://github.com/ZongxianLee/MMD_Loss.Pytorch).
Same functionality but fixed bugs and simplified the code. Importantly, this implementation does not restrict both compared sets to be of the same size.

## See Also

* [Generative Moment Matching Networks](https://arxiv.org/abs/1502.02761)
* [A kernel two-sample test](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjoisyGkK35AhUDrKQKHfq0AyIQFnoECAgQAQ&url=https%3A%2F%2Fwww.jmlr.org%2Fpapers%2Fvolume13%2Fgretton12a%2Fgretton12a.pdf&usg=AOvVaw0Iu5L5aVAIXUYPdt1Tb8Lg)