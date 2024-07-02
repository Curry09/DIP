# DIP
dehaze network

利用生成对抗模型 GAN 和 Transformer 改进 DM2F-Net。在模型创新上，笔者将 Transformer 模块接入 DM2F-Net，
在网络的浅层利用卷积层 (CNN) 提取局部特征，在网络的深层利用 Transformer 做特征间的融合。CNN 擅长捕捉局部信息，而
Transformer 能够处理长距离的依赖关系，因此这种设计有利于整合来自不同区域的特征。在方法创新上，笔者借鉴了生成对
抗模型，利用纳什均衡策略，来增强模型的泛化能力，通过引入一个对抗性的训练过程，模型能够学习到更加丰富和逼真的
输出。改进后的结果与 DM2F-Net 相比，在 SOTS 数据集上，PSNR 提高了 3.1，SSIM 提高了 1%，O-HAZE 数据集上，PSNR
提高了 1.2，SSIM 提高了 1%。
