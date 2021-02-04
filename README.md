# NLP学习指南

本教程致力于帮助同学们快速入门NLP，并掌握各个任务的SOTA模型。

1. [系统入门方法](#para1)
2. 各任务模型list汇总：[文本分类](#para2cls)、[文本匹配](#para2sts)、[序列标注](#para2sl)、[文本生成](#para2seq2seq)(todo)、[语言模型](#para2lm)
3. 各任务综述&技巧：[文本分类](#review_textcls)、文本匹配、序列标注、文本生成、语言模型

# <a id="para1"/>如何系统地入门

机器学习是一门既重理论又重实践的学科，想一口吃下这个老虎是不可能的，因此学习应该是个**循环且逐渐细化**的过程。

首先要有个全局印象，知道minimum的情况下要学哪些知识点：

**学习路线下载地址**：https://wwi.lanzous.com/iIrHMl2ubed

![](https://tva1.sinaimg.cn/large/008eGmZEly1gn5nf8uhpxj31kq0u0n91.jpg)

之后就可以开始逐个击破，但也不用死磕，控制好目标难度，先用三个月时间进行第一轮学习：

1. 读懂机器学习、深度学习原理，不要求手推公式
2. 了解经典任务的baseline，动手实践，看懂代码
3. 深入一个应用场景，尝试自己修改模型，提升效果

迈过了上面这道坎后，就可以重新回归理论，提高对自己的要求，比如**手推公式、盲写模型、拿到比赛Top**等。

## Step1: 基础原理

机器学习最初入门时对数学的要求不是很高，掌握基础的线性代数、概率论就可以了，正常读下来的理工科大学生以上应该都没问题，可以直接开始学，碰到不清楚的概念再去复习。

统计机器学习部分，建议初学者先看懂**线性分类、SVM、树模型和图模型**，这里推荐李航的「统计学习方法」，薄薄的摸起来没有很大压力，背着也方便，我那本已经翻四五遍了。喜欢视频课程的话可以看吴恩达的「CS229公开课」或者林田轩的「机器学习基石」。但不管哪个教程，都不必要求一口气看完吃透。

深度学习部分，推荐吴恩达的「深度学习」网课、李宏毅的「深度学习」网课或者邱锡鹏的「神经网络与深度学习」教材。先弄懂神经网络的反向传播推导，然后去了解词向量和其他的编码器的核心思想、前向反向过程。

## Step2: 经典模型与技巧

有了上述的基础后，应该就能看懂模型结构和论文里的各种名词公式了。接下来就是了解NLP各个经典任务的baseline，并看懂源码。对于TF和Pytorch的问题不用太纠结，接口都差不多，找到什么就看什么，自己写的话建议Pytorch。

快速了解经典任务脉络可以看综述，建议先了解一两个该任务的经典模型再去看，否则容易云里雾里：

- [2020 A Survey on Text Classification: From Shallow to Deep Learning](https://arxiv.org/pdf/2008.00364v2.pdf)
- [2020 A Survey on Recent Advances in Sequence Labeling from Deep Learning Models](https://arxiv.org/pdf/2011.06727)
- [2020 Evolution of Semantic Similarity - A Survey](https://arxiv.org/pdf/2004.13820)
- [2017 Neural text generation: A practical guide](https://arxiv.org/abs/1711.09534)
- [2018 Neural Text Generation: Past, Present and Beyond](https://arxiv.org/pdf/1803.07133.pdf)
- [2019 The survey: Text generation models in deep learning](https://www.sciencedirect.com/science/article/pii/S1319157820303360)
- [2020 Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)

### 文本分类

文本分类是NLP应用最多且入门必备的任务，TextCNN堪称第一baseline，往后的发展就是加RNN、加Attention、用Transformer、用GNN了。第一轮不用看得太细，每类编码器都找个代码看一下即可，顺便也为其他任务打下基础。

但如果要做具体任务的话，建议倒序去看SOTA论文，了解各种技巧，同时善用知乎，可以查到不少提分方法。

### 文本匹配

文本匹配会稍微复杂些，它有双塔和匹配两种任务范式。双塔模型可以先看SiamCNN，了解完结构后，再深入优化编码器的各种方法；基于匹配的方式则在于句子表示间的交互，了解BERT那种TextA+TextB拼接的做法之后，可以再看看阿里的RE2这种轻量级模型的做法：

### 序列标注

序列标注主要是对Embedding、编码器、结果推理三个模块进行优化，可以先读懂Bi-LSTM+CRF这种经典方案的源码，再去根据需要读论文改进。

### 文本生成

文本生成是最复杂的，具体的SOTA模型我还没梳理完，可以先了解Seq2Seq的经典实现，比如基于LSTM的编码解码+Attention、纯Transformer、GPT2以及T5，再根据兴趣学习VAE、GAN、RL等。

### 语言模型

语言模型虽然很早就有了，但18年BERT崛起之后才越来越被重视，成为NLP不可或缺的一个任务。了解BERT肯定是必须的，有时间的话再多看看后续改进，很经典的如XLNet、ALBERT、ELECTRA还是不容错过的。

## Step3: 实践优化

上述任务都了解并且看了一些源码后，就该真正去当炼丹师了。千万别满足于跑通别人的github代码，最好去参加一次Kaggle、天池、Biendata等平台的比赛，享受优化模型的摧残。

Kaggle的优点是有各种kernel可以学习，国内比赛的优点是中文数据方便看case。建议把两者的优点结合，比如参加一个国内的文本匹配比赛，就去kaggle找相同任务的kernel看，学习别人的trick。同时多看些顶会论文并复现，争取做完一个任务后就把这个任务技巧摸清。

# 各任务模型list汇总

**P.S. 对照文首脑图看效果更佳**

## <a id="para2cls"/>文本分类

<table border="0" cellpadding="0" cellspacing="0" width="540" style="border-collapse:
 collapse;table-layout:fixed;width:405pt">
 <colgroup><col class="xl65" width="95" style="mso-width-source:userset;mso-width-alt:3029;
 width:71pt">
 <col class="xl65" width="65" style="mso-width-source:userset;mso-width-alt:2090;
 width:49pt">
 <col class="xl65" width="168" style="mso-width-source:userset;mso-width-alt:5376;
 width:126pt">
 <col class="xl65" width="125" style="mso-width-source:userset;mso-width-alt:4010;
 width:94pt">
 <col class="xl65" width="87" style="width:65pt">
 </colgroup><tbody><tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" width="95" style="height:16.0pt;width:71pt">Model</td>
  <td class="xl66" width="65" style="border-left:none;width:49pt">Year</td>
  <td class="xl66" width="168" style="border-left:none;width:126pt">Method</td>
  <td class="xl66" width="125" style="border-left:none;width:94pt">Venue</td>
  <td class="xl66" width="87" style="border-left:none;width:65pt">Code</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="4" height="84" class="xl67" style="border-bottom:.5pt solid black;
  height:64.0pt;border-top:none">ReNN</td>
  <td class="xl66" style="border-top:none;border-left:none">2011</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="http://ai.stanford.edu/~ang/papers/emnlp11-RecursiveAutoencodersSentimentDistributions.pdf" target="_parent">RAE</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/vin00/Semi-Supervised-Recursive-Autoencoders-for-Predicting-Sentiment-Distributions" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2012</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://ai.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf" target="_parent">MV-RNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/github-pengge/MV_RNN" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2013</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf" target="_parent">RNTN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/pondruska/DeepSentiment" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2014</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://papers.nips.cc/paper/5275-global-belief-recursive-neural-networks.pdf" target="_parent">DeepRNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">NIPS</td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="2" height="42" class="xl67" style="border-bottom:.5pt solid black;
  height:32.0pt;border-top:none">MLP</td>
  <td class="xl66" style="border-top:none;border-left:none">2014</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="http://proceedings.mlr.press/v32/le14.html" target="_parent">Paragraph-Vec</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICML</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/inejc/paragraph-vectors" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2015</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://doi.org/10.3115/v1/p15-1162" target="_parent">DAN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/miyyer/dan" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="11" height="231" class="xl67" style="border-bottom:.5pt solid black;
  height:176.0pt;border-top:none">RNN</td>
  <td class="xl66" style="border-top:none;border-left:none">2015</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://doi.org/10.3115/v1/p15-1150" target="_parent">Tree-LSTM</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/stanfordnlp/treelstm" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2015</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="http://proceedings.mlr.press/v37/zhub15.pdf" target="_parent">S-LSTM</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICML</td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2015</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745" target="_parent">TextRCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">AAAI</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/roomylee/rcnn-text-classification" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2015</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.1430&amp;rep=rep1&amp;type=pdf" target="_parent">MT-LSTM</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/AlexAntn/MTLSTM" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2016</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.researchgate.net/publication/303521296_Adversarial_Training_Methods_for_Semi-Supervised_Text_Classification" target="_parent">oh-2LSTMp</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICML</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="http://riejohnson.com/cnn_20download.html" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2016</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.aclweb.org/anthology/C16-1329.pdf" target="_parent">BLSTM-2DCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">COLING</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/ManuelVs/NNForTextClassification" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2016</td>
  <td class="xl66" style="border-top:none;border-left:none">Multi-Task</td>
  <td class="xl66" style="border-top:none;border-left:none">IJCAI</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/baixl/text_classification" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2017</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.aclweb.org/anthology/D17-1169.pdf" target="_parent">DeepMoji</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/bfelbo/DeepMoji" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2017</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://openreview.net/forum?id=rJbbOLcex" target="_parent">TopicRNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICML</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/dangitstam/topic-rnn" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2017</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/abs/1605.07725" target="_parent">Miyato et al.</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICLR</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/tensorflow/models/tree/master/adversarial_text" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2018</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://link.springer.com/article/10.1007/s42979-020-0076-y" target="_parent">RNN-Capsule</a></td>
  <td class="xl66" style="border-top:none;border-left:none">TheWebConf</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/wangjiosw/Sentiment-Analysis-by-Capsules" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="10" height="210" class="xl67" style="border-bottom:.5pt solid black;
  height:160.0pt;border-top:none">CNN</td>
  <td class="xl66" style="border-top:none;border-left:none">2014</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.aclweb.org/anthology/D14-1181.pdf" target="_parent">TextCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2014</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://doi.org/10.3115/v1/p14-1062" target="_parent">DCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/kinimod23/ATS_Project" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2015</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification" target="_parent">CharCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">NIPS</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/mhjabreel/CharCNN" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2016</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/abs/1603.03827" target="_parent">SeqTextRCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">NAACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/ilimugur/short-text-classification" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2017</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf" target="_parent">XML-CNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">SIGIR</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/siddsax/XML-CNN" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2017</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://doi.org/10.18653/v1/P17-1052" target="_parent">DPCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/Cheneng/DPCNN" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2017</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.ijcai.org/Proceedings/2017/0406.pdf" target="_parent">KPCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">IJCAI</td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2018</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://doi.org/10.18653/v1/d18-1350" target="_parent">TextCapsule</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/andyweizhao/capsule_text_classification" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2018</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.aclweb.org/anthology/D18-1093.pdf" target="_parent">HFT-CNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/ShimShim46/HFT-CNN" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2020</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/abs/1908.06039v1" target="_parent">Bao et al.</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICLR</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/YujiaBao/Distributional-Signatures" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="11" height="231" class="xl67" style="border-bottom:.5pt solid black;
  height:176.0pt;border-top:none">Attention</td>
  <td class="xl66" style="border-top:none;border-left:none">2016</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://doi.org/10.18653/v1/n16-1174" target="_parent">HAN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">NAACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/richliao/textClassifier" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2016</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.aclweb.org/anthology/D16-1024.pdf" target="_parent">BI-Attention</a></td>
  <td class="xl66" style="border-top:none;border-left:none">NAACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/JRC1995/Abstractive-Summarization" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2016</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://doi.org/10.18653/v1/d16-1053" target="_parent">LSTMN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2017</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/abs/1703.03130" target="_parent">Lin et al.</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICLR</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/kaushalshetty/Structured-Self-Attention" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2018</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.aclweb.org/anthology/C18-1330/" target="_parent">SCM</a></td>
  <td class="xl66" style="border-top:none;border-left:none">COLING</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/lancopku/SGM" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2018</td>
  <td class="xl66" style="border-top:none;border-left:none">ELMo</td>
  <td class="xl66" style="border-top:none;border-left:none">NAACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/flairNLP/flair" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2018</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/abs/1804.00857" target="_parent">BiBloSA</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICLR</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/galsang/BiBloSA-pytorch" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/pdf/1811.01727v3.pdf" target="_parent">AttentionXML</a></td>
  <td class="xl66" style="border-top:none;border-left:none">NIPS</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/yourh/AttentionXML" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.aclweb.org/anthology/D19-1045/" target="_parent">HAPN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://gaotianyu1350.github.io/assets/aaai2019_hatt_paper.pdf" target="_parent">Proto-HATT</a></td>
  <td class="xl66" style="border-top:none;border-left:none">AAAI</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/thunlp/HATT-Proto" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/pdf/1902.08050.pdf" target="_parent">STCKA</a></td>
  <td class="xl66" style="border-top:none;border-left:none">AAAI</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/AIRobotZhang/STCKA" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="5" height="105" class="xl67" style="border-bottom:.5pt solid black;
  height:80.0pt;border-top:none">Transformer</td>
  <td class="xl66" style="border-top:none;border-left:none">2019</td>
  <td class="xl66" style="border-top:none;border-left:none">BERT</td>
  <td class="xl66" style="border-top:none;border-left:none">NAACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/google-research/bert" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl66" style="border-top:none;border-left:none">Sun et al.</td>
  <td class="xl66" style="border-top:none;border-left:none">CCL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/xuyige/BERT4doc-Classification" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl66" style="border-top:none;border-left:none">XLNet</td>
  <td class="xl66" style="border-top:none;border-left:none">NIPS</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/zihangdai/xlnet" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl66" style="border-top:none;border-left:none">RoBERTa</td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/pytorch/fairseq" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2020</td>
  <td class="xl66" style="border-top:none;border-left:none">ALBERT</td>
  <td class="xl66" style="border-top:none;border-left:none">ICLR</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/google-research/ALBERT" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="6" height="126" class="xl67" style="border-bottom:.5pt solid black;
  height:96.0pt;border-top:none">GNN</td>
  <td class="xl66" style="border-top:none;border-left:none">2018</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://dl.acm.org/doi/10.1145/3178876.3186005" target="_parent">DGCNN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">TheWebConf</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/HKUST-KnowComp/DeepGraphCNNforTexts" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4725" target="_parent">TextGCN</a></td>
  <td class="xl66" style="border-top:none;border-left:none">AAAI</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/yao8839836/text_gcn" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/pdf/1902.07153.pdf" target="_parent">SGC</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICML</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/Tiiiger/SGC" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://www.aclweb.org/anthology/D19-1345.pdf" target="_parent">Huang
  et al.</a></td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/LindgeW/TextLevelGNN" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/abs/1906.04898" target="_parent">Peng et al.</a></td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2020</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://arxiv.org/abs/2003.11644" target="_parent">MAGNET</a></td>
  <td class="xl66" style="border-top:none;border-left:none">ICAART</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/monk1337/MAGnet" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="3" height="63" class="xl67" style="border-bottom:.5pt solid black;
  height:48.0pt;border-top:none">Others</td>
  <td class="xl66" style="border-top:none;border-left:none">2017</td>
  <td class="xl66" style="border-top:none;border-left:none">Miyato et al.</td>
  <td class="xl66" style="border-top:none;border-left:none">ICLR</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/TobiasLee/Text-Classification" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2018</td>
  <td class="xl66" style="border-top:none;border-left:none">TMN</td>
  <td class="xl66" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl66" style="border-top:none;border-left:none">　</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl66" style="height:16.0pt;border-top:none;border-left:
  none">2019</td>
  <td class="xl66" style="border-top:none;border-left:none">Zhang et al.</td>
  <td class="xl66" style="border-top:none;border-left:none">NAACL</td>
  <td class="xl70" style="border-top:none;border-left:none"><a href="https://github.com/JingqingZ/KG4ZeroShotText" target="_parent">link</a></td>
 </tr>
 <!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="95" style="width:71pt"></td>
  <td width="65" style="width:49pt"></td>
  <td width="168" style="width:126pt"></td>
  <td width="125" style="width:94pt"></td>
  <td width="87" style="width:65pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>

## <a id="para2sts"/>  文本匹配

<table border="0" cellpadding="0" cellspacing="0" width="505" style="border-collapse:
 collapse;table-layout:fixed;width:378pt">
 <colgroup><col width="87" span="2" style="width:65pt">
 <col width="157" style="mso-width-source:userset;mso-width-alt:5034;width:118pt">
 <col width="87" span="2" style="width:65pt">
 </colgroup><tbody><tr height="21" style="height:16.0pt">
  <td height="21" width="87" style="height:16.0pt;width:65pt">Structure</td>
  <td width="87" style="width:65pt">Year</td>
  <td width="157" style="width:118pt">Model</td>
  <td width="87" style="width:65pt">Venue</td>
  <td width="87" style="width:65pt">Ref</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="13" height="273" class="xl67" style="height:208.0pt">Siamese</td>
  <td align="right">2013</td>
  <td>DSSM</td>
  <td>CIKM</td>
  <td class="xl66"><a href="https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2015</td>
  <td>SiamCNN</td>
  <td>ASRU</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1508.01585" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2015</td>
  <td>Skip-Thought</td>
  <td>NIPS</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1506.06726" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2016</td>
  <td>Multi-View</td>
  <td>EMNLP</td>
  <td class="xl66"><a href="https://www.aclweb.org/anthology/D16-1036.pdf" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2016</td>
  <td>FastSent</td>
  <td>ACL</td>
  <td class="xl66"><a href="https://www.aclweb.org/anthology/N16-1162.pdf" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2016</td>
  <td>SiamLSTM</td>
  <td>AAAI</td>
  <td class="xl66"><a href="https://dl.acm.org/doi/10.5555/3016100.3016291" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2017</td>
  <td>Joint-Many</td>
  <td>EMNLP</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1611.01587" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2017</td>
  <td>InferSent</td>
  <td>EMNLP</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1705.02364" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2017</td>
  <td>SSE</td>
  <td>EMNLP</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1708.02312" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2018</td>
  <td>GenSen</td>
  <td>ICLR</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1804.00079" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2018</td>
  <td>USE</td>
  <td>ACL</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1803.11175v2" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td>Sentence-BERT</td>
  <td>EMNLP</td>
  <td class="xl66"><a href="https://www.aclweb.org/anthology/D19-1410.pdf" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2020</td>
  <td>BERT-flow</td>
  <td>EMNLP</td>
  <td class="xl66"><a href="https://arxiv.org/abs/2011.05864" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td rowspan="6" height="126" class="xl67" style="height:96.0pt">Interaction</td>
  <td align="right">2016</td>
  <td>DecAtt</td>
  <td>EMNLP</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1606.01933" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2016</td>
  <td>PWIM</td>
  <td>ACL</td>
  <td class="xl66"><a href="https://www.aclweb.org/anthology/N16-1108.pdf" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2017</td>
  <td>ESIM</td>
  <td>ACL</td>
  <td class="xl66"><a href="https://www.aclweb.org/anthology/P17-1152.pdf" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2018</td>
  <td>DIIN</td>
  <td>ICLR</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1709.04348" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td>HCAN</td>
  <td>EMNLP</td>
  <td class="xl66"><a href="https://cs.uwaterloo.ca/~jimmylin/publications/Rao_etal_EMNLP2019.pdf" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td>RE2</td>
  <td>ACL</td>
  <td class="xl66"><a href="https://www.aclweb.org/anthology/P19-1465/" target="_parent">link</a></td>
 </tr>
 <!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="87" style="width:65pt"></td>
  <td width="87" style="width:65pt"></td>
  <td width="157" style="width:118pt"></td>
  <td width="87" style="width:65pt"></td>
  <td width="87" style="width:65pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>

## <a id="para2sl"/> 序列标注

<table border="0" cellpadding="0" cellspacing="0" width="1315" style="border-collapse:
 collapse;table-layout:fixed;width:985pt">
 <colgroup><col class="xl67" width="87" span="3" style="width:65pt">
 <col class="xl67" width="133" style="mso-width-source:userset;mso-width-alt:4266;
 width:100pt">
 <col class="xl67" width="192" style="mso-width-source:userset;mso-width-alt:6144;
 width:144pt">
 <col class="xl67" width="147" style="mso-width-source:userset;mso-width-alt:4693;
 width:110pt">
 <col class="xl67" width="192" style="mso-width-source:userset;mso-width-alt:6144;
 width:144pt">
 <col class="xl67" width="187" style="mso-width-source:userset;mso-width-alt:5973;
 width:140pt">
 <col class="xl67" width="203" style="mso-width-source:userset;mso-width-alt:6485;
 width:152pt">
 </colgroup><tbody><tr height="21" style="height:16.0pt">
  <td rowspan="2" height="42" class="xl68" width="87" style="height:32.0pt;width:65pt">Ref</td>
  <td rowspan="2" class="xl69" width="87" style="border-bottom:.5pt solid black;
  width:65pt">Year</td>
  <td rowspan="2" class="xl69" width="87" style="border-bottom:.5pt solid black;
  width:65pt">Venue</td>
  <td colspan="3" class="xl68" width="472" style="border-left:none;width:354pt">Embedding
  Module</td>
  <td rowspan="2" class="xl68" width="192" style="width:144pt">Context Encoder</td>
  <td rowspan="2" class="xl68" width="187" style="width:140pt">Inference Module</td>
  <td rowspan="2" class="xl68" width="203" style="width:152pt">Tasks</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl68" style="height:16.0pt;border-top:none;border-left:
  none">external input</td>
  <td class="xl68" style="border-top:none;border-left:none">word embedding</td>
  <td class="xl68" style="border-top:none;border-left:none">character-level</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/abs/1603.01354" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2016</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl68" style="border-top:none;border-left:none">CNN</td>
  <td class="xl68" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">POS, NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/abs/1805.08237" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl68" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">Softmax</td>
  <td class="xl68" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/N18-1089.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">NAACL</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/pdf/1709.04109.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">AAAI</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM+LM</td>
  <td class="xl68" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/abs/1604.05529" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2016</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Polyglot</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/P17-1194.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">Bi-LSTM+LM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/abs/1705.00108" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Senna</td>
  <td class="xl68" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM+pre LM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="http://alanakbik.github.io/papers/coling2018.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">COLING</td>
  <td class="xl68" style="border-top:none;border-left:none">Pre LM emb</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.ijcai.org/Proceedings/2018/0637.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">IJCAI</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">LSTM+Softmax</td>
  <td class="xl68" style="border-top:none;border-left:none">POS, NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/P18-2038.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM+LM</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF+Semi-CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/C18-1061.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">COLING</td>
  <td class="xl68" style="border-top:none;border-left:none">Spelling, gaz</td>
  <td class="xl68" style="border-top:none;border-left:none">Senna</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">Mo-BiLSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">Softmax</td>
  <td class="xl68" style="border-top:none;border-left:none">NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/P18-2012.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Parallel Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">Softmax</td>
  <td class="xl68" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="http://www.cs.cmu.edu/~./wcohen/postscript/iclr-2017-transfer.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">ICLR</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Senna, Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-GRU</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-GRU</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2015</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Trained on wikipedia</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Softmax</td>
  <td class="xl72" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/Q16-1026.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2016</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">Cap, lexicon</td>
  <td class="xl68" style="border-top:none;border-left:none">Senna</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl68" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/C16-1030.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2016</td>
  <td class="xl68" style="border-top:none;border-left:none">COLING</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/D18-1279/" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">InNet</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/P17-2027/" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">ACL</td>
  <td class="xl68" style="border-top:none;border-left:none">Spelling, gaz</td>
  <td class="xl68" style="border-top:none;border-left:none">Senna</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">INN</td>
  <td class="xl68" style="border-top:none;border-left:none">Softmax</td>
  <td class="xl72" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/abs/1809.10835" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">EL-CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">Citation field
  extraction</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/D16-1082.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2016</td>
  <td class="xl68" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Trained with
  skip-gram</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Skip-chain CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">Clinical entities
  detection</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/D18-1310.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">Word shapes, gaz</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2011</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">Cap, gaz</td>
  <td class="xl68" style="border-top:none;border-left:none">Senna</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">POS, NER, chunking,
  SRL</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="http://www.cips-cl.org/static/anthology/CCL-2017/CCL-17-071.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">CCL</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Gated-CNN</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/abs/1702.02098" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">ID-CNN</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/abs/1603.01360" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2016</td>
  <td class="xl68" style="border-top:none;border-left:none">NAACL</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/pdf/1508.01991v1.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2015</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">Spelling, gaz</td>
  <td class="xl68" style="border-top:none;border-left:none">Senna</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="http://proceedings.mlr.press/v32/santos14.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2014</td>
  <td class="xl68" style="border-top:none;border-left:none">ICML</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl68" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/pdf/1701.04027.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">AAAI</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Senna</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Pointer network</td>
  <td class="xl72" style="border-top:none;border-left:none">Chunking, slot
  filling</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/P17-1113.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Entity relation
  extraction</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/C18-1161.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">LS vector, cap</td>
  <td class="xl68" style="border-top:none;border-left:none">SSKIP</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/abs/1707.05928" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">ICLR</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.ijcai.org/Proceedings/2018/0579.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">IJCAI</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-GRU</td>
  <td class="xl72" style="border-top:none;border-left:none">Pointer network</td>
  <td class="xl72" style="border-top:none;border-left:none">Text segmentation</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/D17-1256.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">EMNLP</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Softmax</td>
  <td class="xl72" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://nlp.stanford.edu/pubs/dozat2017stanford.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">CoNLL</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Word2vec, Fasttext</td>
  <td class="xl72" style="border-top:none;border-left:none">LSTM+Attention</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Softmax</td>
  <td class="xl72" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ic19-NCRFT.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2019</td>
  <td class="xl68" style="border-top:none;border-left:none">ICASSP</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">NCRF transducers</td>
  <td class="xl72" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/W18-3401.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2018</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">\</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM+AE</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Softmax</td>
  <td class="xl72" style="border-top:none;border-left:none">POS</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.aclweb.org/anthology/I17-2017.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2017</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">Lexicons</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM</td>
  <td class="xl72" style="border-top:none;border-left:none">Segment-level CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://arxiv.org/pdf/1907.05611v2.pdf" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2019</td>
  <td class="xl68" style="border-top:none;border-left:none">AAAI</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">GRN+CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">NER</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" class="xl71" style="height:16.0pt;border-top:none"><a href="https://www.sciencedirect.com/science/article/pii/S0031320320304398" target="_parent">link</a></td>
  <td class="xl68" style="border-top:none;border-left:none">2020</td>
  <td class="xl68" style="border-top:none;border-left:none">　</td>
  <td class="xl72" style="border-top:none;border-left:none">\</td>
  <td class="xl68" style="border-top:none;border-left:none">Glove</td>
  <td class="xl72" style="border-top:none;border-left:none">CNN</td>
  <td class="xl72" style="border-top:none;border-left:none">Bi-LSTM+SA</td>
  <td class="xl72" style="border-top:none;border-left:none">CRF</td>
  <td class="xl72" style="border-top:none;border-left:none">POS, NER, chunking</td>
 </tr>
 <!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="87" style="width:65pt"></td>
  <td width="87" style="width:65pt"></td>
  <td width="87" style="width:65pt"></td>
  <td width="133" style="width:100pt"></td>
  <td width="192" style="width:144pt"></td>
  <td width="147" style="width:110pt"></td>
  <td width="192" style="width:144pt"></td>
  <td width="187" style="width:140pt"></td>
  <td width="203" style="width:152pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>

## <a id="para2seq2seq"/> 文本生成

## <a id="para2lm"/> 语言模型

<table border="0" cellpadding="0" cellspacing="0" width="350" style="border-collapse:
 collapse;table-layout:fixed;width:262pt">
 <colgroup><col width="87" style="width:65pt">
 <col width="176" style="mso-width-source:userset;mso-width-alt:5632;width:132pt">
 <col width="87" style="width:65pt">
 </colgroup><tbody><tr height="21" style="height:16.0pt">
  <td height="21" width="87" style="height:16.0pt;width:65pt">Year</td>
  <td width="176" style="width:132pt">Model</td>
  <td width="87" style="width:65pt">Code</td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2018</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1810.04805v2" target="_parent">BERT</a></td>
  <td class="xl66"><a href="https://github.com/google-research/bert" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1906.08101" target="_parent">WWM</a></td>
  <td class="xl66"><a href="https://github.com/ymcui/Chinese-BERT-wwm" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://www.jiqizhixin.com/articles/2019-03-16-3" target="_parent">Baidu ERNIE1.0</a></td>
  <td class="xl66"><a href="https://github.com/PaddlePaddle/ERNIE" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://www.jiqizhixin.com/articles/2019-07-31-10" target="_parent">Baidu ERNIE2.0</a></td>
  <td class="xl66"><a href="https://github.com/PaddlePaddle/ERNIE" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1907.10529" target="_parent">SpanBERT</a></td>
  <td class="xl66"><a href="https://github.com/facebookresearch/SpanBERT" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1907.11692" target="_parent">RoBERTa</a></td>
  <td class="xl66"><a href="https://github.com/huggingface/transformers" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1906.08237" target="_parent">XLNet</a></td>
  <td class="xl66"><a href="https://github.com/zihangdai/xlnet" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1908.04577" target="_parent">StructBERT</a></td>
  <td></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://openreview.net/pdf?id=r1xMH1BtvB" target="_parent">ELECTRA</a></td>
  <td class="xl66"><a href="https://github.com/google-research/electra" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2019</td>
  <td class="xl66"><a href="https://arxiv.org/abs/1909.11942" target="_parent">ALBERT</a></td>
  <td class="xl66"><a href="https://github.com/google-research/albert" target="_parent">link</a></td>
 </tr>
 <tr height="21" style="height:16.0pt">
  <td height="21" align="right" style="height:16.0pt">2020</td>
  <td class="xl66"><a href="https://arxiv.org/abs/2006.03654" target="_parent">DeBERTa</a></td>
  <td class="xl66"><a href="https://github.com/microsoft/DeBERTa" target="_parent">link</a></td>
 </tr>
 <!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="87" style="width:65pt"></td>
  <td width="176" style="width:132pt"></td>
  <td width="87" style="width:65pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>

# 各任务综述

## <a id="review_textcls"/> 文本分类

### Fasttext

```
论文：https://arxiv.org/abs/1607.01759
代码：https://github.com/facebookresearch/fastText
```

Fasttext是Facebook推出的一个便捷的工具，包含文本分类和词向量训练两个功能。

Fasttext的分类实现很简单：把输入转化为词向量，取平均，再经过线性分类器得到类别。输入的词向量可以是预先训练好的，也可以随机初始化，跟着分类任务一起训练。

![](https://tva1.sinaimg.cn/large/0081Kckwly1gmclieqa1pj30fg07ydg5.jpg)

Fasttext直到现在还被不少人使用，主要有以下优点：

1. 模型本身复杂度低，但效果不错，能快速产生任务的baseline
2. Facebook使用C++进行实现，进一步提升了计算效率
3. 采用了char-level的n-gram作为附加特征，比如paper的trigram是 [pap, ape, per]，在将输入paper转为向量的同时也会把trigram转为向量一起参与计算。这样一方面解决了长尾词的OOV (out-of-vocabulary)问题，一方面利用n-gram特征提升了表现
4. 当类别过多时，支持采用hierarchical softmax进行分类，提升效率

对于文本长且对速度要求高的场景，Fasttext是baseline首选。同时用它在无监督语料上训练词向量，进行文本表示也不错。不过想继续提升效果还需要更复杂的模型。

### TextCNN

```
论文：https://arxiv.org/abs/1408.5882
代码：https://github.com/yoonkim/CNN_sentence
```

TextCNN是Yoon Kim小哥在2014年提出的模型，开创了用CNN编码n-gram特征的先河。

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmynmvpkdsj314o0hqn09.jpg)

模型结构如图，图像中的卷积都是二维的，而TextCNN则使用「一维卷积」，即`filter_size * embedding_dim`，有一个维度和embedding相等。这样filter_size就能抽取n-gram的信息。以1个样本为例，整体的前向逻辑是：

1. 对词进行embedding，得到`[seq_length, embedding_dim] `
2. 用N个卷积核，得到N个`seq_length-filter_size+1`长度的一维feature map
3. 对feature map进行max-pooling（因为是时间维度的，也称max-over-time pooling），得到N个`1x1`的数值，拼接成一个N维向量，作为文本的句子表示
4. 将N维向量压缩到类目个数的维度，过Softmax

在TextCNN的实践中，有很多地方可以优化（参考这篇论文[](https://arxiv.org/pdf/1510.03820.pdf "A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification")）：

1. Filter尺寸：这个参数决定了抽取n-gram特征的长度，这个参数主要跟数据有关，平均长度在50以内的话，用10以下就可以了，否则可以长一些。在调参时可以先用一个尺寸grid search，找到一个最优尺寸，然后尝试最优尺寸和附近尺寸的组合
2. Filter个数：这个参数会影响最终特征的维度，维度太大的话训练速度就会变慢。这里在100-600之间调参即可
3. CNN的激活函数：可以尝试Identity、ReLU、tanh
4. 正则化：指对CNN参数的正则化，可以使用dropout或L2，但能起的作用很小，可以试下小的dropout率(<0.5)，L2限制大一点
5. Pooling方法：根据情况选择mean、max、k-max pooling，大部分时候max表现就很好，因为分类任务对细粒度语义的要求不高，只抓住最大特征就好了
6. Embedding表：中文可以选择char或word级别的输入，也可以两种都用，会提升些效果。如果训练数据充足（10w+），也可以从头训练
7. 蒸馏BERT的logits，利用领域内无监督数据
8. 加深全连接：原论文只使用了一层全连接，而加到3、4层左右效果会更好[](https://www.zhihu.com/question/270245936 "卷积层和分类层，哪个更重要？")

TextCNN是很适合中短文本场景的强baseline，但不太适合长文本，因为卷积核尺寸通常不会设很大，无法捕获长距离特征。同时max-pooling也存在局限，会丢掉一些有用特征。另外再仔细想的话，TextCNN和传统的n-gram词袋模型本质是一样的，它的好效果很大部分来自于词向量的引入[](https://zhuanlan.zhihu.com/p/35457093 "从经典文本分类模型TextCNN到深度模型DPCNN")，解决了词袋模型的稀疏性问题。

### DPCNN

```
论文：https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
代码：https://github.com/649453932/Chinese-Text-Classification-Pytorch
```

上面介绍TextCNN有太浅和长距离依赖的问题，那直接多怼几层CNN是否可以呢？感兴趣的同学可以试试，就会发现事情没想象的那么简单。直到2017年，腾讯才提出了把TextCNN做到更深的DPCNN模型：

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmxgtn8qqvj30np0e1dio.jpg)

上图中的ShallowCNN指TextCNN。DPCNN的核心改进如下：

1. 在Region embedding时不采用CNN那样加权卷积的做法，而是**对n个词进行pooling后再加个1x1的卷积**，因为实验下来效果差不多，且作者认为前者的表示能力更强，容易过拟合
2. 使用1/2池化层，用size=3 stride=2的卷积核，直接**让模型可编码的sequence长度翻倍**（自己在纸上画一下就get啦）
3. 残差链接，参考ResNet，减缓梯度弥散问题

凭借以上一些精妙的改进，DPCNN相比TextCNN有1-2个百分点的提升。

### TextRCNN

```
论文：https://dl.acm.org/doi/10.5555/2886521.2886636
代码：https://github.com/649453932/Chinese-Text-Classification-Pytorch
```

除了DPCNN那样增加感受野的方式，RNN也可以缓解长距离依赖的问题。下面介绍一篇经典TextRCNN。

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmyp0iphcmj30ok0a90v2.jpg)

模型的前向过程是：

1. 得到单词 i 的表示 $e(w_i)$
2. 通过RNN得到左右双向的表示 $c_l(w_i)$ 和 $c_r(w_i)$
3. 将表示拼接得到 $x_i = [c_l(w_i);e(w_i);c_r(w_i)]$ ，再经过变换得到 $y_i=tanh(Wx_i+b)$
4. 对多个 $y_i$ 进行 max-pooling，得到句子表示 $y$，在做最终的分类

这里的convolutional是指max-pooling。通过加入RNN，比纯CNN提升了1-2个百分点。

### TextBiLSTM+Attention

```
论文：https://www.aclweb.org/anthology/P16-2034.pdf
代码：https://github.com/649453932/Chinese-Text-Classification-Pytorch
```

从前面介绍的几种方法，可以自然地得到文本分类的框架，就是**先基于上下文对token编码，然后pooling出句子表示再分类**。在最终池化时，max-pooling通常表现更好，因为文本分类经常是主题上的分类，从句子中一两个主要的词就可以得到结论，其他大多是噪声，对分类没有意义。而到更细粒度的分析时，max-pooling可能又把有用的特征去掉了，这时便可以用attention进行句子表示的融合：

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmypj7s4ldj31260ioq7b.jpg)

BiLSTM就不解释了，要注意的是，计算attention score时会先进行变换：
$$
M = tanh(H)
$$

$$
\alpha = softmax(w^TM)
$$

$$
r = H\alpha^T
$$

其中 $w$ 是context vector，随机初始化并随着训练更新。最后得到句子表示 $r$，再进行分类。

这个加attention的套路用到CNN编码器之后代替pooling也是可以的，从实验结果来看attention的加入可以提高2个点。如果是情感分析这种由句子整体决定分类结果的任务首选RNN。

### HAN

```
论文：https://www.aclweb.org/anthology/N16-1174.pdf
代码：https://github.com/richliao/textClassifier
```

上文都是句子级别的分类，虽然用到长文本、篇章级也是可以的，但速度精度都会下降，于是有研究者提出了层次注意力分类框架，即Hierarchical Attention。先对每个句子用 BiGRU+Att 编码得到句向量，再对句向量用 BiGRU+Att 得到doc级别的表示进行分类：

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmz2zjmfglj30me0pgtb7.jpg)

方法很符合直觉，不过实验结果来看比起avg、max池化只高了不到1个点（狗头，真要是很大的doc分类，好好清洗下，fasttext其实也能顶的（捂脸。

### BERT

BERT的原理代码就不用放了叭～

BERT分类的优化可以尝试：

1. 多试试不同的预训练模型，比如RoBERT、WWM、ALBERT
2. 除了 [CLS] 外还可以用 avg、max 池化做句表示，甚至可以把不同层组合起来
3. 在领域数据上增量预训练
4. 集成蒸馏，训多个大模型集成起来后蒸馏到一个上
5. 先用多任务训，再迁移到自己的任务

### 其他模型

除了上述常用模型之外，还有Capsule Network[](https://kexue.fm/archives/4819 "揭开迷雾，来一顿美味的Capsule盛宴")、TextGCN[](https://arxiv.org/abs/1809.05679 "Graph Convolutional Networks for Text Classification")等红极一时的模型，因为涉及的背景知识较多，本文就暂不介绍了（嘻嘻）。

虽然实际的落地应用中比较少见，但在机器学习比赛中还是可以用的。Capsule Network被证明在多标签迁移的任务上性能远超CNN和LSTM[](https://zhuanlan.zhihu.com/p/35409788 "胶囊网络（Capsule Network）在文本分类中的探索")，但这方面的研究在18年以后就很少了。TextGCN则可以学到更多的global信息，用在半监督场景中，但碰到较长的需要序列信息的文本表现就会差些[](https://www.zhihu.com/question/307086081/answer/717456124 "怎么看待最近比较火的 GNN？")。

### 技巧

模型说得差不多了，下面介绍一些自己的数据处理血泪经验，如有不同意见欢迎讨论～

#### 数据集构建

首先是**标签体系的构建**，拿到任务时自己先试标一两百条，看有多少是难确定（思考1s以上）的，如果占比太多，那这个任务的定义就有问题。可能是标签体系不清晰，或者是要分的类目太难了，这时候就要找项目owner去反馈而不是继续往下做。

其次是**训练评估集的构建**，可以构建两个评估集，一个是贴合真实数据分布的线上评估集，反映线上效果，另一个是用规则去重后均匀采样的随机评估集，反映模型的真实能力。训练集则尽可能和评估集分布一致，有时候我们会去相近的领域拿现成的有标注训练数据，这时就要注意调整分布，比如句子长度、标点、干净程度等，尽可能做到自己分不出这个句子是本任务的还是从别人那里借来的。

最后是**数据清洗**：

1. 去掉文本强pattern：比如做新闻主题分类，一些爬下来的数据中带有的XX报道、XX编辑高频字段就没有用，可以对语料的片段或词进行统计，把很高频的无用元素去掉。还有一些会明显影响模型的判断，比如之前我在判断句子是否为无意义的闲聊时，发现加个句号就会让样本由正转负，因为训练预料中的闲聊很少带句号（跟大家的打字习惯有关），于是去掉这个pattern就好了不少
2. 纠正标注错误：这个我真的屡试不爽，生生把自己从一个算法变成了标注人员。简单的说就是把训练集和评估集拼起来，用该数据集训练模型两三个epoch（防止过拟合），再去预测这个数据集，把模型判错的拿出来按 abs(label-prob) 排序，少的话就自己看，多的话就反馈给标注人员，把数据质量搞上去了提升好几个点都是可能的

#### 长文本

任务简单的话（比如新闻分类），直接用fasttext就可以达到不错的效果。

想要用BERT的话，最简单的方法是粗暴截断，比如只取句首+句尾、句首+tfidf筛几个词出来；或者每句都预测，最后对结果综合。

另外还有一些魔改的模型可以尝试，比如XLNet、Reformer、Longformer。

如果是离线任务且来得及的话还是建议跑全部，让我们相信模型的编码能力。

#### 少样本

自从用了BERT之后，很少受到数据不均衡或者过少的困扰，先无脑训一版。

如果样本在几百条，可以先把分类问题转化成匹配问题，或者用这种思想再去标一些高置信度的数据，或者用自监督、半监督的方法。

#### 鲁棒性

在实际的应用中，鲁棒性是个很重要的问题，否则在面对badcase时会很尴尬，怎么明明那样就分对了，加一个字就错了呢？

这里可以直接使用一些粗暴的数据增强，加停用词加标点、删词、同义词替换等，如果效果下降就把增强后的训练数据洗一下。

当然也可以用对抗学习、对比学习这样的高阶技巧来提升，一般可以提1个点左右，但不一定能避免上面那种尴尬的情况。

### 总结

文本分类是工业界最常用的任务，同时也是大多数NLPer入门做的第一个任务，我当年就是啥都不会，从训练到部署地实践了文本分类后就顺畅了。上文给出了不少模型，但实际任务中常用的也就那几个，下面是快速选型的建议：

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gn8atgmxrvj30u00ybtgz.jpg" width="50%">

实际上，落地时主要还是和数据的博弈。数据决定模型的上限，大多数人工标注的准确率达到95%以上就很好了，而文本分类通常会对准确率的要求更高一些，与其苦苦调参想fancy的结构，不如好好看看badcase，做一些数据增强提升模型鲁棒性更实用。
