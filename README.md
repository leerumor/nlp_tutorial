# NLP学习指南

本教程致力于帮助同学们快速入门NLP，并掌握各个任务的SOTA模型。

1. [系统入门方法](#para1)
2. 各任务模型list汇总（doing）：[文本分类](#para2cls)、文本匹配、[序列标注](#para2sl)、文本生成、语言模型
3. 文本分类综述&代码&技巧
4. 文本匹配综述&代码&技巧
5. 序列标注综述&代码&技巧
6. 文本生成综述&代码&技巧
7. 语言模型综述&代码&技巧


# <a id="para1"/>如何系统地入门

机器学习是一门既重理论又重实践的学科，想一口吃下这个老虎是不可能的，因此学习应该是个**循环且逐渐细化**的过程。

首先要有个全局印象，知道minimum的情况下要学哪些知识点：

![](https://tva1.sinaimg.cn/large/0081Kckwly1gly7vmtma6j30n30cnjt8.jpg)

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

## 文本匹配

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

## 文本生成

## 语言模型
