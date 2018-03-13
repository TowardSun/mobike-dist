# Mobike Distribution Inference (TKDE)

1. GeoConv Model，从source city中切分validation set，当设计更为深层的CNN模型时，比如两层CNN Model：
    * validation的performance可以有很大的提高，而target city的performance则下降很明显；
    * 证明还是需要端到端的迁移学习方法，或者尝试TCA来进行降维；