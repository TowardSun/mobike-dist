# Mobike Distribution Inference (TKDE)

1. GeoConv Model，从source city中切分validation set，当设计更为深层的CNN模型时，比如两层CNN Model：
    * validation的performance可以有很大的提高，而target city的performance则下降很明显；
    * 证明还是需要端到端的迁移学习方法，或者尝试TCA来进行降维；

2. Grid search log:

    ```
    liuzhao+  57164  154  1.4 21881468 1188236 pts/15 Sl 19:27   1:01 python main.py --train_cities bj --test_cities nb --model_choice 0 --y_scale --epochs 200
    liuzhao+  57166  111  1.5 22046136 1305876 pts/15 Rl 19:27   0:44 python main.py --train_cities bj --test_cities nb --model_choice 1 --y_scale --epochs 200
    liuzhao+  57170  116  1.4 21908036 1231812 pts/15 Sl 19:27   0:46 python main.py --train_cities sh --test_cities nb --model_choice 1 --y_scale --epochs 200 --gpu 1
    liuzhao+  57171  116  1.5 21908036 1236536 pts/15 Sl 19:27   0:46 python main.py --train_cities sh --test_cities nb --model_choice 1 --epochs 200 --gpu 1
    ```

3. 确定实验思路：
    
    * 对于target，采用原始数值进行回归即可，不需要进行min-max或者std scale；
    * 普通的CNN网络性能是好于dense net的（也有可能是dense net的网络结构过于复杂）；