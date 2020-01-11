# pgn_for_baidu_competition

this repository is for one summarization and inference competition launched by baidu.

details can be found from：    https://aistudio.baidu.com/aistudio/competition/detail/3.

I use the pointer-generator-network and get the score(ROUGH_L) about 37. 


![image](https:https://github.com/quanterk/pgn_for_baidu_competition/blob/master/images/pgn.png)



It is not a very good result, cause somebody reached about 50 and even 70. In the case I try to pretrain the word vector, but the result is behind the one which does not pretrain,It's so strange.

I encounter the problem that the loss will change to be NAN after thousands steps, which is about 10 epoachs. It's hard for me to find the reason. Maybe I will fix this problem in the future.

In the future， I will try different approaches to solve this competition, and try my best the get higher scores.

In the last, I refer some codes and ideas from Luojie and Jiangxingfa.

If you are interested about this competiton or you have great interest in summarizition and NLP related task, do not hesitate to contact me by 219040014@link.cuhk.edu.cn



