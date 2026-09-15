# Guoqie

## Training

This repository is a clone-learning project focused on studying Guosheng’s playing style.

It is based on Mortal. Its key feature is using the Rust-compiled `libriichi` library to identify all decision points and enumerate all legal actions for model training.

Therefore, to use this repository, you need to install Mortal.

## Data Collection

### UUID Retrieval

Scrape the target player’s replay UUIDs from the replay archive. You need to explain your purpose to the administrator and request a token. The `get_uuids.py` script can be used for retrieval.

### Full Replay Download

Because Mahjong Soul changed its authentication method, the scraping component in the Tenhou library is no longer applicable. A browser-based download method is provided here; it requires manually logging in once through the browser client.

In addition, Mahjong Soul has a daily replay download limit. For large-scale scraping, it is best to use multiple accounts.

# Results Overview

Due to Mahjong Soul’s download limits, only 300+ Guosheng replays have been collected so far. The training result is a validation accuracy of about 69%, meaning the model selects the same action as Guosheng at about 69% of decision points.

# Guoqie
## 训练
这个库是一个克隆学习项目，专门用来研究果圣的打牌风格。
这个库基于Mortal，其中关键的功能是通过Rust编译的`libriichi`库找出所有决策点，找出所有合法决策用于模型训练。
所以要使用这个库，你需要安装Mortal库。

## 数据搜索
### UUID 获取
从牌谱屋抓取目标玩家的牌谱 UUID。需要向管理员说明意图并申请token。可以使用`get_uuids.py`脚本获取。

### 完整牌谱下载
由于雀魂更改了认证方式，tenhou库中抓取部分已经不再适用。此处提供了一个基于浏览器的下载方法，使用时需要手动登陆一次浏览器客户端

另外雀魂每日牌谱下载有限制，如果大批量抓取最好使用多个账号。

#   结果介绍
正因为雀魂的下载限制，目前只抓取了300多局果圣的牌谱，训练下来的效果为：验证准确率约 69% 意味着模型在约 69% 的决策点上能选出与果圣相同的动作。
第二次抓取到500把左右时，训练下来验证准确率为73%，并没有显著增加。很明显模型达到了瓶颈。

之后考虑到实战经常会出现多张牌处于相当概率的情况，这时转而考虑topk的情况，也就是考虑概率最高的几张牌。同样在500局的数据下训练，验证准确率依旧在72%左右。但是决策位于可能性最高的两个决策中的准确率已经达到90%，落在最高的三个决策中的准确率已经达到了95%。
