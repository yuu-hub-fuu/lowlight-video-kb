"""第二版的场景级数据划分（全部来自 SDSD indoor_np；同一场景的帧只出现在一个集合中）。

训练 34 个场景，均来自作者的训练场景池：
  pair1–6（各30帧）、pair14,16,17,18,19,20,22,24（各40帧）、
  pair25–30,32,34–46（各12帧，场景多样性比帧数更重要）
验证 pair7–12（各30帧）：早停、选择融合/平滑系数
测试 作者官方室内测试场景 12 个（各30帧）：参数冻结后只评测一次
"""
TRAIN = {**{f"pair{i}": 30 for i in range(1, 7)},
         **{f"pair{i}": 40 for i in [14, 16, 17, 18, 19, 20, 22, 24]},
         **{f"pair{i}": 12 for i in [25, 26, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]}}
VAL = [f"pair{i}" for i in range(7, 13)]
TEST = [f"pair{i}" for i in [13, 15, 21, 23, 31, 33, 50, 52, 58, 60, 68, 70]]

assert not (set(TRAIN) & set(VAL)) and not (set(TRAIN) & set(TEST)) and not (set(VAL) & set(TEST))
