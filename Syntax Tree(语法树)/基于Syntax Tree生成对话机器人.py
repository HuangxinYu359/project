import random

grammar = """
战斗 => 施法  ， 结果 。
施法 => 主语 动作 技能 
结果 => 主语 获得 效果
主语 => 张飞 | 关羽 | 赵云 | 典韦 | 许褚 | 刘备 | 黄忠 | 曹操 | 鲁班七号 | 貂蝉
动作 => 施放 | 使用 | 召唤 
技能 => 一骑当千 | 单刀赴会 | 青龙偃月 | 刀锋铁骑 | 黑暗潜能 | 画地为牢 | 守护机关 | 狂兽血性 | 龙鸣 | 惊雷之龙 | 破云之龙 | 天翔之龙
获得 => 损失 | 获得 
效果 => 数值 状态
数值 => 1 | 1000 |5000 | 100 
状态 => 法力 | 生命
"""
