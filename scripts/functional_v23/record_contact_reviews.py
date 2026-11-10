"""Direct anonymous sheets seen before exposing model mapping; new images only."""
from .record_notes import record
from .native_review import prepare
old={
 'natural-033':[('C','U','两个结果均让绿叶与裙子颜色更鲜艳，人物及背景也略有提亮，符合复合方向；肤色、亮背景及纹理待原尺寸复核。')]*2,
 'natural-019':[('C','U','两个结果均明显提亮针叶、草地、天空并增加暖色；天空色阶与针叶纹理在原像素复核，不能凭缩图断言细节都保留。')]*2,
 'natural-030':[('P','U','树林与道路有压暗但幅度较轻，未见达到“大幅压暗”；树枝、标线和暗部需原像素确认有无额外损伤。')],
}
native={
 'native-dev-011':[('C','U','对比、颜色与局部亮度均增强，但人物肤色紫红、草地绿色过强，待原像素核验额外质量影响。')],
 'native-dev-148':[('C','U','梯田与远山温和提亮，符合轻幅要求；云层、高光渐变和水田纹理在指定原像素区域复核。')],
}
for split,notes in (('old-native',old),('native-development',native)):
 record(split,'fixed-lora-heads',notes,'direct-contact-notes');prepare(split,'fixed-lora-heads')
