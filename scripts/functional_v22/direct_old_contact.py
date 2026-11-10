from functional_v22.common import EXP
from functional_v22 import record_notes as rn, native_review as nr
from functional_v1.common import write,sha,stamp
rn.EXP=EXP; nr.EXP=EXP
notes={
'natural-028':[('C','U','人物面部及白衬衫比原图更亮，方向符合整体提亮。联系表未见明显结构改变；皮肤、衣物高光及细节质量需原像素复核。')]*2,
'natural-033':[('C','U','三个结果均提高了植被颜色鲜明程度，人物及暗部也略有提亮，符合更鲜艳更亮的复合要求。细叶、皮肤及高亮空隙需原像素核验。')]*3,
'natural-019':[('C','U','四个结果均明显提亮了前景和天空并变暖，复合方向已实现。不同结果的黄绿色覆盖、天空渐变及树叶细节需原像素复核后判断基本质量。')]*4,
'natural-022':[('C','U','两个结果的岩石、植被颜色均比原图更鲜明，云层和构图保留；岩层、云纹细节是否损失待原像素核验。')]*2,
'natural-030':[('C','U','两个结果均明显压暗树林道路，仍能辨认弯路和黄色标线，达到明显变暗要求；树枝和阴影细节质量待原像素复核。')]*2,
}
rn.record('old-native','fixed-quality-heads',notes,'direct-contact-notes')
write(EXP/'review-helper-binding.json',dict(at=stamp(),frozen_helpers_retained=True,issue='Two copied auxiliary review helpers retain an R21 output-directory literal. Explicitly binding EXP to R22 before their function calls prevents writes to prior sealed observations. Model/training/editor/protocol unchanged.',overrides={'functional_v22.record_notes.EXP':str(EXP),'functional_v22.native_review.EXP':str(EXP)},source_sha256=sha(__file__)))
nr.prepare('old-native','fixed-quality-heads')
