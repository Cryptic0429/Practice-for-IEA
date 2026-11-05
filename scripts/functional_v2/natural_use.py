"""Supplementary original-request probes, not additional training or independent photos."""
from .common import ROOT,DATA,EXP,read,write,sha,stamp

def main():
    assert not (DATA/'natural-use-final-cases.json').exists()
    sources={s['id']:s for s in read(DATA/'source-splits.json')['sources']}
    specs=[(28,'请提亮这张照片。','lighting','整体可见提亮，保留肤色和白衫层次。'),
           (33,'让颜色鲜艳一点，同时把照片提亮一些。','compound','同时略增艳和提亮，保留肤色、亮背景和植被细节。'),
           (55,'稍微增加一点色彩。','saturation','轻微增艳，不能把肤色、黄衣过度增强。'),
           (19,'希望照片明显更亮，整体色彩也更暖一些。','compound','同时明显提亮和暖色调整，保留夕阳天空层次。'),
           (22,'让照片的颜色更丰富、更鲜明。','saturation','整体增艳，避免灌木与岩石变成非自然色。'),
           (30,'请把照片明显调暗。','lighting','整体明显调暗，保留道路及树木可辨层次。')]
    cases=[]
    for n,zh,family,expected in specs:
        source=sources[f'development-{n:03d}'];image=DATA/'inputs'/source['id']/'original.png'
        assert source['split']=='development' and source['original_request'].strip()
        cases.append({'id':f'natural-{n:03d}','source_id':source['id'],'group_id':source['group_id'],'domain':source['domain'],
          'split':'natural-use','image':str(image.relative_to(ROOT)),'input_sha256':sha(image),
          'instructions':{'en':source['original_request'],'zh':zh},'family':family,'action':'original-natural-request',
          'expected_change':expected,'must_preserve':'保留主体、肤色、天空和景物结构及主要明暗细节；避免额外偏色、过曝或压黑。',
          'failure_conditions':['指令漏项或方向错误','明显过调、非预期偏色或细节损失'],
          'instruction_origin':'English: untouched original GIER request; Chinese: AI translation, not a separate sample.',
          'original_source_sha256':source['source_sha256'],'evaluation_role':'Supplementary regression probes only; 6 already-used development photos, no checkpoint ranking, not independent user trial feedback.'})
    write(DATA/'natural-use-final-cases.json',cases)
    protocol=read(EXP/'protocol.json');protocol['final_case_hashes']['natural-use']=sha(DATA/'natural-use-final-cases.json')
    protocol['natural_use_probes']={'frozen_at':stamp(),'photo_groups':6,'portrait':3,'landscape':3,
        'selection':'Supplementary behaviour check, not checkpoint ranking; photos overlap development, original EN requests not training data; ZH AI translations.'}
    write(EXP/'protocol.json',protocol)
if __name__=='__main__':main()
