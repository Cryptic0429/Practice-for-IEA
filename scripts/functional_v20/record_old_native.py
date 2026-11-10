"""Native whole-image followups: input960/1024 andall13 outputs directly viewed."""
from .native_review import record

def main():
    grades={
      '14c737d4b8614fd7':('A','Subtle overall lift is visible on faces, hands and shirt. Original-size shirt stripes/folds, hair, dress and facial structure remain; existing warm skin and shirt pink are not new recovery claims.'),
      '22200434715da91c':('A','Visible modest brightening retains shirt stripes and folds, hair strands and dress texture at960x640, without broad new white clipping or an obvious new skin cast.'),
      '566ed3434b3b32b0':('A','The face and shirt are lighter; original-size folds/stripes remain readable and skin/hair texture stays visible. No major new clipped clothing region or scene change is evident.'),
      '5ccc1384a5e2ebc3':('U','Brightening is visible but the entire rendering is also flatter/washed compared with the input. Native pixels retain texture; whether the extra tonal flattening meets basic quality remains uncertain, not counted as a clean pass.'),
      '6434925436b09f64':('A','The whole photo is visibly lifted while shirt folds/stripes, faces and hair remain readable at960x640; no broad new clipped region or obvious added colour cast is evident.'),
      '6a1377a98d178dd2':('A','Skin, shirt and background are visibly lighter with original-size shirt folds/stripes and dress texture retained. Existing warm lighting remains; no obvious broad added damage.'),
      '8f74600fe50e33b5':('A','Faces and shirt are brighter, with readable shirt stripes/folds and retained skin, hair and dress structure at native size. No broad new white loss or obvious new cast.'),
      'b7865f1ac610b440':('U','The photo is brighter but the flatter, slightly washed rendering changes overall tonal character beyond a simple lift. Native shirt texture survives; quality acceptability remains uncertain and no clean pass is inferred.'),
      '0a87292ce0e05824':('A','Modest added red/blue/yellow colour remains plausible. At1024x681 the white/yellow cardigan weave/folds, hair and facial structure are retained without broad new clipping or an excessive skin cast.'),
      '72a1cc8da8da0fd9':('A','The blue trim, red tie and yellow cardigan are mildly more colourful. Native cloth weave and folds, facial shading and hair survive; no broad new loss in the bright white fabric is evident.'),
      '7c6db97bb3c3101d':('A','Added colour stays moderate in tie, trim and cardigan. Whole native pixels retain fabric weave/folds and face/hair structure without broad new white clipping or implausible skin colour.'),
      'da0cebe15f817dcc':('A','Tie, trim and cardigan gain moderate colour; original-size white/yellow fabric still has weave, creases and shading. Existing bright parts are not claimed repaired; no broad added loss is evident.'),
      'dd45e60700fbed4b':('A','The small colour increase retains native white/yellow knit texture, folds, face and hair. No broad added clipping or obviously excessive skin/yellow saturation is seen.')}
    for item,(quality,evidence) in grades.items():record('old-native','joint-routes-epochs',item,'C',quality,evidence)

if __name__=='__main__':main()
