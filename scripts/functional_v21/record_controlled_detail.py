"""Eight directly viewed whole-native comparisons supplement contact uncertainty."""
from .native_review import record


def main():
    notes={
      '1d0cb3e9f8cb78d4':('C','m','Whole768x1024 comparison confirms the stronger saturation, retaining lace outlines, toy texture and bedding folds. The red garment develops flatter intense-red regions and the face becomes more orange; this minor colour/texture side effect prevents a clean quality pass.'),
      '6afe234f2ce4e05e':('C','m','At whole native768x1024, stronger saturation keeps the lace and bedding structure, but red fabric becomes more flat/intense and skin more orange. This minor extra colour/texture problem is not a clean pass; there is no recovery of source detail.'),
      '1d56ae615d9a72fc':('C','A','Whole683x1024 comparison shows noticeably stronger contrast, retaining shirt grid/folds, face shading and tree branches. The source bright sky is already near white; no broad newly lost shirt or facial structure is apparent.'),
      '4529a52ae459c45c':('C','A','The complete native683x1024 image retains checkered shirt pattern, faces and trouser folds after the stronger contrast. Existing bright sky remains, without a broad extra loss of subject texture or unnatural skin shift.'),
      '095f634487fb2bbd':('C','A','Whole960x720 comparison retains distinct needle/branch silhouettes, field texture and sunset/cloud gradients under the stronger contrast. The shaded tree is darker as expected, but major structure remains readable without a broad newly crushed region.'),
      '70a3583bb1a4ed4b':('C','A','The whole native960x720 scene shows stronger separation while branch/needle shapes, field texture and sunset gradients remain. Source shadows deepen but do not broadly obscure the scene structure.'),
      '03f6a4b0b8af0777':('C','A','Whole1024x768 comparison confirms stronger shadow darkening with distinct shaded shrub shapes, rock gaps and lit rock surface texture retained. Existing sunset colour and smooth sky remain without a broad new crushed area.'),
      'f8652021d007a062':('C','A','At complete native1024x768, shaded vegetation and rock separation remain readable after noticeable shadow darkening. Lit rock cracks, sunset colour and sky gradient are preserved without intrusive new noise or broad crushing.'),
    }
    for item,(completion,quality,evidence) in notes.items():
        record('development','controlled-qualified-policy',item,completion,quality,evidence)


if __name__=='__main__':main()
