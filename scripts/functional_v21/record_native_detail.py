"""Append only native-pixel images directly viewed, keeping contact notes intact."""
from .native_review import record


def main():
    notes={
      'a8ab99208f26c4c7':('C','A','Both overlapping native tiles cover the full1494x2450 scan. Face and clothing gain lightness and separation while beard, folds and hand structure remain. Grain, fading and stains already occur in the original; no broad new clipping or intrusive new artefact is visible.'),
      '7cfa2e45dec749cf':('C','A','Both overlapping native tiles cover the full1494x2450 scan. The face and pale clothing are lighter with stronger separation and retained beard, folds and hands. Existing scan grain and stains remain; no broad new tonal destruction is visible.'),
      'df799e1849fa2952':('C','m','The whole960x1280 image has visible contrast, slight lightening and stronger colour, retaining sky and shirt texture. The already reddish facial shadows become more magenta/red and greens become quite strong; this is a minor colour-quality issue rather than a clean pass.'),
      '27713fb7cef0f5e5':('C','m','Native whole-image comparison retains shirt texture and sky while slightly lightening and adding contrast/colour. The existing reddish facial shadows become more pronounced and foliage more intense; this minor colour issue prevents a clean quality pass.'),
      '864d4b43279ea9ee':('C','A','The whole960x1280 comparison shows a slight lift of railing and tree/building shadows. The source sun is already clipped; its surrounding gradient and visible sky/building structure remain, without intrusive new shadow noise or a broad extra white area.'),
      '23aef532a98a1031':('C','A','The original-size whole image shows a slight lift in the trees and railing with the existing sun/flare layout preserved. Visible sky gradients and buildings remain; no broad added clipping or newly intrusive noise is apparent.'),
      '689a242afa0994da':('C','m','Whole639x1136 comparison shows lighter faces and room with glasses, facial edges and clothing retained. Strong source grain remains and becomes slightly more conspicuous in the lifted face; this is not a clean quality pass or recovered detail claim.'),
    }
    for item,(completion,quality,evidence) in notes.items():
        record('native-development','old-qualified-policies',item,completion,quality,evidence,native_folder='candidate-e20-native-pixels')


if __name__=='__main__':main()
