"""Record the remaining7 directly viewed candidate native-pixel followups."""
from .native_review import record


def main():
    notes={
      'aa1fa5feba0cd744':('C','m','All four overlapping native tiles covering1780x2046 were compared with the original. Face/shirt separation increases and skin/shirt texture remains. Dark hair loses some fine separation and the warm left cheek becomes more orange/red; this minor side effect prevents a clean quality pass.'),
      '0f3ffae0b63d7de4':('C','A','At both explicit native sky-points and foreground regions, darkening and stronger separation retain distinct star streaks and the mountain outline/rock layers. Source grain remains without newly intrusive noise or a broad crushed region. Together with whole-frame context this supports the requested edit; these are sampled regions, not complete native coverage of5616x3744.'),
      '44ef9e6cb54ca834':('C','A','Both native sky-points and foreground samples retain visible star streaks, ridge contours and rock layers after darkening and increasing contrast. No broad added crush or newly intrusive noise is visible in these regions or whole-frame context. The two native regions do not cover the entire5616x3744 original.'),
      '6ba7e975dd62813a':('C','A','The whole1024x1024 original and result show slight foreground lift with stronger colour/contrast. Dock grain, shaded rails and person outline/clothing remain; sun-halo gradients are retained around the already clipped sun, without broad extra clipping or intrusive new noise.'),
      'd380b3e178b0dd38':('C','A','At whole-image native1024x1024, the slightly brighter dock/person and stronger colour/contrast retain surface grain, rail shading and clothing structure. Existing sun clipping is not recovered, but surrounding gradients remain without broad added clipping.'),
      'd76bb6c244bbe713':('C','A','All four explicit1200x1200 native samples were compared: left terraces, right water terraces, mountain/cloud and bright sky. The slight lift retains terrace edges/water texture, distant mountain/cloud separation and bright-sky gradient; no obvious added clipping or new artefact appears. Whole-frame context remains on the anonymous sheet; sampled regions do not claim complete native coverage of9000x5012.'),
      '602c66a826d61062':('C','A','The four explicit native regions retain water/plant texture, terrace edges, cloud/mountain layers and the bright-sky gradient after a slight lift. No broad new tonal destruction is visible with the whole-frame sheet. This is bounded region QA of the9000x5012 image, not an inspection of every original pixel.'),
    }
    for item,(completion,quality,evidence) in notes.items():
        record('native-development','old-qualified-policies',item,completion,quality,evidence,native_folder='candidate-e20-native-pixels')


if __name__=='__main__':main()
