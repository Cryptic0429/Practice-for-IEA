"""Notes from43 directly viewed anonymous full-frame contact-sheet outputs."""
from .record_notes import record

def main():
    notes={
      'natural-028':[
        ('C','U','Faces and overall photograph are visibly lifted; shirt brightness and skin colour need original-size inspection before basic-quality admission.') for _ in range(8)],
      'natural-033':[
        ('C','A','Face and scene are lighter, foliage and dress colours stronger; whole-frame content and visible tonal structure retained without broad new clipping.'),
        ('P','A','The scene is modestly lighter but colour looks muted rather than more vibrant, so the joint brightening and colour request is not complete.'),
        ('C','A','Face and scene are lighter and foliage/dress more colourful; whole-frame composition and visible tonal structure remain acceptable.'),
        ('C','A','Visible lightening accompanies stronger greens and dress colours, without an obvious broad tonal loss on the whole-frame view.'),
        ('C','A','Both the person and surrounding vegetation are lighter and more colourful; no obvious extra cast or broad clipping on the full-frame contact view.'),
        ('C','A','Face, dress and foliage are visibly brighter with added colour; composition and main tonal separations remain visible.'),
        ('C','A','Clear lightening and added colour are both visible in the subject and foliage; no obvious broad structural or tonal damage in context.')],
      'natural-055':[
        ('C','U','Blue trim, red tie and yellow clothing have visibly added colour; original-size inspection is required for white/yellow cloth and skin quality.') for _ in range(5)],
      'natural-019':[
        ('C','A','Foreground and tree are clearly lighter, with a warmer yellow-orange sky; clouds and scene structure stay visible in full-frame context.'),
        ('P','A','Sky is warmer but foreground lift remains modest relative to the request for a much brighter photograph; no obvious broad added damage.'),
        ('C','A','The field and foreground branches are clearly lifted and the sky warmer, with visible cloud and foreground tonal structure.'),
        ('P','A','The foreground is lighter, but the sky remains cool blue rather than receiving the requested overall warmth; joint request incomplete.'),
        ('C','A','The photograph is clearly lighter and warmer, while the sky bands, clouds and foreground structure remain readable.'),
        ('C','A','Both a clear foreground lift and warmer sky are visible, with the original composition and cloud structure retained.'),
        ('C','A','The darker foreground is visibly lifted and the overall rendering warmer; no broad added clipping or loss of cloud structure on this view.')],
      'natural-022':[
        ('C','A','Vegetation and warm rock colours are more vivid, with the cloudy sky, rocky layers and composition retained and no obvious non-natural cast.') for _ in range(8)],
      'natural-030':[
        ('C','A','The entire scene, road and bright upper background are visibly darker; road markings, tree separations and ground remain readable without broad crushing.') for _ in range(8)]}
    record('old-native','joint-routes-epochs',notes,'direct-contact-observations')

if __name__=='__main__':main()
