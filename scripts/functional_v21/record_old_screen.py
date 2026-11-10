"""Record13 directly viewed new anonymous outputs; prior exact pixels remain inherited."""
from .record_notes import record

def main():
    notes={
      'natural-033':[
        ('C','A','The face and whole scene are lighter and vegetation/dress more colourful. Frame and visible tonal structure remain without obvious broad clipping.'),
        ('C','A','Both visible lightening and stronger greens/dress colours are present, with the whole-frame scene and facial shading retained.'),
        ('C','A','Moderate visible lightening accompanies more vivid foliage and dress colour; no obvious broad extra cast or tonal loss in context.'),
        ('C','A','The person and scene are visibly lighter with added foliage/dress colour, while major tonal structure and composition remain.')],
      'natural-055':[
        ('C','U','The trim, tie and yellow fabric gain a little colour. White/yellow weave and skin preservation need native-pixel inspection before a clean quality pass.')],
      'natural-019':[
        ('P','A','The sky is warmer but the foreground lift is modest relative to the request for a much brighter image. The frame/cloud layout remains without obvious broad added damage.'),
        ('C','A','The foreground and tree are clearly brighter and the sky warmer, with visible cloud and foreground structure retained.'),
        ('C','A','A substantial foreground lift and warmer sky are both present; the tree, field and cloud structure remain in whole-frame context.')],
      'natural-022':[
        ('C','A','Green vegetation and warm rock colours are visibly more vivid, with cloudy sky and layered rocks retained and no obvious unnatural cast.'),
        ('C','A','The vegetation and rock gain clear colour while sky and rock-layer structure remain readable without fluorescent colour or broad added clipping.')],
      'natural-030':[
        ('C','A','The entire photograph is clearly darker; road markings, roadside ground and tree separations remain visible rather than being broadly crushed.'),
        ('C','A','Strong whole-frame darkening is visible, preserving the curved road, yellow markings and tree/ground separations.'),
        ('C','A','Road, trees and bright upper background are visibly darker, with main markings and forest structure still readable.')]
    }
    record('old-native','shared-head-policies',notes,'direct-contact-observations')

if __name__=='__main__':main()
