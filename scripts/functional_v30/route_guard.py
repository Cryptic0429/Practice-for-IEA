"""Conservative agreement check for explicit, single signed tonal requests."""
import re
import unicodedata

CUT = re.compile(r'[，。；,.!?;]|\b(?:while|without|and then|and also|but)\b|同时|然后|再给',re.I)
PRESERVE = re.compile(
    r'^(?:请)?(?:保留|保住|保持|留住|留着|不改变|不要|别|不能|不用|无需)|'
    r'^(?:keep|keeping|preserve|preserving|retain|retaining|leave|leaving|hold|avoid|do not|don.t|no)\b',re.I)
UP = re.compile(r'提亮|抬亮|变亮|调亮|加亮|亮一点|(?:提高|增加)\S{0,7}(?:亮度|曝光)|'
    r'\b(?:brighten|lighten|raise|lift)\b|\b(?:make|bring)\b.{0,55}\bbrighter\b|'
    r'\b(?:increase|add)\b.{0,15}\b(?:exposure|brightness|light)\b',re.I)
DOWN = re.compile(r'压暗|调暗|收暗|变暗|暗一点|(?:降低|调低|收|压低)\S{0,12}(?:曝光|亮度)|'
    r'(?:曝光|亮度)\S{0,6}(?:降低|调低|收一点|降一点)|'
    r'\b(?:darken|deepen|lower|reduce)\b|\b(?:bring|take|pull)\b.{0,65}\bdown\b|'
    r'\bmake\b.{0,65}\bdarker\b',re.I)
WHOLE = re.compile(r'整[张幅]|整体|全图|全画面|全幅|\b(?:whole|overall|entire|across the frame)\b',re.I)
SHADOW = re.compile(r'暗部|阴影|暗处|暗面|树荫|\b(?:shadows?|shaded|shade|dark tones)\b',re.I)
BRIGHT = re.compile(r'最亮|高光|阳光照到|受光处|亮处|\b(?:sunlit|highlights?|brightest)\b',re.I)
GENERIC = re.compile(r'曝光|亮度|照片|画面|\b(?:exposure|brightness|photograph|photo|image|scene)\b',re.I)
OTHER = re.compile(r'棕褐|饱和度|色温|对比度|移除|删除|\b(?:sepia|saturation|white balance|contrast|remove)\b',re.I)
QUESTION = re.compile(r'要不要|是否|先告诉|\b(?:should|tell me what|do you think)\b',re.I)


def explicit_route(text):
    text = unicodedata.normalize('NFKC',text).strip()
    if QUESTION.search(text):
        return 'defer'
    routes = set()
    for part in (s.strip() for s in CUT.split(text)):
        if not part or PRESERVE.search(part):
            continue
        if OTHER.search(part):
            return 'defer'
        up,down = bool(UP.search(part)),bool(DOWN.search(part))
        if up and down:
            return 'defer'
        if not (up or down):
            continue
        whole,shadow,bright = bool(WHOLE.search(part)),bool(SHADOW.search(part)),bool(BRIGHT.search(part))
        if (whole and shadow) or (bright and not whole) or (not whole and not shadow and not GENERIC.search(part)):
            return 'defer'
        routes.add(('shadow' if shadow else 'global')+('_up' if up else '_down'))
    return next(iter(routes)) if len(routes)==1 else 'defer'
