# A little collection of tools
#
#
#  _    _ _   _ _     
# | |  | | | (_) |    
# | |  | | |_ _| |___ 
# | |  | | __| | / __|
# | |__| | |_| | \__ \
#  \____/ \__|_|_|___/
#                     
#  
class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

def cprint( text, color="OKGREEN" ):
    return print( getattr(bcolors, color) + text + bcolors.ENDC )

# takes [xcenter,y,center,w,h] in px
# return [x1,y1,x2,y2] in px
def to_xyxy(a,box=False,cw=1280,ch=720):
    r = []
    for b in a:
        xc = b[0]
        yc = b[1]
        w  = b[2]
        h  = b[3]
        x1 = (xc-w/2)
        y1 = (yc-h/2)
        x2 = (x1+w)
        y2 = (y1+h)
        if box:
            x1 = int(box[0]+x1)
            y1 = int(box[1]+y1)
            x2 = int(x1+w)
            y2 = int(y1+h)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        r.append( ( x1, y1, x2, y2 ) )
    return r
# converts box format from px to percent
# input: [xcenter,ycenter,w,h] in px
# output: [xcenter,ycenter,w,h] in percentage to frame
# it returns the data as string with class_id
# and as [xcenter,ycenter,w,h] list
# @return ( str, list ) 
def convert_darknet(box,frame,class_id):
    h, w, _ = frame.shape
    x = box[0]/w
    y = box[1]/h
    w = box[2]/w
    h = box[3]/h
    return ( " ".join([str(class_id), str(x), str(y), str(w), str(h)]), [x,y,w,h], )

# checks if given [x,y] are within given button [x1,x2,y1,y2]
def check_if_cord_is_within(a,b):
    if a[0] >= b[0] and a[0] <= b[2] and a[1] >= b[1] and a[1] <= b[3]:
        return True
    return False


class jpObject:

    def __init__(self,response):
        self.__dict__['_response'] = response

    def __getattr__(self, key):
        # First, try to return from _response
        try:
            return self.__dict__['_response'][key]
        except KeyError:
            pass
        # If that fails, return default behavior so we don't break Python
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError