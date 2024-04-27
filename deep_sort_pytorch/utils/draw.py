import numpy as np
import cv2
import time

from language_pack.language import dictionary as dicts
dicts = dicts("de")


a       = 2
palette = (a ** 11 - 1, a ** 15 - 1, a ** 20 - 1)



colors = [ (148,0,211),(36, 170, 210),(255,0,0),(0,255,0) ]


def compute_color_for_labels(label):

    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0), square=True, dotonly=False, conf=None):

    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0   

        color = compute_color_for_labels(id) if id is not 0 else (255, 255, 255)

        if id is not 0:
            #color = colors[1]
            label = clabel=dicts.get("app_49") + " " + str(id)
        else:
            color = colors[1]
            label = dicts.get("app_49")

        clabel = str(round(conf[i],2))+"%" if conf is not None else False


        blabel = time.strftime("%m %d %Y %H:%M:%S", time.localtime())


        #t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]

        w = x2-x1
        h = y2-y1

        sq = min(w,h)

        centerx = x1+w/2
        centery = y1+h/2

        #if square:
            #x1 = int(centerx-sq/2)
            #y1 = int(centery-sq/2)
            #w  = int(sq)
            #h  = int(sq)

        draw_rectangle( img, x1, y1, w, h, label, color=color, with_center=True, blabel=blabel, clabel=clabel)
        #cv2.rectangle(img,(x1, y1),(x2,y2),color,1)
        #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        #cv2.putText(img,label,(x1,y1+t_size[1]-2), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
    return img


def rgbtobgr(rgbcolor):
    return (rgbcolor[2],rgbcolor[1],rgbcolor[0])


def draw_rectangle(frame,x,y,w,h,text=False,color=(255,255,255),with_center=True,border_radius=4,dotonly=False,blabel=False, clabel=False):
    #c=(0, 163, 221)
    c = rgbtobgr(color)


    #border_radius  = 5
    #line_length    = border_radius+10
    line_length_w    = int(border_radius+w/3)
    line_length_h    = int(border_radius+h/3)

    lt = float(w)*0.15
    aa = int(x+w/2)
    bb = int(y-lt/2)
    cc = int(y+h/2)
    dd = int(y-h/2)

    center_top    = [ ( aa, int(y-lt/2) ), ( aa, int(y+lt/2) ) ]
    center_bottom = [ ( aa, int(y+h-lt/2) ),   ( aa, int(y+h+lt/2) ) ]
    center_left   = [ ( int(x-lt/2), cc ),   ( int(x+lt/2), cc ) ]
    center_right  = [ ( int(x+w-lt/2), cc ), ( int(x+w+lt/2), cc ) ]

    th=2

    if not dotonly:

        br = border_radius

        cv2.line(frame,center_top[0],center_top[1],c,1)
        cv2.line(frame,center_bottom[0],center_bottom[1],c,1)
        cv2.line(frame,center_left[0],center_left[1],c,1)
        cv2.line(frame,center_right[0],center_right[1],c,1)


        cv2.line(frame,(x+br, y),(x+w-br, y),c,1) #top
        cv2.line(frame,(x+br, y+h),(x+w-br, y+h),c,1) #bottom
        cv2.line(frame,(x, y+br),(x, y+h-br),c,1) #left
        cv2.line(frame,(x+w, y+br),(x+w, y+h-br),c,1) #right


        cv2.line(frame,(x+br, y+h),(x+line_length_w, y+h),c,th)
        cv2.line(frame,(x+w-line_length_w, y+h),(x+w-br, y+h),c,th)
        cv2.line(frame,(x+br, y),(x+line_length_w, y),c,th)
        cv2.line(frame,(x+w-line_length_w, y),(x+w-br, y),c,th)
        cv2.line(frame,( x+w, y+br ),( w+x, y+line_length_h ),c,th)
        cv2.line(frame,( x+w, y+h-line_length_h ),( w+x, h+y-br ),c,th)
        cv2.line(frame,( x, y+br ),( x, y+line_length_h ),c,th)
        cv2.line(frame,( x, y+h-line_length_h ),( x, h+y-br ),c,th)
        cv2.ellipse(frame, (x+br, y+br), (br, br), 180.0, 0, 90, c, th);
        cv2.ellipse(frame, (x+w-br, y+br), (br, br), 270.0, 0, 90, c, th);
        cv2.ellipse(frame, (x+w-br, y+h-br), (br, br), 0.0, 0, 90, c, th);
        cv2.ellipse(frame, (x+br, y+h-br), (br, br), 90.0, 0, 90, c, th);
    else:
        frame = mark_center_of_rectangle(frame,x,y,w,h,c,with_center)


    if text:
        centerx = x+w/2
        centery = y+h/2
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2 , 1)[0]
        tb_size = cv2.getTextSize(blabel, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        #tc_size = cv2.getTextSize(clabel, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.putText(frame,text,(int(x+w+8),int(y+t_size[1])), cv2.FONT_HERSHEY_PLAIN, 2, c, 1)

        if blabel:
            cv2.putText(frame,blabel,(int(x+w+8),int(y+t_size[1]*2)), cv2.FONT_HERSHEY_PLAIN, 1, c, 1)

        if clabel:

            cv2.putText(frame,clabel,(int(x+w+8),int(y+tb_size[1]+10+t_size[1]*2)), cv2.FONT_HERSHEY_PLAIN, 1, c, 1)

        #cv2.putText(frame,text,(x+3,y+t_size[1]), cv2.FONT_HERSHEY_PLAIN, 1.2, c, 1)
        #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        #cv2.putText(frame, text, (int(x), int(y-15)), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1)

    return frame

def calculate_center_position(x,y,w,h):
    x_center         = int( x+w/2 )
    y_center         = int( y+h/2 )
    return x_center, y_center

def draw_center_dot(frame,x,y,color=(0, 163, 221),log=True):
    cv2.circle(frame, (x, y), 4, color, -1)
    return frame

def mark_center_of_rectangle(frame,x,y,w,h,color=(0, 163, 221),log=True):
    x_center, y_center = calculate_center_position(x,y,w,h)
    return draw_center_dot(frame,x_center,y_center,color,log)




if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
