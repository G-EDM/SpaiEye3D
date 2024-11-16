# coding: utf-8 
import os, sys, cv2, time, argparse, json, natsort, torch, uuid, pickle, struct, threading, socket
import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui

#import subprocess
from multiprocessing import Process, Pipe, Queue, Pool

# Import the stuff from deep_sort_pytorch
# Credits to ZQPei
# Github: https://github.com/ZQPei
# https://github.com/ZQPei/deep_sort_pytorch
from deep_sort_pytorch.detector import build_detector
from deep_sort_pytorch.deep_sort import build_tracker
from deep_sort_pytorch.utils.draw import draw_boxes
from deep_sort_pytorch.utils.draw import draw_rectangle
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.detector.YOLOv3 import YOLOv3

# import our stuff
from language_pack.language import dictionary as dicts
from artwork.header_art import get_header
from utils.tools import *


#
#
#  __  __       _               _           _               
# |  \/  |     (_)             (_)         | |              
# | \  / | __ _ _ _ ____      ___ _ __   __| | _____      __
# | |\/| |/ _` | | '_ \ \ /\ / / | '_ \ / _` |/ _ \ \ /\ / /
# | |  | | (_| | | | | \ V  V /| | | | | (_| | (_) \ V  V / 
# |_|  |_|\__,_|_|_| |_|\_/\_/ |_|_| |_|\__,_|\___/ \_/\_/  
#                                                                                                                     
#   
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, conn=None, args=None, parent=None):
        super(MainWindow,self).__init__(parent)
        cprint("MainWindow init","WARNING")
        self.window_title   = "Face_tracker"
        self.display_width  = args.display_width
        self.display_height = args.display_height
        self.args           = args
        self.cfg            = cfg
        self.configs        = shared_configs(args)
        self.setWindowTitle(self.window_title)
        self.setGeometry(10, 10, self.display_width, self.display_height)

    def destroy(self):
        self.central_widget.destroy()
        sys.exit(1)

    def closeEvent(self, event):
        self.central_widget.destroy()
        event.accept() 

    def run(self):
        self.central_widget = MainWidget( self.cfg, self.configs, self )
        self.setCentralWidget( self.central_widget )
        self.central_widget.run()
        self.resize( self.display_width, self.display_height )
        self.show()

    def reset_size(self,w,h):
        cprint(dicts.get("app_81"))
        self.display_width  = w
        self.display_height = h
        self.setGeometry(10, 10, w, h)
        self.setFixedSize(w,h)

    def get_cap_dims(self):
        w = self.display_width
        h = self.display_height
        return w, h




# The main Widget
# Connects all the widgets, buttons, physics
#
#
#  __  __       _               _     _            _   
# |  \/  |     (_)             (_)   | |          | |  
# | \  / | __ _ _ _ ____      ___  __| | __ _  ___| |_ 
# | |\/| |/ _` | | '_ \ \ /\ / / |/ _` |/ _` |/ _ \ __|
# | |  | | (_| | | | | \ V  V /| | (_| | (_| |  __/ |_ 
# |_|  |_|\__,_|_|_| |_|\_/\_/ |_|\__,_|\__, |\___|\__|
#                                        __/ |         
#                                       |___/          
#
#
class MainWidget(QtWidgets.QWidget):
    def __init__(self, cfg, configs, parent=None):
        super(MainWidget, self).__init__(parent)
        cprint("MainWidget init","WARNING")
        self.parent        = parent
        self.cfg           = cfg
        self.greeter_build = False
        self.count         = 0
        self.configs       = configs
        self.buttons       = []
        self.record_video  = False
        self.physics       = False
        self.has_init      = False
        self.terminate_event = False

        self.setCursor( QtCore.Qt.WaitCursor )

        # Start the detection Widget
        self.face_detection = FaceDetectionWidget(self.configs,self)
        # Add the buttons
        # We only add buttons for Webcam and Video
        # Prelabeling and Images don't need the buttons
        # Also physics is only needed in Webcammode
        if not self.configs.is_image:

            self.record_video = RecordVideo( self.configs )

            # Start the recorder Widget
            if self.configs.webcam_mode:
                cprint(dicts.get("app_74"))
                # Start the physics features
                w, h = self.get_cap_dims()
                self.physics = physics(self.record_video)
                self.physics.set_steps(w,h)

                if self.configs.remote_video:
                    cprint(dicts.get("app_75"))
                    self.physics.set_callback( self.record_video.sock_send )

            # Emitter used to draw an image with detection to the window
            self.record_video.image_data.connect( self.face_detection.image_data_slot )

            #
            self.record_video.destroy_event.connect( self.parent.destroy )

            # Emitter used to record the motions in monitor mode
            self.face_detection.monitoring.connect( self.record_video.monitor_record )
            # Emitter used to transfer current frames ( raw and bboxed )
            self.face_detection.frametransfer.connect( self.record_video.set_current_frames )

        # After the recorder got the first frame done we emit the state
        self.face_detection.initemitter.connect( self.adjust_size )

        # Add some buttons
        self.create_buttons()

        # Place Widget
        self.face_detection.move(0,0)

        # Load the stylesheet
        self.setStyleSheet(open('./artwork/main.css').read())

        self.build_greeter()
        

    def adjust_size(self,w,h):
        cprint( dicts.get("app_76"))
        self.setCursor( QtCore.Qt.ArrowCursor )
        self.parent.reset_size(w,h)
        self.has_init = True
        return self.adjust()

    def get_cap_dims(self):
        return self.parent.get_cap_dims()

    def run(self):
        if self.configs.scan_folder:
            # Prelabel images in given folder
            return self.face_detection.label_folder()
        if not self.configs.is_image:
            # Run the recorder for video or webcam
            self.record_video.start_recording()
        else:
            # Run the detector over an image
            self.face_detection.scan_image()

    def destroy(self):
        #self.create_timer(0,self.face_detection.destroy)
        self.face_detection.destroy()
        if self.record_video:
            #self.create_timer(0,self.record_video.destroy)
            self.record_video.destroy()

        if self.terminate_event:
            return 
        self.terminate_event = True


    # Build a little greeter Image until
    # stuff is ready
    def build_greeter(self):
        cprint(dicts.get("app_80"))
        if self.greeter_build:
            return
        self.greeter_build = True
        w, h = self.get_cap_dims()
        frame = cv2.resize( cv2.imread(self.configs.loader_image, cv2.IMREAD_COLOR), (w,h),  interpolation = cv2.INTER_NEAREST)
        self.face_detection.image_data_slot_direct( frame, False )

    def paintEvent(self, event):
        #print("MainWidget paint event")
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.face_detection.image)
        if self.has_init:
            return
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setPen(QtGui.QColor(36, 170, 210))
        qp.setFont(QtGui.QFont('Decorative', 20))
        qp.drawText(20, 40, dicts.get("app_72"))  
        qp.setPen(QtGui.QColor(36, 170, 210))
        qp.setFont(QtGui.QFont('Decorative', 10))
        qp.drawText(20, 60, dicts.get("app_73"))
        qp.end()

    def create_timer(self,i,cb,args=None):
        if self.terminate_event:
            print("MainWidget timer blocked")
            return False
        #print("MainWidget::Timer")
        timer = threading.Timer(i, cb, args)
        timer.start()
        return timer

    def create_buttons(self):
        if not self.configs.extraction_mode and not self.configs.is_image:
            self.set_default_buttons()
            if self.configs.has_physics:
                self.set_physics_buttons()
            self.checkout()

    def resizeEvent(self,event):
        #print("MainWidget resize event")
        if not self.configs.is_image and self.greeter_build:
            self.create_timer(0.01,self.adjust)

    # buttons used for physics
    def set_physics_buttons(self):
        
        # Buttons to control the physics and move the cam
        move_up_button    = QtWidgets.QPushButton('↑',self)
        move_down_button  = QtWidgets.QPushButton('↓',self)
        move_left_button  = QtWidgets.QPushButton('←',self)
        move_right_button = QtWidgets.QPushButton('→',self)

        self.buttons.append( [move_up_button,    self.physics.up, dicts.get("app_83"),    [90,80],[30,30]] )
        self.buttons.append( [move_down_button,  self.physics.down, dicts.get("app_84"),  [90,40],[30,30]] )
        self.buttons.append( [move_left_button,  self.physics.left, dicts.get("app_85"),  [125,60],[30,30]] )
        self.buttons.append( [move_right_button, self.physics.right, dicts.get("app_86"), [55,60],[30,30]] )


    # default button
    def set_default_buttons(self):
        w, h = self.get_cap_dims()
        save_frame_button = QtWidgets.QPushButton(dicts.get("app_51"),self)
        self.buttons.append( [save_frame_button, self.record_video.save, dicts.get("app_55"), [0,40],[70,30]] )
        # Show web
        show_web_button = QtWidgets.QPushButton("Ψ",self)
        self.buttons.append( [show_web_button, self.face_detection.toggle_web, dicts.get("app_87"), [0,40],[30,30]] )

    # finally connect the buttons
    def checkout(self):
        w, h = self.get_cap_dims()
        for pair in self.buttons:
            pair[0].clicked.connect( pair[1] )
            pair[0].setFixedSize( pair[4][0],pair[4][1] )
            pair[0].move(-1000,-1000)
            pair[0].setToolTip( pair[2] )
            pair[0].setAttribute( QtCore.Qt.WA_TranslucentBackground, True )
            pair[0].setCursor( QtCore.Qt.PointingHandCursor)

    # readjust buttonposition
    def adjust(self):
        cprint( dicts.get("app_77"))
        if not self.has_init:
            return
        w, h = self.get_cap_dims()
        if self.physics:
            self.physics.set_steps(w,h)

        margin    = 5
        current_x = 10


        for pair in self.buttons:

            if pair[3][0] <= 0:
                sw = current_x
                current_x+=margin
                current_x+=pair[4][0]
            else:
                sw = w-pair[3][0]
            if pair[3][1] <= 0:
                sh = 10
            sh = h-pair[3][1]
            pair[0].move(sw,sh)





# The Class that does the detection
# and tracking
# It does some more stuff too
# Drawing the image, keeping track of the mousepress event,
# Motion detection etc.
#
#  _____       _            _             
# |  __ \     | |          | |            
# | |  | | ___| |_ ___  ___| |_ ___  _ __ 
# | |  | |/ _ \ __/ _ \/ __| __/ _ \| '__|
# | |__| |  __/ ||  __/ (__| || (_) | |   
# |_____/ \___|\__\___|\___|\__\___/|_|   
#                                         
#                                         
class FaceDetectionWidget(QtWidgets.QWidget):
    monitoring    = QtCore.pyqtSignal(int)
    frametransfer = QtCore.pyqtSignal(np.ndarray,np.ndarray)
    initemitter   = QtCore.pyqtSignal(int,int)

    def __init__( self, configs=None, parent=None ):
        super(FaceDetectionWidget,self).__init__(parent)
        cprint("FaceDetectionWidget init","WARNING")
        self.configs               = configs
        self.image                 = QtGui.QImage()
        self.previous_frame        = None
        self.current_bboxes        = False
        self.current_bboxes_xcycwh = False
        self.skip_motion           = False
        self.skip_motion_count     = 0
        self.no_motion_iter        = 10
        self.no_motion_count       = 0
        self.record_after          = 0
        self.has_motion            = False
        self.is_first_run          = True
        self.cuda_enabled          = self.configs.use_cuda and torch.cuda.is_available()
        self.terminate_event       = False
        self.init_send             = False
        self.show_webs             = False
        self.is_processing         = False
        self.extraction_number     = self.configs.extraction_number# 50 # store every nth frame
        self.extraction_counter    = 0 
        self.old_version           = False
        self.new_version           = False

        # in extration mode we don't need any detectors
        if not self.configs.extraction_mode:

            if not self.cuda_enabled:
                cprint(dicts.get("app_19"),"WARNING")
                #cprint('Falling back to YoloV3 tiny',"WARNING")
            

            #cprint('\n\nYoloV3 config:',"WARNING");
            # this is for the default YoloV3 Weights
            #print(cfg.YOLOV3)
            #self.detector = YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, is_xywh=True, use_cuda=self.cuda_enabled)

            # This is for the head detector weights 
            #print(cfg.YOLOV3HEAD)
            self.detector = YOLOv3(cfg.YOLOV3HEAD.CFG, cfg.YOLOV3HEAD.WEIGHT, cfg.YOLOV3HEAD.CLASS_NAMES, score_thresh=cfg.YOLOV3HEAD.SCORE_THRESH, nms_thresh=cfg.YOLOV3HEAD.NMS_THRESH, is_xywh=True, use_cuda=self.cuda_enabled)

            self.deepsort    = False #build_tracker(cfg, use_cuda=self.cuda_enabled)
            self.class_names = self.detector.class_names

            cprint('\n\nClassnames',"WARNING");
            print(self.class_names)


    def on_error(self,msg_key,text=False):
        cprint(dicts.get(msg_key),"FAIL")
        if text:
            cprint(text,"FAIL")
        cprint(dicts.get("app_1"),"FAIL")
        raise SystemExit(0)



    def create_timer(self,i,cb,args=None):
        if self.terminate_event:
            print("FaceDetectionWidget timer blocked")
            return False
        timer = threading.Timer(i, cb, args)
        timer.start()
        return timer

    def destroy(self):
        cprint("FaceDetectionWidget destroy called")
        self.terminate_event = True

    def mousePressEvent(self, QMouseEvent):
        self.on_mouse( int(QMouseEvent.x()), int(QMouseEvent.y()) )

    def on_mouse(self,x,y):
        if self.current_bboxes:
            for index, box in enumerate(self.current_bboxes):
                try:
                    within = check_if_cord_is_within([x,y],box)
                    if within:
                        cprint("Within","WARNING")
                except:
                    cprint("ERR0004","WARNING")

    def resize_image(self,frame,maxwh=1280):
        m       = maxwh
        h, w, _ = frame.shape
        r       = min( m/w,m/h )
        h       = int(h*r)
        w       = int(w*r)
        return cv2.resize(frame.copy(), (w,h),  interpolation = cv2.INTER_NEAREST)

    # Store the width and height of the frame and return w,h
    def set_frame_dims(self,frame):
        h, w, _         = frame.shape
        self.cap_width  = w
        self.cap_height = h
        return w, h


    # Functions used if an image file is supplied as input
    # It processes only the given image
    def scan_image(self):
        frame                                       = cv2.imread(self.configs.image_file, cv2.IMREAD_COLOR)
        frame_raw                                   = frame.copy()
        frame, boxes, confidences, class_ids, match = self.scan(frame,nosort=True)
        #print(boxes)
        filehandler.save_image(frame,self.configs.image_folder)
        frame = self.resize_image(frame)
        self.image_data_slot_direct( frame )


    # Converts cv image for use with Qt
    def get_qimage(self, image: np.ndarray):
        #print("Convert to Qt image")
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage       = QtGui.QImage
        image        = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image        = image.rgbSwapped()
        return image
    
    # If a folder is supplied
    # We use this function as callback
    # and create labelfiles in darknet format
    def loop_over_files(self):
        if int(len(self.files)) <= 0:
            cprint("Done","WARNING")
            return
        file = self.files.pop(0)
        name, extension = os.path.splitext(file)
        if extension not in self.configs.valid_image_extenions:
            return self.loop_over_files()
        path  = self.configs.scan_folder+file
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        label_file_path  = self.configs.labelfiles_folder+name+".txt"
        cprint(file)
        if frame is None:
            filehandler.override(label_file_path,"")
            return self.loop_over_files()
        else:
            # First we display the raw image without boxes
            self.current_image = self.resize_image(frame)
            self.image_data_slot_direct( self.current_image )
            frame, boxes, confidences, class_ids, match = self.scan(frame,returnboxes=True,nosort=True)
            if boxes is None:
                filehandler.override(label_file_path,"")
                return self.loop_over_files()
            lines = set()
            for i, box in enumerate( boxes ):
                class_id = class_ids[i]
                b = convert_darknet(box,frame,class_id)[0]
                lines.add(b)
            batch = "\n".join(lines)
            print(batch)
            filehandler.override(label_file_path,batch)
            # Display image with boxes
            self.current_image = self.resize_image(frame)
            self.image_data_slot_direct( self.current_image )
            self.create_timer(0,self.loop_over_files)

    # Scans a given folder for images
    # and creates a labelfile with the boxes the scanner finds
    def label_folder(self):
        files = natsort.natsorted(os.listdir(self.configs.scan_folder))
        filehandler.create_folder(self.configs.labelfiles_folder)
        self.files = files
        self.create_timer(0,self.loop_over_files)


    def toggle_web(self):
        #print("toggle_web")
        if self.show_webs:
            self.show_webs = False
        else:
            self.show_webs = True


    def draw_webs(self,frame):
        c=(0,255,0)
        #print("Drawing webs")
        if self.current_bboxes == False:
            #print("No boxes")
            return frame

        w = self.cap_width 
        h = self.cap_height
        wc = int( w/2 )
        hc = int( h/2 )

        for box in self.current_bboxes_xcycwh:
            # get center
            xcenter = int( box[0] )
            ycenter = int( box[1] )

            cv2.line(frame,( xcenter, ycenter ),( wc, hc ),c,1)

        return frame


    # draws the image to our qt window
    def image_data_slot_direct(self, frame, count=True):
        #print("Detector image_data_slot_direct called")
        self.image = self.get_qimage(frame)
        if self.image.size() != self.size() or not self.init_send:
            self.setFixedSize(self.image.size())
            w, h = self.set_frame_dims(frame)
            if count:
                #print("Detector emitting adjust event")
                self.initemitter.emit(w,h)
                self.init_send = True
        #print("Detector update called")
        #self.create_timer(0,self.update)
        self.update()



    # Motion detection
    # It just checks if we have motion
    # And returns True or False
    def motion_detection(self,gray,previous_frame,w,h):
        frameDelta          = cv2.absdiff(previous_frame, gray)
        thresh              = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow("www",thresh)

        if self.old_version:
            _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elif self.new_version:
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            try:
                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                self.new_version = True
            except ValueError:
                _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                self.old_version = True



        height, width       = thresh.shape
        min_x, min_y        = width, height
        max_x = max_y       = 0
        try: hierarchy      = hierarchy[0]
        except: hierarchy   = []
        total               = 0.0
        for contour, hier in zip(contours, hierarchy):
            (xc,yc,wc,hc) = cv2.boundingRect(contour)
            total = total + wc*hc
            #min_x, max_x = min(xc, min_x), max(xc+wc, max_x)
            #min_y, max_y = min(yc, min_y), max(yc+hc, max_y)
        #dilated1 = cv2.dilate(thresh, es)
        #cv2.imshow('www', thresh)
        a = w*h
        t = a * 0.005
        #print("total",total)
        #print("has",t)
        if total <= t:
            return False
        return True

    # the main scan function
    # given the frame it detects the objects
    # note it returns all boxes no matter what class_id
    # but it draws only boxes for the wanted class_id
    def scan(self,frame,returnboxes=False,nosort=False):
        ori_im = frame
        w, h = self.set_frame_dims(frame)
        bbox_xywh, confidences, class_ids = self.detector(frame)
        #print(bbox_xywh)
        #print(confidences)
        #print(class_ids)
        if bbox_xywh is not None:
            has_match = True
            for class_id in self.configs.detector_class_ids:
                if not class_id in class_ids:
                    continue
                mask          = class_ids == class_id
                pboxes        = bbox_xywh[mask]
                pconfidences  = confidences[mask]
                #pboxes[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                b = to_xyxy( pboxes, False, w, h )
                self.current_bboxes        = b
                self.current_bboxes_xcycwh = pboxes
                if nosort or not self.configs.use_deepsort or not self.deepsort:
                    # if we don't want the deepsort feature
                    # detecting objects in images, labeling folder etc.
                    ori_im = draw_boxes(ori_im, b, pconfidences )
                else:
                    tracked, untracked, track_success = self.deepsort.update(pboxes, pconfidences, frame)
                    if track_success:
                        if len(tracked) > 0:
                            bbox_xyxy   = tracked[:,:4]
                            identities  = tracked[:,-2]
                            conf        = tracked[:,-1]
                            ori_im = draw_boxes( ori_im, bbox_xyxy, identities, conf=conf )
                        if len(untracked) > 0:
                            bbox_xyxy  = untracked[:,:4]
                            identities = untracked[:,-2]
                            conf       = untracked[:,-1]
                            ori_im = draw_boxes( ori_im, bbox_xyxy, identities, conf=conf )
                    else:
                        ori_im = draw_boxes(ori_im, b, pconfidences )
        else:
            has_match                  = False
            self.current_bboxes        = False
            self.current_bboxes_xcycwh = False

        if self.show_webs:
            ori_im = self.draw_webs( ori_im )

        #print("Done scan")

        return ( ori_im, bbox_xywh, confidences, class_ids, has_match )

    def extrat_frames(self,frame):
        if self.extraction_counter >= self.extraction_number:
            # store the frame
            self.extraction_counter=0
            filehandler.save_image(frame,self.configs.frame_folder)
            self.image_data_slot_direct(frame)
        else:
            self.extraction_counter+=1

    # this is the function that is called after an video frame is emitted
    def image_data_slot(self, frame):

        if self.terminate_event or self.is_processing:
            return

        self.is_processing = True
        w, h = self.set_frame_dims(frame)

        if self.configs.video_stream_only:
        	self.image_data_slot_direct(frame)
        	self.is_first_run = False
        	self.is_processing = False
        	return

        # If we only want to extract some frames
        # We can skip all the scanning
        if self.configs.extraction_mode:
            self.extrat_frames(frame)
            self.is_processing = False
            return


        frame_raw = frame.copy()
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray      = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray


        skip_yolo = False

        if self.configs.webcam_mode:

            # motion detection part
            # used to safe energy and detect motions
            if self.skip_motion:
                #print("Motion skipped")
                self.record_after = 20
                # if we have yolo matches we can skip motion detection
                # if we know we have motion we can skip motion detection for some rounds
                self.skip_motion_count+=1
                if self.skip_motion_count>=self.no_motion_iter:
                    self.skip_motion_count = 0
                    self.skip_motion       = False
            else:

                if self.no_motion_count>=2:
                    self.no_motion_count=0

                if self.no_motion_count<=0:
                    self.has_motion = self.motion_detection(gray,self.previous_frame,w,h)
                    #print("Has motion? ",self.has_motion)

                if not self.has_motion:
                    self.no_motion_count+=1
                    skip_yolo = True
                else:
                    self.skip_motion       = True
                    self.skip_motion_count = 0
                    self.no_motion_count   = 0
                    self.record_after      = 20
            

        # The YOLO Object detection part
        #skip_yolo = True
        if not skip_yolo:
            #print("Yolo")
            #frame, boxes, confidences, class_ids, match = self.scan(frame,False,self.configs.is_video)
            frame, boxes, confidences, class_ids, match = self.scan(frame)
            if match:
                # if we have a yolo match we can skip
                # motion detect
                self.skip_motion       = True
                self.has_motion        = True
                self.skip_motion_count = 0
                self.record_after      = 20
            else:
                self.skip_motion = False
        else:
            #print("Yolo skipped")
            self.current_bboxes = False

        self.record_after-=1
        if self.record_after < 0 or self.is_first_run:
            self.record_after = 0

        self.previous_frame = gray

        # Emit the frames to the recorder
        # If we writeout the full stream
        # The recorder does this after recieving the frames
        #if self.configs.writeout:
        self.frametransfer.emit(frame,frame_raw)

        # Writeout movement
        if self.configs.webcam_mode and not self.is_first_run and (self.skip_motion or self.has_motion or self.record_after > 0):
            self.monitoring.emit(1)

        # Draw frame to window
        self.image_data_slot_direct(frame)

        self.is_first_run = False

        self.is_processing = False




# Just a simple implementation of
# Emitter functionality
# use a.on("event_id",fn) to add a callback
# a.emit("event_id",args) to call the stack
# off function not tested. COuld be buggy if same callback is added mutliple times
# so we only accept callbacks that are not in the stack within the on() function
class Emitter(object):

    def __init__(self):
        self.callbacks       = {}

    # Add a callback to the stack
    def on(self,event,fn):
        if not event in self.callbacks:
            self.callbacks[event] = set()

        if not fn in self.callbacks[event]:
            print("adding callback",event)
            self.callbacks[event].add(fn)
    # Remove a callback from the stack
    def off(self,event,fn):
        if not event in self.callbacks:
            return
        if fn in self.callbacks[event]:
            self.callbacks[event].remove(element)
    # Call the stack
    def emit(self,event,args=None):
        callbacks = self.callbacks[event]
        for fn in callbacks:
            print("calling callback",event)
            self.create_timer(0,fn,args)
    # Note: this class only extends other classes
    # self.terminate_event is set in the parent class
    def create_timer(self,i,cb,args=None):
        if self.terminate_event: # terminate_event switch of other class
            print("Emitter timer blocked")
            return False
        timer = threading.Timer(i, cb, args)
        timer.start()
        return timer



class socket_on_connect(Emitter):
    def __init__(self,client_socket,addr):
        super(socket_on_connect, self).__init__()

        #super(socket_on_connect,self).__init__(Emitter)
        print("Recieved socket connection")
        self.heartbeat_interval = 5 # seconds
        self.client_socket    = client_socket
        self.addr             = addr
        self.terminate_event  = False
        self.data             = b""
        self.payload_size     = struct.calcsize(">2L")
        self.valid_session    = False
        self.callbacks        = {}
        self.uuid             = str(uuid.uuid4())
        self.key              = "0ecc39d6e5dc8d794ebc19a77863263611f05ebcf402cb0f7e430707347cc457698b5515e18ee61e9038b2c8148c00e0d728098a09f811bc2de06febd8a9adbf"

    def get_uuid(self):
        return self.uuid

    def destroy(self):
        if self.terminate_event:
            return
        try:
            self.client_socket.close()
        except:
            print("SOCKERR::0001")
        
        # The Emitter uses timer that will not fire if terminate_event = True
        # We need to emit the destroy event before we set the termination flag
        self.emit("destroy",[self.uuid,])
        self.terminate_event = True


    def create_session(self):
        print("Creating socket session")
        data = self.client_socket.recv(1024)
        cprint(dicts.get("app_66"))
        data = data.decode('utf-8')
        # The key recieved seems valid
        # and matches our key here
        if data == self.key:
            cprint(dicts.get("app_64"))
            cprint(data,"WARNING")
            # send the key to the client
            self.client_socket.send(self.key.encode('utf-8'))
            # activate session
            self.valid_session = True
            # start the loop to recieve data
            self.create_timer(0, self.start_stream_loop)
            self.send_heartbeat()
        else:
            # Key not valid
            cprint(dicts.get("app_65"))
            cprint(data,"FAIL")
            self.valid_session = False

    # Magic happens here
    # Waiting for the stream
    # Grab the stream, create the frame
    # Starts the loop used to recieve data over the socket
    def start_stream_loop(self):
        cprint("Starting stream loop")
        # The main loop that grabs the frame
        # from the remote clients stream
        while self.valid_session and not self.terminate_event:
            skipped = self.socket_recieve()
            if skipped:
                self.valid_session = False
                print("Stream loop interupted")
                break
        print("Lost socket connection")
        self.destroy()

    # Get the payload
    # Extract the size of the packet
    # And finally load the packet and pass it to a callback
    # It returns True if we skipped
    # due to invalid session
    # else it returns false
    def socket_recieve(self):

        # First we want the metadata like datasize and callback_id
        packet, skip    = self.socket_get_packet( self.payload_size )
        # if we lost socket connection in the loop we can exit 
        # the loop and restart all the stuff
        if skip:
            #print("skipped 1")
            return True

        try:
            #cprint("Recv: {}".format(len(self.data)))
            result          = struct.unpack(">2L", packet)
            packet_size     = result[0]
            callback_id     = result[1]
            #cprint(dicts.get("app_63")+str(packet_size))
        except:
            print("SOCKERR0007")
            return False
        
        # Now we want the real data
        packet, skip = self.socket_get_packet( packet_size )
        if skip:
            #print("skipped 2")
            return True

        # Finally fire the callback
        if type(callback_id) != int:
            print("Invalid type for callback_id")
            return
        self.emit(callback_id,[packet,])

        return skip

    def send_heartbeat(self):
        if self.terminate_event:
            return
        self.socket_send("1",1)
        self.create_timer(self.heartbeat_interval,self.send_heartbeat)


    # Loop until we have the full packet
    # @return packet, (bool) skip
    # if skipped due to invalid session
    # it return true for skip
    def socket_get_packet(self,packet_size,buffsize=4096):
        skip = False

        if packet_size > 500000:
            cprint("Packetsize too large","FAIL")
            return False, True

        while self.valid_session and not self.terminate_event and len( self.data ) < packet_size:
            try:
                self.data += self.client_socket.recv(buffsize)
                #self.socket_send("HelloCLient".encode(),0)
                if self.terminate_event or not self.valid_session:
                    skip = True
                    cprint("Connection interupted")
                    break
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                cprint("SOCKERR0008","WARNING")
                skip = True
                break

        if not self.valid_session or self.terminate_event:
            skip = True

        if skip:
            return False, True

        package    = self.data[:packet_size]
        self.data  = self.data[packet_size:]

        return package, skip



    # Send a packet through the socket
    # It can handle str and byte data
    # dkey = the id of the callback we want to trigger
    # on the other side
    # todo encrypt the byte data
    def socket_send(self,data,dkey=0):
        if type( data ) == str:
            data = data.encode('utf-8')
        try:
            #data = crypt.encrypt(data)
            pack = struct.pack('>2L', len(data), dkey )
        except:
            print("Struc pack error")
        #cprint(dicts.get("app_63")+str(len(data)))
        try:
            self.client_socket.sendall( pack + data )
        except:
            self.valid_session = False



# Create the Server and listen for incoming connections
# It extends the recorder class
# Not standalone
class sock_server(object):

    # if we use a remote webcam this part
    # creates the server and socket stuff
    def setup_server(self):
        self.fps              = 10
        self.host             = "0.0.0.0"
        self.port             = 1338
        self.server           = None
        self.max_bind_retries = 10
        self.is_accepting     = False
        self.run_counter      = 0
        self.retry_delay      = 2
        self.server           = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.max_connections  = 2
        self.sockets          = {}
        self.bind()


    def remove_socket(self,uuid):
        try:
            del self.sockets[uuid]
        except:
            print("Couldn't remove socket. Socket not found",uuid)


    def sock_send(self,data,dkey=0):
        if self.terminate_event:
            return

        keys = self.sockets.copy().keys()
        for key in keys:
            try:
                self.sockets[key].socket_send(data,dkey)
            except:
                continue


    # called on shutdown
    def close_sockets(self):

        keys = self.sockets.copy().keys()
        for key in keys:
            try:
                self.sockets[key].destroy()
            except:
                continue

    def shutdown(self):
        self.close_sockets()



    # Bind the server
    def bind(self, current_try=0):
        try:
            cprint( dicts.get("app_67")+str(self.port) )
            cprint( dicts.get("app_68")+str(current_try) )
            self.server.bind((self.host, self.port))
            self.server.listen(self.max_connections)
        except:
            cprint("socket binding error")
            if current_try < self.max_bind_retries: 
                cprint("Binding retry")
                self.bind(current_try + 1)


    # Wait for connection
    # Only 1 connection is accepted
    def accept(self):

        #self.close_sockets()

        while not self.terminate_event:

            try:
                cprint( dicts.get("app_71"), "WARNING" )
                # Wait for incoming connections
                client_socket, addr = self.server.accept()
                # Create the instance for this connection
                a = socket_on_connect( client_socket, addr )
                # Connect the callbacks
                # Callbacks with string key can only be accessed from here
                # Whatever the client will send as callbackid: Only int is accepted from clients
                a.on("destroy",self.remove_socket)
                a.on(10,self.emit_stream_image) 
                uuid = a.get_uuid()
                self.sockets[uuid]  = a
                # Finnaly start and validate the session
                self.create_timer(0,a.create_session)

            except:
                #raise()
                cprint("socket accepting error","WARNING")
                #time.sleep( self.retry_delay )
                #self.create_timer(0,self.accept)


    # Convert streamed frame and emit it to the detector
    def emit_stream_image(self,data):
        try:
            if not data:
                return
            frame = pickle.loads(data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            self.emit_image(frame)
        except:
            return













# Class used only to grab our frame from videosource, image or cam
# It emits the image
#
#
#  _____                        _           
# |  __ \                      | |          
# | |__) |___  ___ ___  _ __ __| | ___ _ __ 
# |  _  // _ \/ __/ _ \| '__/ _` |/ _ \ '__|
# | | \ \  __/ (_| (_) | | | (_| |  __/ |   
# |_|  \_\___|\___\___/|_|  \__,_|\___|_|                                              
#
#                                           
class RecordVideo(QtCore.QObject, sock_server):
    image_data    = QtCore.pyqtSignal(np.ndarray)
    destroy_event = QtCore.pyqtSignal()

    def __init__( self, configs=None, parent=None ):
        super(RecordVideo,self).__init__(parent)
        cprint("RecordVideo init","WARNING")
        self.configs                 = configs
        self.video_writer            = False
        self.monitor_writer          = False
        self.previous_frame          = None
        self.tmp_frame               = None # only used to create the initial frame
        self.count                   = 0    # only used to create the initial frame
        self.emitter_timer           = False
        self.terminate_event         = False
        self.frames_emitted          = 0

        self.frame_emitter_interval = QtCore.QTimer(self)

        # Part used for webcam- and videomode
        if not self.configs.is_image:

            if self.configs.remote_video:
                self.setup_server()
                self.fps = 15
            else:
                self.camera = cv2.VideoCapture(self.configs.camera_port)
                self.fps    = int( self.camera.get(cv2.CAP_PROP_FPS) )
                #print("FPS: ",self.fps)
                # To speed up stuff 
                if self.configs.webcam_mode:
                    # To speed up stuff 
                    self.fps = int(self.fps/2)
                    cprint(dicts.get("app_79"),"WARNING")

                if self.configs.extraction_mode:
                    self.fps = int( 1000 )
                else:
                    self.camera.set(cv2.CAP_PROP_FPS,int(self.fps))


            #self.fps = int(self.fps/2)

            cprint(str(self.fps)+" "+dicts.get("app_78"),"WARNING")


            self.set_size()

    def set_current_frames(self,frame,frame_raw):
        self.current_frame     = frame
        self.current_frame_raw = frame_raw
        self.record()


    def create_timer(self,i,cb,args=None):
        if self.terminate_event:
            print("RecordVideo timer blocked")
            return False
        timer = threading.Timer(i, cb, args)
        timer.start()
        return timer

    # Emit the frame to our detector
    def emit_image(self,frame=None):
        try:
            if self.terminate_event:
                return
            if frame is not None:
                return self.image_data.emit(frame)
            read, frame = self.camera.read()
            if read:
                if not self.configs.extraction_mode and not self.create_initial_frame( frame ):
                    return
                self.frames_emitted+=1
                #print("Recorder emitted "+str(self.frames_emitted)+" frame")
                self.image_data.emit(frame)
            else:
                if self.configs.is_video:
                    self.destroy_event.emit()

                cprint("No frame recieved","WARNING")
        except:
            print("EMITIMAGEERR00001")

    def interval_transmitter(self):
        if self.fps <= 0:
            cprint(dicts.get(89),"FAIL")
            self.destroy_event.emit()
            return

        interval = 1000/self.fps
        self.frame_emitter_interval.setInterval( int(interval) )
        self.frame_emitter_interval.timeout.connect(self.emit_image) 
        self.frame_emitter_interval.start()
        #self.emit_image()
        #self.emitter_timer = self.create_timer(1/self.fps,self.interval_transmitter)

    # starts delivering frames from cam or video
    def start_recording(self):
        cprint(dicts.get("app_82"))
        if self.configs.remote_video:
            self.create_timer(0.5,self.accept)
            return

        self.interval_transmitter()
        #self.create_timer(1/self.fps,self.interval_transmitter)

    # Set the size of the webcam capture
    def set_size(self):
        if self.configs.remote_video:
            return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.configs.cap_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.configs.cap_height)

    def on_error(self,msg_key,text=False):
        cprint(dicts.get(msg_key),"FAIL")
        if text:
            cprint(text,"FAIL")
        cprint(dicts.get("app_1"),"FAIL")
        raise SystemExit(0)
    
    def destroy(self):
        cprint("RecordVideo destroy called")
        self.terminate_event = True
        try:
            self.frame_emitter_interval.stop()
        except:
            print("No emitter timer to cancel")
        if self.configs.remote_video:
            self.shutdown() # closing all sockets etc
            self.server.shutdown(socket.SHUT_RDWR) 
            self.server.close()
        

    # save the raw frame without bounding boxes etc
    # Used as callback for the savebutton
    # so we can easily extract frames with undetected objects
    # and label them later
    def save(self):
        self.create_timer(0,filehandler.save_image,[self.current_frame_raw, self.configs.frame_folder,self.configs.file_name])
        #filehandler.save_image( self.current_frame_raw, self.configs.frame_folder, self.configs.file_name )

    # Create folder if needed
    # and setup the specific recorder
    def prepare_recorder(self,root_folder,w,h):
        print("Creating recorder")
        folder     = time.strftime("%Y-%m-%d")
        video_name = time.strftime("%H-%M-%S")+"_"+str(round(time.time() * 1000))+"."+self.configs.video_extension
        file_path  = root_folder+folder+"/"+video_name
        filehandler.create_folder(root_folder)
        filehandler.create_folder(root_folder+folder)
        return cv2.VideoWriter( file_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, ( ( w, h ) ) )

    # Record frame
    def record(self):
        if not self.configs.extraction_mode and self.configs.writeout and not self.configs.is_image:

            if self.video_writer == False:
                # We wait until we now the exact size of the given frames and setup the recorder on the fly
                # If writeout is True we save the full video
                try:
                    h, w, _ = self.current_frame.shape
                except:
                    return
                if self.configs.writeout:
                    cprint(dicts.get("app_53"),"WARNING")
                    self.video_writer = self.prepare_recorder( self.configs.video_folder, w, h )
            # Write the frame to the video
            self.video_writer.write(self.current_frame)


    # Record motion
    def monitor_record(self):
        if not self.configs.extraction_mode and self.configs.monitor and not self.configs.is_image:
            if self.monitor_writer == False:
                # We wait until we now the exact size of the given frames and setup the recorder on the fly
                # If monitor is True we save the motions
                try:
                    h, w, _ = self.current_frame.shape
                except:
                    return
                cprint(dicts.get("app_54"),"WARNING")
                self.monitor_writer = self.prepare_recorder( self.configs.monitor_folder, w, h )
            # Write the frame to the video
            self.monitor_writer.write(self.current_frame)


    def create_initial_frame(self,frame):

        #
        # create the initial frame
        #
        if self.previous_frame is None:
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray      = cv2.GaussianBlur(gray, (21, 21), 0)
            if self.count > 0 and (self.count % 2) == 0:
                cprint( dicts.get("app_3") )
            if self.tmp_frame is None:
                self.tmp_frame = gray
                return False
            else:
                time.sleep(1)
                self.previous_frame = gray
                #return True
                delta          = cv2.absdiff(self.tmp_frame, gray)
                self.tmp_frame = gray
                tst            = cv2.dilate(cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
                if self.count > 30:
                    cprint( dicts.get("app_2") )
                    if not cv2.countNonZero(tst) > 0:
                        self.previous_frame = gray
                    else:
                        return False
                else:
                    self.count += 1
                    return False
        return True







# Controller for the physics
# Movements etc.
#
#
#  _____  _               _          
# |  __ \| |             (_)         
# | |__) | |__  _   _ ___ _  ___ ___ 
# |  ___/| '_ \| | | / __| |/ __/ __|
# | |    | | | | |_| \__ \ | (__\__ \
# |_|    |_| |_|\__, |___/_|\___|___/
#                __/ |               
#               |___/                
#
#
class physics(object):
    def __init__(self,parent):
        cprint("physics init","WARNING")
        self.parent = parent
        self.degrees_hori      = 180
        self.degrees_vert      = 180
        self.current_pos_x     = 90
        self.current_pos_y     = 90
        self.distance_x_left   = 0
        self.distance_x_right  = 0
        self.distance_y_top    = 0
        self.distance_y_bottom = 0
        self.xstep             = False
        self.ystep             = False
        self.frame_width       = False
        self.frame_height      = False
        self.event_id          = 2
        self.callback          = self.default_callback

    def default_callback(self,data,id):
        print(data)

    def set_callback(self,cb):
        self.callback = cb

    # Callback triggered if the move up button is pressed
    def up(self):
        if self.current_pos_y <= 0:
            return
        self.current_pos_y -= 1
        self.callback( "Y:"+str(self.current_pos_y), self.event_id )


    # Callback triggered if the move down button is pressed
    def down(self):
        if self.current_pos_y >= self.degrees_vert:
            return
        self.current_pos_y += 1
        self.callback( "Y:"+str(self.current_pos_y), self.event_id )

    # Callback triggered if the move left button is pressed
    def left(self):
        if self.current_pos_x <= 0:
            return
        self.current_pos_x -= 1
        self.callback( "X:"+str(self.current_pos_x), self.event_id )

    # Callback triggered if the move right button is pressed
    def right(self):
        if self.current_pos_x >= self.degrees_hori:
            return
        self.current_pos_x += 1
        self.callback( "X:"+str(self.current_pos_x), self.event_id )

    def get_steps(self,frame):
        if self.xstep and self.ystep:
            return (self.xstep,self.ystep)
        h, w, _ = frame.shape
        return self.set_steps(w,h)

    def set_steps(self,w,h):
        self.frame_width  = w
        self.frame_height = h
        self.xstep = (w-(self.distance_x_left+self.distance_x_right))/self.degrees_hori
        self.ystep = (h-(self.distance_y_bottom+self.distance_y_top))/self.degrees_vert
        return (self.xstep,self.ystep)

    def draw_position(self,frame):
        co           = (255,255,255)
        co2          = (255,255,255)
        co3          = (152,255,51)
        xstep, ystep = self.get_steps(frame)
        return
        cv2.line(frame,(ca-5, 10),(int(ca+5), 10),co,1)
        cv2.line(frame,(w-10, ca-5),(w-10, int(ca+5)),co,1)












# Configuration that is shared 
# among all the widgets and classes
# Takes the args from the argparser
# and builds the configs
class shared_configs(object):
    def __init__(self,args):
        self.detector_class_ids      = (0,) # the class_ids we use, default yolo 0 = person, in our custom yolo 0 = head, we can use multiple ids (0,3,4,42)
        self.camera_port             = 0
        self.cap_width               = args.display_width
        self.cap_height              = args.display_height
        self.video_extension         = "avi"
        self.writeout                = args.writeout
        self.monitor                 = args.monitor
        self.loader_image            = "./artwork/loader.jpg"
        self.image_folder            = "./output/images/"
        self.frame_folder            = "./output/frames/"
        self.video_folder            = "./output/video/"
        self.monitor_folder          = "./output/monitor/"
        self.webcam_folder           = "./output/webcam/"
        self.labelfiles_folder       = "./output/labels_tmp/"
        self.valid_video_extenions   = [".avi",".AVI",".mpeg",".MPEG",".mp4",".MP4"]
        self.valid_image_extenions   = [".jpg",".JPG",".jpeg",".JPEG",".png",".PNG"]
        self.is_image                = False
        self.file_name               = False
        self.is_video                = False
        self.is_folder               = False
        self.previous_frame          = None
        self.webcam_mode             = False
        self.has_physics             = False
        self.remote_video            = args.remote
        self.use_cuda                = args.use_cuda
        self.scan_folder             = False
        self.extraction_mode         = args.extract
        self.video_stream_only       = args.stream_only
        self.extraction_number       = 50
        self.use_deepsort            = False

        if self.video_stream_only:
            cprint('Simple Videostreammode activated')

        if args.dir:
            cprint("Got Directory")
            self.is_image     = True
            self.scan_folder  = args.dir
            if not self.scan_folder.endswith(os.sep):
                self.scan_folder = self.scan_folder+os.sep

        # If we supply an inputfile we scan it for objects 
        # Videos and Images are supported
        if args.file:
            cprint("Got File")
            self.camera_port   = args.file
            name, ext          = filehandler.get_file_name(args.file)
            self.file_name     = name+ext
            self.file_name_raw = name
            if ext in self.valid_image_extenions:
                self.is_image   = True
                self.image_file = args.file
            elif ext in self.valid_video_extenions:
                self.is_video = True

                if self.extraction_mode:
                    cprint(dicts.get(88))

            else:
                return self.on_error("app_27",args.file)
            if not os.path.isfile(args.file):
                return self.on_error("app_30",args.file)
        else:
            self.video_folder = self.webcam_folder

        if not self.is_image:
            if not args.file and not args.dir:
                self.webcam_mode = True
                self.has_physics = True


    def on_error(self,msg_key,text=False):
        cprint(dicts.get(msg_key),"FAIL")
        if text:
            cprint(text,"FAIL")
        cprint(dicts.get("app_1"),"FAIL")
        raise SystemExit(0)












# class with tools to handle files and folder
# write files, create folder etc.
#
#
#  ______ _ _      _                     _ _           
# |  ____(_) |    | |                   | | |          
# | |__   _| | ___| |__   __ _ _ __   __| | | ___ _ __ 
# |  __| | | |/ _ \ '_ \ / _` | '_ \ / _` | |/ _ \ '__|
# | |    | | |  __/ | | | (_| | | | | (_| | |  __/ |   
# |_|    |_|_|\___|_| |_|\__,_|_| |_|\__,_|_|\___|_|   
#                                                      
#
class file_handler(object):
    def __init__(self):
        self.init  = True
        self.count = 0

    # returs name and extension of given file
    def get_file_name(self,path):
        file            = os.path.basename(path)
        name, extension = os.path.splitext(file)
        return (name,extension)

    # creates a folder if needed
    def create_folder(self,folder):
        if not os.path.exists(folder):
            cprint( dicts.get("app_20")+folder)
            os.makedirs(folder)

    # saves given image file
    def save_image(self,frame,image_folder,file_name=False):
        if not file_name:
            file_name = "capture.jpg"
        self.count+=1
        name, ext  = self.get_file_name(file_name)
        folder     = time.strftime("%Y-%m-%d")
        nname      = str(round(time.time() * 1000))+"_"+str(self.count)+".jpg"
        file_path  = image_folder+folder+"/"+nname
        self.create_folder(image_folder+folder)
        cprint(dicts.get("app_21")+file_path)
        cv2.imwrite( file_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100] );

    # creates file if needed
    # or overrides existing file
    def override(self,path,text):
        cprint(dicts.get("app_52")+path,"WARNING")
        if os.path.isfile(path):
            cprint("File exists: "+path,"FAIL")
            return
        with open(path, 'w+') as fh:
            fh.seek(0)
            fh.write(text)
            fh.truncate()


filehandler = file_handler()















#
#                                                    
#     /\                                             
#    /  \   _ __ __ _ _ __   __ _ _ __ ___  ___ _ __ 
#   / /\ \ | '__/ _` | '_ \ / _` | '__/ __|/ _ \ '__|
#  / ____ \| | | (_| | |_) | (_| | |  \__ \  __/ |   
# /_/    \_\_|  \__, | .__/ \__,_|_|  |___/\___|_|   
#                __/ | |                             
#               |___/|_|                             
#
#
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f",         type=str,  default=False, help=dicts.get("app_37"))
    parser.add_argument("--show", "-s",         type=bool, default=True,  help=dicts.get("app_39"))
    parser.add_argument("--dir",  "-d",         type=str,  default=False, help=dicts.get("app_50"))
    parser.add_argument("--remote",      dest="remote",      action="store_true", default=False)
    parser.add_argument("--stream_only", dest="stream_only", action="store_true", default=False)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--extract", dest="extract", action="store_true", default=False)
    parser.add_argument("--writeout", dest="writeout", action="store_true", default=False)
    parser.add_argument("--monitor", dest="monitor", action="store_true", default=False)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    #parser.add_argument("--config_detection", type=str, default="./configs/yolov3_tiny.yaml")
    parser.add_argument("--config_detection_head", type=str, default="./configs/yolov3-single-head-512.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display_width", type=int, default=1280)
    parser.add_argument("--display_height", type=int, default=720)
    return parser.parse_args()





# Usage:
#
#
# If we only want a video_stream with no further processing:
#     python3 main.py --stream_only
#
#
# Simply tag objects on an image:
#     python3 main.py --file '/path/file.jpg' [image or video]
#
# IP-Cam Mode [stream images from webcam to server]
# On the PC with the cam
#     python3 stream_client.py
# On the PC we use for controling and processing the stream
#     python3 main.py --remote
#
#
# It is possible to create a video of the full webcamstream
# just add --writeout to the command


# Monitor mode
#
#
#



# We can use this app to extract frames from videofiles (not the whole video just every nth frame)
# At the moment the default setting is every 50th frame
# This can be changed at "self.extraction_number = 50"
#
# Those frames can then be used for training purposes
# To do so, just use a command like this:
#
# python3 main.py --file '/path/to/videofile' --extract


# Label folders with images
#
# For training we need images with the labelfiles
# To speed up the process we can run this app over a full folder with images
# It will grab each image an create a labelfile with the matches
# The result still will need to get verified by hand but it really speeds up the work
# 
# Use this command to label a full folder:
#
# python3 main.py --dir '/path/to/imagefolder'
# The resulting labels will be stored at ./output/labels_tmp/



# Output files will be saved under ./output/specific_folder
#
# (monitor mode) If we are in monitor mode the videos from movements are stored in ./output/monitor
# (frame extraction mode) If we axtract frames from videos, the frames are stored in ./output/frames
# (tag images) If we simple tag matches in an image the tagged file is stored under ./output/images
# (tag video) If we simple tag matches in a videofile the result is stored under ./output/videos
# (label dir) The resulting labels will be stored at ./output/labels_tmp/

# If no Cuda is installed he app will switch to really slow CPU processing
# It is possible to force cpu usage for whatever reason. Just add --cpu to the command





def main_process( conn, args ):
    app      = QtWidgets.QApplication(sys.argv)
    main_app = MainWindow(conn,args)
    main_app.run()
    app.exec_()



#
#  _____       _ _   
# |_   _|     (_) |  
#   | |  _ __  _| |_ 
#   | | | '_ \| | __|
# |_____|_| |_|_|\__|
#                    
#                    
if __name__=="__main__":

    # load the dictionary
    dicts = dicts("de")
    # parse arguments
    args  = parse_args()
    # create the detector configs
    cfg   = get_config()
    cfg.merge_from_file(args.config_detection)

    cfg.merge_from_file(args.config_detection_head)

    cfg.merge_from_file(args.config_deepsort)
    # Display header
    cprint( get_header(), "HEADER" )
    # Show the language loaded
    dicts.print_result()
    #print(cv2.getBuildInformation())
    parent_conn, child_conn = Pipe()

    proc =  [
        Process(target=main_process, name = ' __MAIN__',   args=( parent_conn, args)),
        #Process(target=controller,   name = ' __REMOTE__', args=( child_conn,  args))
    ]

    # starting processes
    for p in proc:
        #p.daemon = True
        p.start()


    stop = False
    while not stop:
        for p in proc:
            if not p.is_alive():
                stop = True
                break

    for p in proc:
        p.terminate()
        #p.join()

    cprint(dicts.get("app_1"),"FAIL")

    sys.exit(1)


    #os._exit(1)
