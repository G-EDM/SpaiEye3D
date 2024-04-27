import argparse
import socket
import os
import subprocess
import sys
import re
import time
import cv2
import threading
import struct
import pickle
import requests

from multiprocessing import Process, Event, Pipe
from utils.tools import *
from language_pack.language import dictionary as dicts


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
    #http://your-url.com/config.json"
    # Load a json file from a remote source that contains the connection details from the server we want to reverse connect to
    # If we dont supply a file link the app uses a default config
    # Default: basically means we use client and server on the same mashine
        #self.host = "0.0.0.0"
        #self.port = 1338

    parser.add_argument("--remote_config", type=str, default=None)
    return parser.parse_args()






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
class RecordVideo(object):
    camera            = None
    fps               = 10
    cap_width         = 640#1280
    cap_height        = 480#720
    camera_port       = 0
    timer             = False
    img_counter       = 0
    timers            = set()
    block_emitting    = True

    def __init__(self, conn=None, args=None):
        #super(RecordVideo,self).__init__(parent)
        cprint("RecordVideo init","WARNING")
        self.conn    = conn
        self.run()

    def run(self):
        self.camera   = cv2.VideoCapture(self.camera_port)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        self.fps      = self.camera.get(cv2.CAP_PROP_FPS)
        print("FPS: ",self.fps)
        # speed up
        self.interval = float(1/self.fps)
        self.start_timer()
        self.create_timer(0,self.start_conn_loop)

    def on_error(self,msg_key,text=False):
        cprint(dicts.get(msg_key),"FAIL")
        if text:
            cprint(text,"FAIL")
        cprint(dicts.get("app_1"),"FAIL")
        raise SystemExit(0)

    def start_timer(self):
        if self.timer:
            try:
                self.timer.cancel()
            except:
                a=1
        self.timer = threading.Timer(self.interval, self.emit_image)
        self.timer.start()

    def emit_image(self):
        if self.block_emitting:
            #print("Not emitting")
            return self.start_timer()
        read, frame = self.camera.read()
        if read:
            #is_success, im_buf_arr = cv2.imencode(".jpg", frame)
            #byte_im = im_buf_arr.tobytes()
            #self.conn.send( byte_im )
            result, frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            data = pickle.dumps(frame, 0)
            self.conn.send( data )
            self.img_counter += 1
        self.start_timer()

    def create_timer(self,t,cb):
        timer = threading.Timer(t, cb)
        timer.start()


    # Takes the imagedata and sends it to the server
    def start_conn_loop(self):
        cprint( dicts.get("app_70"), "WARNING" )
        while True:
            data = self.conn.recv()

            if data == "Start":
                self.block_emitting = False
                cprint(dicts.get("app_61"),"WARNING")
            elif data == "Stop":
                self.block_emitting = True
                cprint(dicts.get("app_62"),"WARNING")



# Reverse Connection
# Try to connect to the server
# Validate connection
# Start stream
class ReverseStreamClient(object):

    def __init__(self,conn=None,args=None):
        #super(ReverseStreamClient, self).__init__()
        self.recorder_pipe          = conn
        self.allow_connection       = False
        self.socket                 = None
        self.host                   = None
        self.port                   = None
        self.retry_delay            = 2
        self.block                  = True 
        self.socket_fail            = False
        self.is_connecting          = False
        self.valid_session          = False
        self.reload_config_interval = 500 # reload remote config every n seconds
        self.terminate_event        = False
        self.load_config_count      = 0
        self.has_configs            = False
        self.remote_config          = args.remote_config

        # Simple validation
        # We send this key to server and the server send it to the client
        # If no key matches we allow the connection
        # Warning! No SSL at this time
        # Imagestream and key are send plaintext
        # Man in the Middle attackers can capture it
        self.key   = "0ecc39d6e5dc8d794ebc19a77863263611f05ebcf402cb0f7e430707347cc457698b5515e18ee61e9038b2c8148c00e0d728098a09f811bc2de06febd8a9adbf"

        if not self.remote_config:
            self.set_default_config()
            cb = self.run
        else:
            #self.set_default_config()
            #cb = self.run
            #self.remoteconfigfile = "http://makergrube.com/static/data/180320201845TESTOBJECTtestObJEcT.json"
            cb = self.load_remote_file

        self.create_timer(5,cb)

    def destroy(self):
        cprint("ReverseStreamClient destroy called")
        self.terminate_event = True


    # Set the default configs here
    # If a remote config file is used
    # this part can be left as it is
    # to use a remote file start the script with --r flag
    # ( "python stream_client.py --r" )
    def set_default_config(self):
        # the server adress we want to send the images to
        self.host = "0.0.0.0"
        # the port we want to connect to
        self.port = 1338
        return False

    def set_remote_config(self,data):
        changed = False
        if self.host != data.server_ip:
            changed = True
        if self.port != int(data.server_port):
            changed = True

        if not changed:
            return False

        else:
            cprint(dicts.get("app_58"),"WARNING")
            self.host = data.server_ip
            self.port = int(data.server_port)
            print("\n")
            cprint("Server: "+str(data.server_ip))
            cprint("Port: "+str(data.server_port))
            print("\n")
            return True

    # Loading the configs for our server from remote source
    def load_remote_file(self):
        self.load_config_count+=1
        cprint( dicts.get("app_56")+" - "+str(self.load_config_count) )
        cprint( self.remote_config, "WARNING" )
        try:
            resp = requests.get( self.remote_config )
            data = resp.json()
            changed = self.set_remote_config( jpObject(data) )
            if changed:
                self.refresh()
                self.create_timer(2,self.run)
            # we reload the configs every now and then
            # so we can react to changes
            self.create_timer(self.reload_config_interval, self.load_remote_file)
        except:
            cprint( dicts.get("app_57") )
            self.create_timer(5, self.load_remote_file)


    def create_timer(self,i,cb,args=None):
        if self.terminate_event:
            print("MainWidget timer blocked")
            return False
        timer = threading.Timer(i, cb, args)
        timer.start()
        return timer

    def connect(self):

        if self.is_connecting:
            return

        self.is_connecting = True

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))

            self.socket.send(self.key.encode('utf-8'))
            key = self.socket.recv(1024)
            if key.decode('utf-8') == self.key:
                cprint( dicts.get("app_60") )
                self.is_connecting  = False
                self.valid_session = True
                time.sleep(1)
                self.block   = False
                self.create_timer(0,self.start_stream_loop)
                self.start_conn_loop()
            else:
                self.is_connecting  = False
                self.block   = True
                self.valid_session = False
                cprint( dicts.get("app_59") )
                self.create_timer(self.retry_delay,self.connect)

        except socket.error as msg:
            self.socket.close()
            self.is_connecting  = False
            self.block   = True
            self.valid_session = False
            cprint( dicts.get("app_59") )
            self.create_timer(self.retry_delay,self.connect)


    def run(self):
        self.socket_fail    = False
        self.valid_session = False
        self.block   = True
        self.resetting      = False
        if not self.is_connecting:
            self.create_timer(1,self.connect)

    def refresh(self):
        self.resetting = True
        try:
            self.socket.close()
        except:
            a=1



    def printer(self,data):
        print(data)



    # Starts the loop used to recieve data over the socket
    def start_stream_loop(self):
        print("Starting stream loop")
        # The main loop that grabs data from the server
        # we get over socket
        self.data         = b""
        self.payload_size = struct.calcsize(">2L")
        while self.valid_session and not self.terminate_event:
            skipped = self.socket_recieve()
            if skipped:
                self.valid_session = False
                #print("Reset connection")
                break
        if self.terminate_event:
            return
        print("Lost connection. Streamloop stopped")
        #self.run_stream()

    # We map the callbacks here
    # Data recieved over socket comes
    # with a callback_id like 0,1...
    # we map this id to a callback here
    # Note: Data ist suppled as byte type here
    def callback_map(self,package,callback_id):
        callback_id = int(callback_id)
        callback    = False

        #print(package.decode("utf-8"))

        if callback_id == 0:
            # 0 just print stuff for
            # testing purposes
            callback = self.printer

        elif callback_id == 1:
            # just a heartbeat
            print("Heartbeat recieved")

        elif callback_id == 2:
            # 2 physics movement
            print( package )

        if callback == False or package == False:
            return

        self.create_timer(0,callback,[package,])

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
            print("err 5")
            return False
        
        # Now we want the real data
        packet, skip = self.socket_get_packet( packet_size )
        if skip:
            #print("skipped 2")
            return True

        #print( type(packet))
        #r = crypt.decrypt(packet)
        #print(r)

        # Finally fire the callback
        self.callback_map(packet,callback_id)

        return skip

    # Loop until we have the full packet
    # @return packet, (bool) skip
    # if skipped due to invalid session
    # it return true for skip
    def socket_get_packet(self,packet_size,buffsize=4096):
        skip = False

        if packet_size > 500000:
            cprint("Packetsize too large","FAIL")
            return False, True

        while len( self.data ) < packet_size:
            try:
                self.data += self.socket.recv(buffsize)
                if self.terminate_event or not self.valid_session:
                    skip = True
                    print("no conn")
                    break
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print("socket_get_packet::err0003")
                skip = True
                break

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
    def socket_send(self,data,dkey):
        if type( data ) == str:
            data = data.encode('utf-8')
        try:
            pack = struct.pack('>2L', len(data), dkey )
        except:
            print("Struc pack error")
        #cprint(dicts.get("app_63")+str(len(data)))
        self.socket.sendall( pack + data )


















        
    # Takes the data and sends it to the server
    # The recorder sends the data through
    # the pipe to this loop. From there we send it over the 
    # socket to the server
    def start_conn_loop(self):

        init = False

        while True:

            if not init:
                time.sleep(1)
                self.recorder_pipe.send("Start")
                time.sleep(1)
                init = True

            if self.resetting:
                break

            if not self.valid_session:
                continue

            try:
                data = self.recorder_pipe.recv()
                if type(data) is bytes:
                    if self.block:
                        print("Frame blocked",size)
                        continue
                    try:
                        self.socket_send( data, 10 )
                        #self.socket_send( "Hello".encode(),2 )
                        self.socket_fail = False
                    except:
                        #raise
                        self.socket_fail = True
                        #print("Socket fail")
                        break

            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                #raise
                print("Conn loop fail")
                break

        self.recorder_pipe.send("Stop")

        if self.resetting:
            return

        self.run()



# The main app
# It starts the recorder 
# and the socket stuff
# It also keeps track of the processes
class main_app(object):
    def __init__(self,args=None):
        self.init = True
        self.proc = []
        self.args = args
        self.parent_conn, self.child_conn = Pipe()

    def run(self):
        self.add_socket()
        self.add_recorder()
        self.start_processes()
        self.exit_loop()

    def add_socket(self):
        s = Process(target=ReverseStreamClient, name = ' __REMOTE__', args=( self.child_conn,  self.args))
        self.proc.append( s )

    def add_recorder(self):
        recorder = Process(target=RecordVideo, name = ' __MAIN__',   args=( self.parent_conn, self.args))
        self.proc.append( recorder )

    def start_processes(self):
        for p in self.proc:
            p.daemon = True
            p.start()

    def destroy(self):
        for p in self.proc:
            p.terminate()


    def exit_loop(self):
        stop = False

        while not stop:
            for p in self.proc:
                try:
                    if not p.is_alive():
                        stop = True
                        break
                except:
                    stop = True
                    break


        self.destroy()

if __name__ == '__main__':
    # parse arguments
    args  = parse_args()
    dicts = dicts("de")
    app = main_app( args )
    app.run()
    sys.exit(1)