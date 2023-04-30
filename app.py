

#作者：小约翰啊伟
#B站主页：https://space.bilibili.com/420694489
#源码下载：https://github.com/GodVvvWei/yolo-flask-html


from flask import Flask,render_template,Response
import cv2, time, ffmpeg, numpy as np, av

from models.experimental import attempt_load
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device

app = Flask(__name__)
#user, pwd, ip = "admin", "123456zaQ", "[192.168.100.196]"
user, pwd, ip = "admin", "123456zaQ", "[10.0.0.225]"
from camera_ready import detect


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流（海康摄像头）
        self.count = 0
        #self.video = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, 1))
        #self.video = cv2.VideoCapture("rtsp://10.0.0.225:8554/1")
        self.video = cv2.VideoCapture(0)
        #大华摄像头
        #self.video = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))

        self.weights, imgsz = 'yolov5s.pt', 640
        set_logging()
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        #print("half", self.half)
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        print("half", self.stride)
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()  # 视频帧
        image = detect(source=image,half=self.half,model=self.model,device=self.device,imgsz=self.imgsz,stride=self.stride)

        img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        ret, jpeg = cv2.imencode('.jpg', image, img_param)

        return jpeg.tobytes()


def rstp_flow():
    source = "rtsp://jiu:8554/1"
    args = {"rtsp_transport": "tcp"}    # 添加参数
    probe = ffmpeg.probe(source)
    cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    print("fps: {}".format(cap_info['r_frame_rate']))
    width = cap_info['width']           # 获取视频流的宽度
    height = cap_info['height']         # 获取视频流的高度
    up, down = str(cap_info['r_frame_rate']).split('/')
    fps = eval(up) / eval(down)
    print("fps: {}".format(fps))    # 读取可能会出错错误
    process1 = (
        ffmpeg
        .input(source, **args)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .overwrite_output()
        .run_async(pipe_stdout=True)
    )
    while True:
        in_bytes = process1.stdout.read(width * height * 3)     # 读取图片
        if not in_bytes:
            break
        # 转成ndarray
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        frame = cv2.resize(in_frame, (1280, 720))   # 改变图片尺寸
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转成BGR
        #cv2.imshow("ffmpeg", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        ret, frame_out = cv2.imencode('.jpg', frame, img_param)
        frame_out_encoded = frame_out.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_out_encoded + b'\r\n')
    process1.kill()             # 关闭


def pyav():
    # rtsp 是标准的海康威视3级子码流
    #video = av.open('rtsp://127.0.0.1:15556/1')
    video = av.open('rtsp://jiu:8554/1')
    print("format:", video.dumps_format())
    #video_context = video.streams.video[0].codec_context

    #container = av.open('test.mp4', mode='w')
    #stream = container.add_stream('h264', rate=video_context.framerate)
    #stream.width = video_context.width
    #stream.height = video_context.height
    #stream.pix_fmt = 'yuv420p'

    try:
        for packet in video.demux():
            for frame in packet.decode():
                if packet.stream.type == 'video':
                    print("frame = ", frame)

                    img = frame.to_ndarray(format='bgr24')
                    #frame2 = av.VideoFrame.from_ndarray(img, format='bgr24')
                    #for packet2 in stream.encode(frame2):
                    #    container.mux(packet2)
                    cv2.imshow("Video", img)
                    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    ret, frame_out = cv2.imencode('.jpg', img, img_param)
                    frame_out_encoded = frame_out.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_out_encoded + b'\r\n')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Flush stream
                #for packet2 in stream.encode():
                #    container.mux(packet2)
                # Close the file
                #container.close()
                break
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()


@app.route('/xyhaw')
def xyhaw():
    return render_template('xyhaw.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    mode = 1
    if mode == 0:
        return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    elif mode == 1:
        return Response(rstp_flow(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
