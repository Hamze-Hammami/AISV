from flask import Flask, render_template, Response
import cv2
import depthai as dai

pipeline = dai.Pipeline()


# Define source and output

camRgb = pipeline.create(dai.node.ColorCamera)

xoutRgb = pipeline.create(dai.node.XLinkOut)


xoutRgb.setStreamName("rgb")


# Properties

camRgb.setPreviewSize(300, 300)

camRgb.setInterleaved(False)

camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)


# Linking

camRgb.preview.link(xoutRgb.input)

app = Flask(__name__)

# Use /dev/video0 (default) or change if needed
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)  # V4L2 for Linux optimization
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Optimize resolution for performance
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30 for smoother streaming

def generate_frames():
    with dai.Device(pipeline) as device:
        print('Connected cameras:', device.getConnectedCameraFeatures())

        # Print out usb speed

        print('Usb speed:', device.getUsbSpeed().name)

        # Bootloader version

        if device.getBootloaderVersion() is not None:

            print('Bootloader version:', device.getBootloaderVersion())

        # Device name

        print('Device name:', device.getDeviceName(), ' Product name:', device.getProductName())


        # Output queue will be used to get the rgb frames from the output defined above

        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)


        while True:

            inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived


            # Retrieve 'bgr' (opencv format) frame

            ret, buffer = cv2.imencode('.jpg', inRgb.getCvFrame(), [cv2.IMWRITE_JPEG_QUALITY, 85])  # Optimize JPEG quality
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # while True:
    #     success, frame = camera.read()
    #     if not success:
    #         break
    #     else:
    #         ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Optimize JPEG quality
    #         frame = buffer.tobytes()
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
