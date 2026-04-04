import cv2
import threading
import time
from flask import Flask, render_template_string

app = Flask(__name__)
latest_frame = None
lock = threading.Lock()

def camera_worker():
    global latest_frame
    pipeline = (
        "udpsrc address=127.0.0.1 port=5600 ! "
        "application/x-rtp,payload=96 ! "
        "rtph264depay ! "
        "h264parse ! "
        "avdec_h264 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1"
    )

    while True:
        print("Starting GStreamer handshake...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if latest_frame is None:
                    print("!!! CONNECTION ESTABLISHED: VIDEO DETECTED !!!")
                with lock:
                    latest_frame = frame
            else:
                print("Stream sync lost, attempting reconnect...")
                break
        
        cap.release()
        time.sleep(1)

@app.route('/capture')
def capture():
    global latest_frame

    # Discard the current buffered frame and wait for a fresh one
    with lock:
        stale = latest_frame.copy() if latest_frame is not None else None

    # Wait for a new frame that is different from the stale one
    for _ in range(30):
        time.sleep(0.1)
        with lock:
            if latest_frame is not None:
                if stale is None or not (latest_frame == stale).all():
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(f"/userdata/{filename}", latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    return f"Captured! Saved as {filename}"

    return "Error: No fresh frame available. Check logs for connection status."

@app.route('/')
def index():
    status = "READY" if latest_frame is not None else "CONNECTING..."
    return render_template_string(f'''
        <h1>Status: {status}</h1>
        <a href="/capture"><button style="font-size:40px">CAPTURE IMAGE</button></a>
    ''')

if __name__ == '__main__':
    t = threading.Thread(target=camera_worker, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=8080)