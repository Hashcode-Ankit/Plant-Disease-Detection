import os
import uuid
import flask
import urllib
from PIL import Image
import numpy as np
import joblib
import cv2
from flask import Flask , render_template  , request , send_file,Response
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = np.array(['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedindexar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'])

image_frame = []
@app.route('/')
def home():
        image_frame.clear()
        # camera.release()
        return render_template("index.html")
predictLive ="Not Detected"
def gen_frames(camera):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        image_frame.clear()
        image_frame.append(frame)
        if not success: 
            print("not going to take")
            break
        else:
            
            frame=cv2.flip(frame,1) 
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    camera=cv2.VideoCapture(0)
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/live')
def live() : 
    return render_template("liveDetection.html")

@app.route('/live', methods = ['GET' , 'POST'])
def livePred() :
    print("here is the image : ",image_frame)
    if(image_frame[-1]!=""): 
        img=cv2.resize(image_frame[-1],[72,72])
        image_frame.clear()
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        clf = joblib.load("model.pkl")
        prediction = clf.predict(img)
        print(np.argmax(prediction))
        predict_class = classes[np.argmax(prediction)]  
        return render_template("liveDetection.html",Output = predict_class)
    else :
        return render_template("liveDetection.html",Output = "Not Detected")


@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img=image.load_img(img_path,target_size=(72,72))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis = 0)
                clf = joblib.load("model.pkl")
                prediction = clf.predict(img)
                print(np.argmax(prediction))
                predict_class = classes[np.argmax(prediction)]
            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = filename , output = predict_class )
            else:
                return render_template('index.html' , error = error) 

            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
                img = image.load_img(img_path,target_size=(72,72))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis = 0)
                clf = joblib.load("model.pkl")
                prediction = clf.predict(img)
                # print(np.argmax(prediction))
                predict_class = classes[np.argmax(prediction)]
                print(img_path)
            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = file.filename, output = predict_class)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True, port=80)


