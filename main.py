import pymongo
from datetime import datetime
import numpy as np
import cv2
import face_recognition
import streamlit as st
 
 
st.header('Attendence using face recognisation system')
#client = MongoClient("mongodb+srv://himanshuparida191003:hi%40191003@projects.dvdnu49.mongodb.net/")
client = pymongo.MongoClient(st.secrets['mongo'])
db=client.Attendence
collection=db.time
Roll_no=db.Roll_no
cnt=0
while(True):
    placeholder = st.empty()
    if cnt==1 :
        break
    cnt=1
    a=st.text_input("Enter your roll no: ")
    if(a=="quit" or a=="QUIT"):
        break
    present=0
    if(a==""):
        st.write("Enter your Roll_no")
    elif(Roll_no.find_one({"Roll_no":"{}".format(a)})):
        morning = collection.find_one({"Roll no": "{}".format(a)})
        if(morning):
            present = morning["Morning_Present"]
        if present==1:
                #cam = cv2.VideoCapture(0)
                with placeholder.container():
                    img_file_buffer = st.camera_input('')
                if img_file_buffer is not None:
                    with open ("dynamic.jpg","wb") as f:
                        f.write(img_file_buffer.getbuffer())
                #    # To read image file buffer with OpenCV:
                #    bytes_data = img_file_buffer.getvalue()
                #    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                #    # Check the type of cv2_img:
                #    # Should output: <class 'numpy.ndarray'>
                #    st.write(type(cv2_img))

                #    # Check the shape of cv2_img:
                #    # Should output shape: (height, width, channels)
                #    st.write(cv2_img.shape)
                try:
                    known_image = face_recognition.load_image_file("{}.jpg".format(a))
                    unknown_image = face_recognition.load_image_file("dynamic.jpg")
                    biden_encoding = face_recognition.face_encodings(known_image)[0]
                    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
                    if(results==[True]):
                        #print("RECOGNIZED")
                        st.write('RECOGNIZED')
                        placeholder.empty()
                        #img = cv2.imread(img_name)
                        #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        #path="haarcascade_frontalface_default.xml"
 #
                        #face_cascade=cv2.CascadeClassifier(path)
                        #faces=face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(40,40))
                        ##print(len(faces))
                        #i=0
                        #for(x,y,w,h) in faces:
                        #    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        #    i=1
                        #cv2.putText(img, strftime("%H:%M:%S"), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),2,cv2.LINE_AA)
                        #cv2.imwrite("dynamic.jpg".format(a),img)
                        #cv2.imshow("image",img)
                        if collection.find_one({"Roll no":"{}".format(a)}):
                            document = collection.find_one({"Roll no": "{}".format(a)})
                            attendance = document["Period_Attendent"]
                            Entry=document["Entry_Time"]
                            collage_hour=6
                            Entry_hour=Entry.hour
                            Exit=datetime.now()
                            Exit_hour=Exit.hour
                            if(Exit_hour<Entry_hour):
                                time=(Exit_hour+24)-Entry_hour
                            else:
                                time=Exit_hour-Entry_hour
                            if(time>collage_hour):
                                time=collage_hour
                            collection.update_one({
                            "Roll no":"{}".format(a)
                            },
                            {"$set": { "Exit_Time":datetime.now(),"Period_Attendent":attendance + time ,"Morning_Present":0}}
                            )
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    else:
                       # print("NOT RECOGNIZED")
                        st.write("NOT RECOGNIZED")
                        placeholder.empty()
 
                except IndexError:
                    #print("NO FACE IS RECOGNIZED")
                    st.write('NO FACE IS RECOGNIZED')
                    #print("TRY ONCE AGAIN")
                    st.write('Try once again')
                    placeholder.empty()
        else:
            #cam = cv2.VideoCapture(0)
            img_file_buffer = st.camera_input('')
            if img_file_buffer is not None:
                with open ("dynamic.jpg","wb") as f:
                    f.write(img_file_buffer.getbuffer())
            try:
                known_image = face_recognition.load_image_file("{}.jpg".format(a))
                unknown_image = face_recognition.load_image_file("dynamic.jpg".format(a))
                biden_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
                if(results==[True]):
                    st.write("RECOGNIZED")
                    if collection.find_one({"Roll no":"{}".format(a)}):
                        collection.update_one({
                        "Roll no":"{}".format(a)
                        },
                        {"$set": { "Entry_Time":datetime.now() ,"Exit_Time":0,"Morning_Present":1}}
                        )
                    else:
                        collection.insert_one({
                        "Roll no":"{}".format(a),
                        "Entry_Time":datetime.now(),
                        "Exit_Time":0,
                        "Period_Attendent":0,
                        "Morning_Present":1
                        })
                    placeholder.empty()
                else:
                    st.write("NOT RECOGNIZED")
                    placeholder.empty()
            except IndexError:
                st.write("NO FACE IS RECOGNIZED")
                st.write("TRY ONCE AGAIN")
                placeholder.empty()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        st.write("You have entered Wrong Roll_no")
