import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
count = 1
picture = 0
class App:

    def __init__(self, window, window_title, video_source=0):

        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = 1400, height = 500)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="CAPTURE", width=20, command=self.snapshot, fg = 'white', bg = 'orange')
        self.btn_snapshot.pack(anchor=tkinter.W, expand = True, padx = 240, pady = 20)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        global count
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame_"+time.strftime("%Y_%m_%d_%H_%M_%S") + ".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            #cv2.imwrite("frame_" + str(count) + ".png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #count = count + 1

    def update(self):
        global picture
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        gif1 = tkinter.PhotoImage(
            file='C:\\Users\Park\PycharmProjects\PyCharmCamera\\frame_2019_06_26_11_52_02.png')
        label = tkinter.Label(image=gif1)
        label.image = gif1  # keep a reference!
        picture += 1

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

            self.canvas.create_image(730, 0, image=gif1, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source = 0):
        #open video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open source", video_source)

        #Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)

        else:
            return (ret, None)

    #Release video when object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")