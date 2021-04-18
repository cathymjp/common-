from tkinter import *

window = Tk()

def b1event():
    if(btn1['text'] == 'hello'):
        btn1['text'] = 'world'
        btn1['fg'] = 'red'
        btn1['bg'] = 'yellow'
    else:
        btn1['text'] = 'hello'
        btn1['fg'] = 'yellow'
        btn1['bg'] = 'red'
topFrame = Frame(window)
topFrame.pack(side = TOP)

btn1 = Button(window, text = "hello", command = b1event, fg = 'yellow', bg = 'red')
btn2 = Button(window, text = "button2")

btn1.pack(side = RIGHT)
btn2.pack()


window.mainloop()