# Trying to make a GUI demo
# Probably going to leave this unfinished
# Would prefer to collaborate with someone who knows GUI stuff better


#   Load a random list of sample captchas
#       For each sample captcha:
#           Load the captcha in the window
#           Run my code to break it, return a solution
#           Automatically fill in a submission form with the solution
#           Automatically submit the form
#           If wrong:
#               Error message
#               Continue
#           If correct:
#               You are human, success
#               Break


from tkinter import *
from PIL import Image, ImageTk


def makeform(root, answer):
    entries = {}
    row = Frame(root)
    lab = Label(row, width=22, text="Enter text: ", anchor='w')
    ent = Entry(row)
    ent.insert(0, "")
    row.pack(side=TOP, fill=X, padx=5 , pady=5)
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, expand=YES, fill=X)
    entries["Enter text"] = ent
    return entries


def submit(entries):
    answer = str(entries["Enter text"].get())
    print(answer)


def break_cap():
    answer = "XXXX"
    return answer


#if __name__ == '__main__':
photo1 = "test_captchas/2A5Z.png"
photo2 = "test_captchas/2AD9.png"
answer = "2A5Z"
answer2 = "2AD9"


root = Tk()
root.title("Are you a robot?")

root.photo1 = ImageTk.PhotoImage(Image.open(photo1))
root.photo2 = ImageTk.PhotoImage(Image.open(photo2))
vlabel = Label(root,image=root.photo1)
vlabel.pack()

#root.geometry("400x400")

ents = makeform(root, answer)

#root.bind('<Return>', (lambda event, e = ents: fetch(e)))
b1 = Button(root, text = 'submit', command=(lambda e = ents: submit(e)))
#b1 = Button(root, text='submit', command=(submit(answer)))
b1.pack(side=LEFT, padx=5, pady=5)

b3 = Button(root, text='Quit', command=root.quit)
b3.pack(side=LEFT, padx=5, pady=5)

ans = break_cap()

root.mainloop()



#
