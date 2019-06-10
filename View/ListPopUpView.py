from tkinter import *

class ListPopUp:
    def __init__(self,controller,listbox,variable_list):
        self.controller = controller
        self.listbox=listbox
        self.variable_list=variable_list
        self.top = Toplevel()
        self.top.wm_geometry("200x250")
        self.top.title("Available variables")



        lbl = Label(self.top, text="Available Variables")
        lbl.pack(side="top")


        # I controller lav en indsæt og slet, som kan indsætte værdier i begge variable lists
        self.available_listbox = Listbox(self.top,selectmode=MULTIPLE)
        self.available_listbox.pack(side="top")
        self.controller.add_variable(self.available_listbox,"available_listbox","")

        dis_button = Button(self.top, text="Dismiss", command=self.top.destroy)
        dis_button.pack()
        dis_button.place(x=100,y=200)


        add_button = Button(self.top, text="Add", command=self.add_selected_list)

        add_button.pack()
        add_button.place(x=10, y=200)

    def add_selected_list(self):
        selected_values = [self.available_listbox.get(i) for i in self.available_listbox.curselection()]
        self.controller.add_variable(self.listbox, self.variable_list, selected_values)
        self.top.destroy()
