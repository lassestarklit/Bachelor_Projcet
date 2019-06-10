from tkinter import *
from View.CompareFrameView import *
from View.LoadFrameView import *
from View.ModelFrameView import *
from View.VariableFrameView import *

class MainView(Frame):
    filename: str

    def __init__(self, controller,*args, **kwargs):
        Frame.__init__(self, *args, **kwargs)

        self.controller=controller
        self.variable_menu = VariableFrame(self.controller, self)
        self.compare_menu = CompareFrame(self.controller, self)
        self.upload_data = LoadDataFrame(self.controller, self)
        self.model_menu = ModelFrame(self.controller, self)



        button_frame = Frame(self)
        container = Frame(self)
        self.filename_frame = Frame(self)


        button_frame.pack(side="top", fill="x", expand=False)
        self.filename_frame.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        self.file_name_label = Label(self.filename_frame, text="File name: ")
        self.file_name_label.pack(side="left")
        self.variable_menu.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.compare_menu.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.upload_data.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.model_menu.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        # adding ttk on button object because of bug
        b1 =Button(button_frame, text="Variable menu", command=self.variable_menu.show)
        b2 = Button(button_frame, text="Compare", command=self.combine_functions_for_graph)
        b3 = Button(button_frame, text="Data upload", command=self.upload_data.show)
        b4 = Button(button_frame, text="Model Menu", command=self.combine_functions_for_model)

        b3.pack(side="left")
        b1.pack(side="left")
        b2.pack(side="left")
        b4.pack(side="left")

        self.upload_data.show()

    def update_filename(self,filename):
        self.filename=filename
        self.file_name_label.config(text="File name: {0}".format(self.filename))

    def combine_functions_for_graph(self):
        self.compare_menu.show()
        self.compare_menu.update_button_frame()
        self.controller.set_buttons()

    def combine_functions_for_model(self):

        self.model_menu.show()
        self.model_menu.data_process_tab()


