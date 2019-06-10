from View.PageView import *

class LoadDataFrame(Page):
    filepath = ""

    def __init__(self, n_controller, *args, **kwargs):
        Page.__init__(self,*args, **kwargs)
        self.controller = n_controller
        separator_label = Label(self, text="Select type of separator:")
        separator_label.pack()
        separator_label.place(x=50, y=100)

        self.separator_var = StringVar(self)
        self.separator_var.set(",")  # default value
        self.drop_down_separator = OptionMenu(self, self.separator_var, ",", ";")
        self.drop_down_separator.pack()
        self.drop_down_separator.place(x=230, y=100)

        file_path_button = Button(self, text="Browse", command=self.controller.browse_file_btn)
        file_path_button.pack()
        file_path_button.place(x=240, y=180)

        file_path_label = Label(self, text="File path")
        file_path_label.pack()
        file_path_label.place(x=50, y=150)

        # Self to be able to config in load_file function
        self.file_path_txt = Entry(self, width=40)
        self.file_path_txt.pack()
        self.file_path_txt.place(x=120, y=150)



    def insert_to_file_path_txt(self,value):
        self.file_path_txt.delete(0, 'end')
        self.file_path_txt.insert(0,value)


        load_file = Button(self, text="Load file", command=lambda:self.controller.load_file(self.separator_var.get(),self.file_path_txt.get()))
        load_file.pack()
        load_file.place(x=300, y=350)