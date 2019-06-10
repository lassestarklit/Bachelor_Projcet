from View.PageView import *
from tkinter.messagebox import showerror

class ModelFrame(Page):
    def __init__(self, n_controller, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        self.controller = n_controller

        self.navigation_frame = Frame(self)
        self.navigation_frame.pack(side="top", fill="x", expand=False)

        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)



        #Initialize frames within model containerframe
        self.process_menu = ModelDataProcess(self.controller, self)
        self.process_menu.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.model_selection_menu = ModelSelection(self.controller, self)
        self.model_selection_menu.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        self.model_analysis_menu = ModelAnalysis(self.controller, self)
        self.model_analysis_menu.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        # Initialize buttons  within model frame
        self.button_data_process = Button(self.navigation_frame, text="Data process", command=self.process_menu.show)
        self.button_data_process.pack(in_=self.navigation_frame, side="left")

        self.button_models = Button(self.navigation_frame, text="Models", command=self.model_selection_menu.show)
        self.button_models.pack(in_=self.navigation_frame, side="left")

        self.button_next = Button(self.navigation_frame, text="Analysis", command=self.go_to_analysis)
        self.button_next.pack(in_=self.navigation_frame, side="left")

    def data_process_tab(self):
        self.process_menu.config_frame()
        self.process_menu.show()

    def go_to_analysis(self):
        if self.controller.target_variable.get_number_of_variables()==0:
            showerror("Missing Data","No target variable is chosen")
        else:
            self.model_analysis_menu.show()
            self.model_analysis_menu.initialize()

class ModelDataProcess(Page):
    def __init__(self, n_controller, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        self.controller = n_controller


    def config_frame(self):
        self.radio_button_frame = Frame(self)
        self.radio_button_frame.place(height=250, width=150, x=20, y=30)
        self.lab = Label(self.radio_button_frame, text="Choose target")
        self.lab.pack(in_=self.radio_button_frame, side="top")

        self.prim_target_frame = Frame(self)

        self.prim_target_frame.place(height=200, width=150, x=20, y=230)

        self.independent_variables_frame = Frame(self)
        self.independent_variables_frame.place(height=300, width=280, x=210, y=30)

        self.controller.set_radio_buttons_target_variable(self.radio_button_frame)


class ModelSelection(Page):
    def __init__(self, n_controller, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        self.controller = n_controller
        self.controller.initialize_models()
        self.controller.set_checkboxes(self)


class ModelAnalysis(Page):
    def __init__(self, n_controller, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        self.controller = n_controller

        self.option_frame = Frame(self)
        self.option_frame.pack(side="top",fill="x", expand=False)

        self.container_analysis = Frame(self)
        self.container_analysis.pack(side="top",fill="both",expand=True)
        self.separator_var_analysis = StringVar(self)
        self.options = ["General", "Comparison", "Models"]
        drop_down_separator = OptionMenu(self.option_frame, self.separator_var_analysis, *self.options,
                                             command=self.change_dropdown)
        drop_down_separator.config(width=13)
        drop_down_separator.pack(in_=self.option_frame, side="left")

    def initialize(self):
            self.separator_var_analysis.set("General")
            self.controller.fill_general(self.container_analysis)

    # on change dropdown value
    def change_dropdown(self,value):
        if value == "General":
            self.controller.fill_general(self.container_analysis)
        elif value=="Comparison":
            self.controller.fill_compare(self.container_analysis)
        elif value=="Models":
            self.controller.fill_models(self.container_analysis)