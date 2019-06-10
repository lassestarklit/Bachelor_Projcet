from view.PageView import *
import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class CompareFrame(Page):

    def __init__(self,n_controller, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        self.controller = n_controller

        self.button_frame = Frame(self)
        self.button_frame.pack(side="top", fill="x", expand=False)

        self.title_label = Label(self, text="compare")
        self.title_label.pack(side="top", fill="both", expand=False)

        self.navigation_frame = Frame(self)
        self.navigation_frame.pack(side="top", fill="x", expand=False)
        self.button_previous=Button(self.navigation_frame, text="Previous feature",
                                    command=lambda: self.controller.change_graph("previous"))
        self.button_previous.pack(in_=self.navigation_frame, side="left")
        self.button_next=Button(self.navigation_frame, text="Next feature",command=lambda: self.controller.change_graph("forward"))
        self.button_next.pack(in_=self.navigation_frame, side="left")
        self.button_stats = Button(self.navigation_frame, text="Run statistical test",
                                  command=lambda: self.controller.run_statistics())
        self.button_stats.pack(in_=self.navigation_frame, side="left")

    def update_button_frame(self):
        for widgets in self.button_frame.winfo_children():
            widgets.destroy()

    def plot(self,ax):
        self.clear_canvas()
        self.canvas = FigureCanvasTkAgg(ax, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack()

    def set_title(self,text):
        self.title_label.config(text=text)

    def clear_canvas(self):
        try:
            self.canvas.get_tk_widget().destroy()
            self.toolbar.destroy()
        except AttributeError as e: print(e)