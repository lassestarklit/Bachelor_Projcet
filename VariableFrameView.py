from view.PageView import *
from view.ListPopUpView import *
class VariableFrame(Page):
    def __init__(self, n_controller, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)

        self.controller=n_controller


        dependent_lbl = Label(self, text="Dependent Variables")
        dependent_lbl.pack()
        dependent_lbl.place(x=20, y=50)

        independent1_lbl = Label(self, text="Independent Variables")
        independent1_lbl.pack()
        independent1_lbl.place(x=250, y=50)

        self.dependent_listbox = Listbox(self, selectmode=MULTIPLE)
        self.dependent_listbox.pack()
        self.dependent_listbox.place(x=20, y=80)

        # this button will delete the selected item from the list. input listbox
        delete_depen_btn = Button(self, text="Delete",
                                  command=lambda: self.delete_variable(self.dependent_listbox, "dependent"))
        delete_depen_btn.pack()
        delete_depen_btn.place(x=115, y=255)

        # Function in controller to add to list
        # adds lambda to command, so the function wont exectue before button trickered
        delete_depen_btn = Button(self, text="Add", command=lambda: self.open_pop_up(self.dependent_listbox, "dependent"))
        delete_depen_btn.pack()
        delete_depen_btn.place(x=20, y=255)

        self.independent_listbox = Listbox(self, selectmode=MULTIPLE)

        self.independent_listbox.pack()
        self.independent_listbox.place(x=250, y=80)

        # this button will delete the selected item from the list. input listbox
        delete_indepen_btn = Button(self, text="Delete", command=lambda: self.delete_variable(self.independent_listbox, "independent"))
        delete_indepen_btn.pack()
        delete_indepen_btn.place(x=345, y=255)

        # Function in controller to add to list
        # adds lambda to command, so the function wont exectue before button trickered
        add_depen1_btn = Button(self, text="Add", command=lambda: self.open_pop_up(self.independent_listbox, "independent"))
        add_depen1_btn.pack()
        add_depen1_btn.place(x=250, y=255)

    def open_pop_up(self,variable_listbox,list_name):
        ListPopUp(self.controller,variable_listbox,list_name)



    def delete_variable(self,variable_listbox, list_name):
        selected_values = [variable_listbox.get(i) for i in variable_listbox.curselection()]
        #Delete from listbox starting from the highest index
        for index in variable_listbox.curselection()[::-1]:
            variable_listbox.delete(index)
        self.controller.delete_variable(list_name,selected_values)