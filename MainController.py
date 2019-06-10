from model.VariableCollections import *
from model.VariableCollections import IndependentVariables
from view.MainView import *
from view.ModelFrameView import *
from view.MultiListbox import *
from model.Models import *
from model.Compare import *
import pandas as pd
from tkinter import *
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfilename


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest, chi2,SelectPercentile



class MainController:



    def __init__(self):
        self.root = Tk()
        # passes the controller and window to mainview
        self.main = MainView(self, self.root)
        self.main.pack(side="top", fill="both", expand=True)
        self.root.wm_geometry("500x500")

    def run(self):
        self.root.title("Data Management System")

        while True:

            try:
                self.root.mainloop()
                break
            except UnicodeDecodeError:
                pass





    #Need to be the one commented out when done testing
    separator: str
    all_attributes = pd.DataFrame()


    def browse_file_btn(self):
        filepath = askopenfilename(filetypes=[("Data files", "*.csv"),
                                                   ("All files", "*.*")])

        if filepath != "":
            try:
                self.main.upload_data.insert_to_file_path_txt(filepath)
            except:
                showerror("Open Source File", "Failed to read file\n'%s'" % self.filepath)
            return

    def load_file(self,sep,file_path):
        #Commented out is the right. this is only for testing
        if file_path != "":
            is_initialized=True
            try:
                #uncomment sep and filepath
                separator = sep
                filepath = file_path


                #check if a file has been loaded. If so empty is not possible as all_attributes will be an instance of a class instead of pd
                try:
                    self.all_attributes.empty
                    is_initialized=False
                except AttributeError:
                    self.dependent_variables.delete_all()
                    self.independent_variables.delete_all()
                    self.target_variable.delete_all()
                    self.main.variable_menu.independent_listbox.delete(0, 'end')
                    self.main.variable_menu.dependent_listbox.delete(0, 'end')

                self.all_attributes = AllAttributes(pd.read_csv(filepath, sep=separator, index_col=False),filepath)
                self.main.update_filename(self.all_attributes.get_file_name())
                if not is_initialized:
                    self.initialize_instances()

            except FileNotFoundError:
                showerror("File not found", "File was not found in location:\n'%s'" % filepath)
            except pd.errors.ParserError:
                showerror("Incorrect format", "Issues occurred when trying to parse the file:\n'%s'" % filepath)
        else:
            showerror("Missing file", "Choose file first")

    all_available = pd.DataFrame()
    independent_variables: IndependentVariables
    dependent_variables: DependentVariables
    comparison: Comparison

    def initialize_instances(self):
        self.all_available = AllAvailable(self.all_attributes.get_table().copy())
        self.independent_variables = IndependentVariables()
        self.dependent_variables = DependentVariables()
        self.comparison = Comparison(self.all_attributes)
        self.initialize_models()

    def initialize_models(self):
        self.KNN=KNNModel()
        self.models.append(self.KNN)
        self.active_models.append(self.KNN)

        self.SVC=SVCModel()
        self.models.append(self.SVC)
        self.active_models.append(self.SVC)

        self.DecisionTree=DecTreeModel()
        self.models.append(self.DecisionTree)
        self.active_models.append(self.DecisionTree)

        self.NB=NBModel()
        self.models.append(self.NB)
        self.active_models.append(self.NB)

        self.RandomForest=RandomForestModel()
        self.models.append(self.RandomForest)
        self.active_models.append(self.RandomForest)

        self.NN=NNModel()
        self.models.append(self.NN)
        self.active_models.append(self.NN)

        self.LR=LRModel()
        self.models.append(self.LR)
        self.active_models.append(self.LR)

        self.GDC=GBCModel()
        self.models.append(self.GDC)
        self.active_models.append(self.GDC)

    ##For variable menu
    #variable used later to know if independent variable frame is changed
    variable_list_is_changed=False
    def add_variable(self, listbox, variable_list,variables):

        if variable_list == 'available_listbox':
            n = 1
            try:
                for variable in self.all_available.get_table():

                    listbox.insert(n, variable)
                    n += 1
            except AttributeError:
                showerror("File not chosen", "There is no file uploaded to the system")
                return
        else :

            for variable in variables:
                if variable_list == 'independent':
                    if self.all_attributes.variable_is_numerical(variable):
                        self.independent_variables.add(self.all_attributes.get_variable(variable))
                        self.all_available.delete(variable)
                        listbox.insert(END, variable)
                        self.variable_list_is_changed=True
                    else:
                        showerror("Wrong variable", "Cannot choose {0}. Only numerical variables can be loaded as independent variables".format(variable))


                else:
                    if self.all_attributes.variable_is_binary(variable):

                        self.dependent_variables.add(self.all_attributes.get_variable(variable))
                        self.all_available.delete(variable)
                        listbox.insert(END, variable)
                    else:
                        showerror("Wrong variable", "Cannot choose {0}. Only 2-class variables can be loaded as a dependent variable".format(variable))



    def delete_variable(self,variable_list,variables):
        for variable in variables:
            self.all_available.add(self.all_attributes.get_variable(variable))
            if variable_list == 'independent':
                self.independent_variables.delete(variable)
                self.variable_list_is_changed = True


            else:
                self.dependent_variables.delete(variable)




    ##For Compare menu
    index_dependent=0
    index_independent=0
    graph_index=0


    def set_buttons(self):
        for i in range(self.dependent_variables.get_number_of_variables()):

            compare_btn=Button(self.main.compare_menu.button_frame, text=self.dependent_variables.get_variable_at_index(i).columns[0], command=lambda i=i: self.show_new_groups(i))
            compare_btn.pack(in_=self.main.compare_menu.button_frame, side="left")

    def change_graph(self,direction):

        if direction=="forward":
            if self.index_independent>=self.independent_variables.get_number_of_variables()-1:
                self.index_independent = 0
            else:
                self.index_independent += 1
        else:
            if self.index_independent == 0:
                self.index_independent = self.independent_variables.get_number_of_variables() - 1
            else:
                self.index_independent -= 1
        try:
            self.show_graphs()
        except IndexError:
            showerror("Unknown target","Please select the target variable")

    def show_new_groups(self,index):
        self.index_dependent=index
        self.index_independent=0
        self.show_graphs()



    def show_graphs(self):
        if self.independent_variables.get_number_of_variables()==0:
            showerror("No features to compare", "Please select features to compare")
        else:
            independent_variable=self.independent_variables.get_variable_at_index(self.index_independent)

            variable_name=self.dependent_variables.get_variable_name_at_index(self.index_dependent)
            list_classes = self.all_attributes.get_unique_classes_of_variable(variable_name)

            self.comparison.config_graph(variable_name, list_classes, independent_variable)

            fig=self.comparison.get_graph()

            self.main.compare_menu.plot(fig)
            self.main.compare_menu.set_title("Comparison of {0} using feature {1}".format(list_classes, independent_variable.columns[0]))


    def run_statistics(self):
        if self.independent_variables.get_number_of_variables() == 0:
            showerror("No features to compare", "Please select features to compare")
        else:
            text=self.comparison.perform_ttest()
            nullhypothesis="Null Hypothesis: There is no significant difference between class A and B"
            messagebox.showinfo("Statistical Analysis", "{0} \n{1}".format(nullhypothesis,text))



    #for machine learning part
    ##For target and feature selections
    target_variable=TargetVariable()
    target_class=""

    def set_radio_buttons_target_variable(self,frame):
        v = StringVar()
        for variable in self.dependent_variables.get_variable_names():
            Radiobutton(frame, text=variable, variable=v, value=variable,command=lambda val=variable: self.choose_target_variable(val)).pack(in_=frame, side=TOP, anchor="w",pady=(10,0))

    def choose_target_variable(self,val):

        if self.target_variable.get_number_of_variables()!=0:
            self.target_variable.delete_all()
            self.delete_frame_widgets(self.main.model_menu.process_menu.prim_target_frame)
            self.delete_frame_widgets(self.main.model_menu.process_menu.independent_variables_frame)
        self.target_variable.add(self.all_attributes.get_variable(val))
        if not(self.all_attributes.variable_is_numerical(self.target_variable.get_variable_name_at_index(0))):
            self.target_variable.label_encode_variable(val)
        self.variable_list_is_changed = True

        #Set up other frames in current container, when a target value is chosen.
        self.rescaled_data_frame()
        self.set_up_independent_analysis()

    def get_target_is_label_encoded(self):
        return self.target_variable.get_number_of_variables() == 2


    def rescaled_data_frame(self):
        frame = self.main.model_menu.process_menu.prim_target_frame
        self.delete_frame_widgets(frame)
        lab = Label(frame, text="Rescaled variables")
        lab.pack(in_=frame, side="top")

        self.var_rescaled = []
        self.column_names = []

        for i in range(self.independent_variables.get_number_of_variables()):
            col_name=self.independent_variables.get_variable_name_at_index(i)

            self.column_names.append(col_name)
            self.var_rescaled.append(StringVar())
            #checks if column is rescaled
            if self.independent_variables.variable_is_rescaled(col_name):
                self.var_rescaled[-1].set(1)
            else:
                self.var_rescaled[-1].set(0)

            c = Checkbutton(frame, text=col_name, variable=self.var_rescaled[-1],
                            command=lambda i=i: self.rescaled_event(i), onvalue=1, offvalue=0).pack()


    def rescaled_event(self, i):

        if self.var_rescaled[i].get() == "1":

            self.independent_variables.add_to_rescaled_table(self.independent_variables.get_table()[self.column_names[i]])
            self.variable_list_is_changed = True



        elif self.var_rescaled[i].get() == "0":
            self.independent_variables.remove_from_rescaled_table(self.column_names[i])
            self.variable_list_is_changed = True

    def set_up_independent_analysis(self):
        frame=self.main.model_menu.process_menu.independent_variables_frame
        self.delete_frame_widgets(frame)

        self.independent_variable_mult=MultiListbox(frame, (('Column Name', 15), ('Correlation', 10)))
        self.independent_variable_mult.pack()

        #populates multilist
        for i in range(self.independent_variables.get_number_of_variables()):
            feature = self.independent_variables.get_active_variable_at_index(i)
            self.independent_variable_mult.insert(END, (
            feature.columns[0], round(self.target_variable.get_feature_correlation(feature), 4)))



        delete_button=Button(frame,text="Remove selected",command=self.delete_variable_model_menu)
        delete_button.pack()
        delete_button.place(x=50,y=200)

        correlation_button=Button(frame,text="Show Correlation",command=self.plt_corr_graph)
        correlation_button.pack()
        correlation_button.place(x=50,y=240)

    def delete_frame_widgets(self,frame):
        for widgets in frame.winfo_children():
            widgets.destroy()

    def delete_variable_model_menu(self):

        try:
            variable=self.independent_variable_mult.get(self.independent_variable_mult.curselection())[0]
            self.all_available.add(self.all_attributes.get_variable(variable))
            self.independent_variables.delete(variable)
            self.main.variable_menu.independent_listbox.delete(self.independent_variable_mult.curselection()[0])
            self.independent_variable_mult.delete(self.independent_variable_mult.curselection())
            self.rescaled_data_frame()
            self.variable_list_is_changed = True
        except TclError:
            pass

    def plt_corr_graph(self):
        top = Toplevel()
        top.wm_geometry("420x400")
        fig=self.comparison.plot_correlation_matrix(self.independent_variables)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, top)
        canvas.draw()
        canvas.get_tk_widget().pack()





    def set_target_class(self,val):
        self.target_class = val
    ###
    def target_variable_is_chosen(self):
        return

    ##For model selection

    models=[]
    active_models=[]



    def set_checkboxes(self,frame):
        self.vars = []
        for i in range(len(self.models)):
            self.vars.append(StringVar())
            self.vars[-1].set(1)
            c = Checkbutton(frame, text=self.models[i].get_name(), variable=self.vars[-1],
                            command=lambda i=i: self.checkbox_event(i), onvalue=1, offvalue=0)
            c.pack()

    def checkbox_event(self,i):
        if self.vars[i].get()=="1":
            self.insert_to_models(self.models[i])
        elif self.vars[i].get()=="0":
            self.delete_from_models(self.models[i])

    def insert_to_models(self,model):
        for i in range(len(self.models)):
            if self.models[i].get_name()==model.get_name():
                self.active_models.insert(i,model)

    def delete_from_models(self,model):
        self.active_models.remove(model)

    #For model analysis

    train_test_indices=[]
    training_size=0
    test_size=0

    def generate_train_test(self,cv):
        #resets list of train_test_indices
        self.train_test_indices = []
        K = cv
        CV = model_selection.KFold(n_splits=K, shuffle=True)

        X = self.independent_variables.get_active_table()
        y=self.target_variable.get_target_variable()

        # Get train and test data for K fold CV
        for train_index, test_index in CV.split(X, y):

            self.train_test_indices.append([train_index, test_index])
            self.training_size=len(train_index)
            self.test_size=len(test_index)


    def set_model_scores(self):

        X = self.independent_variables.get_active_table()

        y=self.target_variable.get_target_variable()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,
                                                            random_state=42)
        models=self.active_models
        #try. if single model is uploaded except

        for model in models:
            model.load_current_features_target(X, y)
            model.set_score(self.train_test_indices)
            model.set_precision_specificity_recall_f1(X_train, X_test, y_train, y_test)


    def model_is_empty(self):
        return len(self.active_models)==0

    def populate_model_buttons(self):
        frame = self.main.model_menu.process_menu.model_analysis_menu
        for model in self.active_models():
            frame.compare_btn=Button(frame, text=self.model.get_name(),command=lambda model=model: self.model_regulation(model))
            frame.compare_btn.pack(in_=frame, side="left")

    #specifies current folds that models are evaluated on
    num_of_cv = 10
    #changes in train and test set
    train_test_changed=False
    def fill_general(self,frame):

        if not (self.models_are_initialized()) or self.variable_list_is_changed:

            self.generate_train_test(self.num_of_cv)
            self.set_model_scores()
            self.variable_list_is_changed = False


        self.delete_frame_widgets(frame)

        Label(frame, text="Values in independent variables list").pack(side="top")


        columns=[]
        num_of_columns=self.independent_variables.get_number_of_variables()
        for i in range(self.independent_variables.get_number_of_variables()):

            columns.append((self.independent_variables.get_active_variable_name_at_index(i), int(55 / num_of_columns)))




        mult = MultiListbox(frame, columns)
        mult.pack(side="top")


        #Button to generate new train and test data

        button=Button(frame, text="Regenerate train and test sets", command=lambda:self.call_gen_train_test(self.num_of_cv_txt.get()))
        button.pack()
        button.place(x=30,y=280)


        descr_text = Label(frame, text="New train and test set on")
        descr_text.pack()
        descr_text.place(x=30,y=250)

        descr_text_1= Label(frame, text="fold cross validation")
        descr_text_1.pack()
        descr_text_1.place(x=230, y=250)

        # number of cv
        self.num_of_cv_txt = Entry(frame, width=3)
        self.num_of_cv_txt.pack()
        self.num_of_cv_txt.place(x=195, y=250)
        self.num_of_cv_txt.insert(0,self.num_of_cv)

        #current train and test size:
        self.current_train=Label(frame, text="Current training size: {0}".format(len(self.train_test_indices[0][0])))
        self.current_train.pack()
        self.current_train.place(x=45, y=330)
        self.current_test = Label(frame, text="Current test size: {0}".format(len(self.train_test_indices[0][1])))
        self.current_test.pack()
        self.current_test.place(x=45, y=360)


        for j in range(len(self.independent_variables.get_active_variable_at_index(0))):
            values=[]
            for i in range(self.independent_variables.get_number_of_variables()):
                values.append(round(self.independent_variables.get_active_variable_value_at(j, i), 4))

            mult.insert(END, (values))

    def call_gen_train_test(self,val):

        try:
            self.num_of_cv=val

            value=int(val)
            if value<=1:
                showerror("Wrong input", "Number of folds have to be more than 1")
            else:

                self.generate_train_test(value)
                self.current_train.config(text="Current training size: {0}".format(self.training_size))
                self.current_test.config(text="Current test size: {0}".format(self.test_size))
                self.num_of_cv=value
                self.train_test_changed=True



        except ValueError:
            showerror("Wrong input", "Input has to be an integer")



    def fill_compare(self,frame):
        self.delete_frame_widgets(frame)

        if self.train_test_changed:
            self.set_model_scores()
            self.train_test_changed=False




        mult = MultiListbox(frame, (('Model', 15), ('Accuracy[%]', 8), (" Error Rate[%]", 10),(" Run time [s]", 8)))
        mult.pack()
        mult.place(x=0,y=0)

        acc_plot_button = Button(frame,text="Show Accuracy boxplot",command=lambda: self.plot_acc_err("Accuracy"))
        acc_plot_button.pack()
        acc_plot_button.place(x=0,y=300)
        err_plot_button = Button(frame, text="Show Error boxplot",command=lambda: self.plot_acc_err("Error"))
        err_plot_button.pack()
        err_plot_button.place(x=170, y=300)

        if self.all_attributes.variable_is_binary(self.target_variable.get_variable_name_at_index(0)):
            roc_plot = Button(frame, text="Show Roc curves", command=self.plot_roc)
            roc_plot.pack()
            roc_plot.place(x=315, y=300)

        for model in self.active_models:
            mult.insert(END, (model.get_name(), model.get_accuracy(), model.get_error(),model.get_time()))



    def fill_models(self, frame):
        self.delete_frame_widgets(frame)

        # Set the treeview
        self.tree = ttk.Treeview(frame)
        self.tree.pack(side="top",anchor="w")


        self.tree.heading('#0', text='Models', anchor="w")
        self.tree.column("#0", minwidth=0, width=200, stretch=NO)


        #insert models in treeview
        for model in self.active_models:

            self.tree.insert("", "end",model.get_name(), text=model.get_name())
            self.tree.insert(model.get_name(),"end",text="Tune")
            self.tree.insert(model.get_name(),"end",text="Predict")
            self.tree.insert(model.get_name(), "end",text="Performance Metrics")
            self.tree.insert(model.get_name(), "end", text="Feature selection")

            #instantiate precision etc..

        #buttonRelease, otherwise the onClick function wont be executed before next click
        self.tree.bind("<ButtonRelease-1>", self.onClick)

        self.content_model_frame = Frame(frame)
        self.content_model_frame.pack()
        self.content_model_frame.place(height=200, width=300, x=205, y=0)

        self.content_model_bottom = Frame(frame)
        self.content_model_bottom.pack()
        self.content_model_bottom.place(height=200, width=500, x=0, y=201)


    #active model is the model, which is chosen for modelling
    active_model:object
    def onClick(self, event):

        item_iid = self.tree.selection()[0]
        parent_iid = self.tree.parent(item_iid)

        #if parent node is clicked select model
        if parent_iid:
            model_name=self.tree.item(parent_iid)['text']
            for model in self.active_models:
                if model_name == model.get_name():
                    self.active_model = model
                    self.active_model.load_current_features_target(self.independent_variables.get_active_table(),
                                                                   self.target_variable.get_target_variable()
)

        #if child node is clicked
        else:
            #make sure it's still the same model
            model_name = self.tree.item(item_iid)['text']
            #try since active model isn't activated
            try:
                if model_name != self.active_model.get_name():
                    # If model tree is chosen

                    for model in self.active_models:

                        if model_name == model.get_name():
                            self.active_model = model
                            self.active_model.load_current_features_target(
                                self.independent_variables.get_active_table(), self.target_variable.get_table())
            except AttributeError:
                pass
        #make sure data in the model is up to date

        #now for child nodes
        curItem = self.tree.focus()
        item=self.tree.item(curItem)['text']



        if item=="Tune":
            self.tune_model()

        if item == "Predict":
            self.active_model.fit_for_finale()
            self.predict()
        if item == "Performance Metrics":
            self.performance_metrics()

        if item == "Feature selection":
            self.feature_selection()




        #when subtree of specific model is chosen

    def tune_model(self):
        right_frame = self.content_model_frame
        bottom_frame = self.content_model_bottom

        self.delete_frame_widgets(right_frame)
        self.delete_frame_widgets(bottom_frame)

        Label(right_frame,text="Auto tune").pack(side="top")
        button1=Button(right_frame, text="Randomized search CV",command=self.rand_fit)

        button1.pack(side="left",anchor="nw")
        button2 = Button(right_frame, text="Grid search CV", command=self.grid_fit)
        button2.pack(side="left", anchor="ne")


        # Create a Tkinter variable
        tkvar_model_para = StringVar()

        # Dictionary with options
        parameter_choices=[]

        for params in self.active_model.get_parameters():
            parameter_choices.append(params)


        tkvar_model_para.set(parameter_choices[0])  # set the default option

        para_menu = OptionMenu(right_frame, tkvar_model_para, *parameter_choices,command=self.change_current_param_val)
        n = Label(right_frame, text="Tune Manually")
        n.pack()
        n.place(x=100,y=60)

        para_menu.pack()
        para_menu.place(x=0, y=80)

        current_para,current_type=self.active_model.get_value_parameter_and_type(tkvar_model_para.get())

        self.val_label = Label(right_frame, text="Value: {0}".format(current_para))
        self.val_label.pack()
        self.val_label.place(x=140, y=80)
        self.type_label = Label(right_frame, text="Type: {0}".format(current_type))
        self.type_label.pack()
        self.type_label.place(x=140, y=100)

        self.entry_val = Entry(right_frame, width=10)
        self.entry_val.pack()
        self.entry_val.place(x=75, y=120)

        change_param = Button(right_frame, text="Change Parameter",command=lambda: self.change_param(tkvar_model_para.get(), self.entry_val.get()))
        change_param.pack()
        change_param.place(x=72, y=150)

        self.accu_label = Label(right_frame)
        self.accu_label.pack()
        self.accu_label.place(x=10, y=180)


    def predict(self):
        right_frame = self.content_model_frame
        bottom_frame=self.content_model_bottom

        self.delete_frame_widgets(right_frame)
        self.delete_frame_widgets(bottom_frame)

        self.active_model.fit_for_finale()


        self.scalevars = []
        for i in range(len(self.independent_variables.get_variable_names())):
            column_name=self.independent_variables.get_variable_names()[i]
            la=Label(right_frame, text=column_name)
            la.pack()
            la.place(x=0,y=0+(i*20))


            self.scalevars.append(DoubleVar())

            self.scalevars[i].set(self.independent_variables.get_variable_mean(column_name))
            max,min=self.independent_variables.get_variable_max_min(column_name)
            #max*scale is max input
            scale = Scale(right_frame, from_=0, to_=max*2,
                             variable=self.scalevars[i], orient="horizontal",showvalue=0)
            label = Entry(right_frame, textvariable=self.scalevars[i],width=5)
            scale.pack()
            scale.place(x=124,y=0+(i*20))
            label.pack()
            label.place(x=234, y=0 + (i * 20))

        Button(right_frame,text="Make prediction",command=self.make_prediction).pack(side="bottom")

        Label(bottom_frame,text="The data implies the following prediction:").pack(side="left",anchor="n")
        self.pre_label=Label(bottom_frame,text="")
        self.pre_label.pack(side="left",anchor="n")

    def make_prediction(self,):
        bottom_frame = self.content_model_bottom
        self.delete_frame_widgets(bottom_frame)

        new_predict=[]
        value_ok = True
        for i in range(len(self.scalevars)):

            column_name=self.independent_variables.get_variable_names()[i]
            value=self.scalevars[i].get()
            try:
                if self.independent_variables.variable_is_rescaled(column_name):
                    new_predict.append(self.independent_variables.rescale_single_value(column_name, value))

                else:
                    new_predict.append(value)
            except ValueError:
                showerror("Wrong format", "The value {0} entered for {1}".format(value, column_name))
                value_ok=False
                break
        if value_ok:

            predict_proba=self.active_model.get_predict_prob_new(new_predict)[0]




            fig = Figure(figsize=(6, 4), dpi=96)
            ax = fig.add_subplot(111)

            N = len(predict_proba)
            x = range(N)

            ax.bar(x, predict_proba,
                   color="r", align="center")



            labels=self.target_variable.get_labels()
            ax.set_xticks(x)
            if self.target_variable.get_number_of_variables()==2:

                ax.set_xticklabels(labels)

            ax.set_title("Prediction")
            ax.set_ylabel('Chance of outcome [%]')
            ax.set_xlabel('Target class')

            canvas = FigureCanvasTkAgg(fig, bottom_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side="top")
            #self.pre_label.configure(text=prediction)


    def performance_metrics(self):
        right_frame = self.content_model_frame
        bottom_frame = self.content_model_bottom

        self.delete_frame_widgets(right_frame)
        self.delete_frame_widgets(bottom_frame)

        mult = MultiListbox(bottom_frame,
                                 (('Accuracy', 8), ('Error', 8), ('Precision', 8),
                                 ('Recall',8),('Specificity',8),('F1', 8)))
        mult.pack()

        acc=self.active_model.get_accuracy()
        err=self.active_model.get_error()
        precision,specificity,recall,f1=self.active_model.get_precision_specificity_recall_f1()
        mult.insert(END, (acc,err,precision,recall,specificity,f1))

        self.create_confusion_matrix(right_frame)

        self.active_model.forward_selection()

    def create_confusion_matrix(self,frame):
        tn,fp,fn,tp=self.active_model.get_confusion_matrix()
        Label(frame,text="Predicted").pack(side="top")
        Label(frame,text="Actual", wraplength=1).pack(side="left")
        mult = MultiListbox(frame,
                     (('', 8),('Positive', 8), ('Negative', 8)))
        mult.pack()
        mult.insert(END,('Positive',tp,fn))
        mult.insert(END, ('Negative', fp, tn))

    def feature_selection(self):
        frame = self.content_model_frame
        bottom_frame = self.content_model_bottom

        self.delete_frame_widgets(bottom_frame)
        self.delete_frame_widgets(frame)

        features=self.active_model.forward_selection()

        if len(features)==0:
            Label(frame, text="The chosen features could not\n outperform the model without features \n"
                              "i.e. the features are not\n suited to predict using the model").pack(side="top")
        else:
            Label(frame,text="According to the feature selection method \n forward selection"
                             "the following features\n are to be chosen for\n the corresponding model").pack(side="top")
            list=Listbox(frame)
            list.pack(side="top")
            for feature in features:
                list.insert(END,feature)

    def change_current_param_val(self,param):
        current_para, current_type = self.active_model.get_value_parameter_and_type(param)
        self.val_label.configure(text="Value: {0}".format(current_para))
        self.type_label.configure (text="Type: {0}".format(current_type))
        self.entry_val.delete(0,END)

    def grid_fit(self):

        old_score = self.active_model.get_accuracy()
        self.active_model.grid_fit()
        self.set_model_scores()
        self.populate_score_comparison(old_score)

    def rand_fit(self):
        old_score = self.active_model.get_accuracy()
        self.active_model.random_search_fit()
        self.set_model_scores()
        self.populate_score_comparison(old_score)

    def change_param(self,param,val):

        old_score=self.active_model.get_accuracy()

        if self.active_model.set_parameter(param,val):
            current_para, current_type = self.active_model.get_value_parameter_and_type(param)
            self.val_label.configure(text="Value: {0}".format(current_para))
            self.type_label.configure(text="Type: {0}".format(current_type))
            self.set_model_scores()
            self.populate_score_comparison(old_score)

        else:
            showerror("Incorrect Parameter Value", "Incorrect Parameter Value")

    def populate_score_comparison(self,old_acc):
        self.accu_label.configure(
            text="Old accuracy: {0} -- new accuracy: {1}"
                .format(old_acc, self.active_model.get_accuracy()))

    def models_are_initialized(self):
        for model in self.active_models:
            if not(model.is_initialized()):
                return False
        return True

    def plot_acc_err(self,graph_type):
        top = Toplevel()
        top.wm_geometry("420x400")
        fig = Figure(figsize=(6, 4), dpi=96)
        ax = fig.add_subplot(111)
        data=[]
        names=[]

        for model in self.active_models:
            names.append(model.get_name())
            if graph_type == "Accuracy":
                top.title("Accuracy")
                data.append(model.get_accuracy_list())

                ax.set_ylabel('Accuracy [%]')
            else:
                top.title("Error")
                data.append(model.get_error_list())
                ax.set_ylabel('Cross-validation error [%]')
        ax.boxplot(data)
        ax.set_xticklabels(names, rotation='vertical')


        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, top)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top")

    def plot_roc(self):
        top = Toplevel()
        top.wm_geometry("600x600")

        y=self.target_variable.get_target_variable()


        X_train, X_test, y_train, y_test = train_test_split(self.independent_variables.get_active_table(), y, test_size=0.25, random_state=42)
        fig = Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        legends=[]
        legends_description=[]
        for model in self.active_models:
            # Compute False postive rate, and True positive rate
            prob_pred=model.get_predict_prob(X_train,y_train,X_test)
            fpr, tpr, thresholds = roc_curve(y_test, prob_pred)
            # Calculate Area under the curve to display on the plot
            roc_auc = auc(fpr, tpr)
            le = ax.plot(fpr, tpr)
            legends.append(le[0])
            legends_description.append( "%s (%0.2f)" % (model.get_name(),roc_auc))

        ax.plot([0, 1], [0, 1], 'k--')
        ax.legend(legends, legends_description, loc="best")
        ax.set_title("ROC curves")
        ax.set_ylabel('True positive rate')
        ax.set_xlabel('False positive rate')


        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack()
        canvas.draw()





