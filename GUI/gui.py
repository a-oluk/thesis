import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

from matplotlib import pyplot as plt

import _params as param
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


class Gui():
    """
        GUI class.

        Implements a GUI to visualize the results
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Main Window")
        #self.root.state('zoom') #Full window or not
        # self.root.geometry("1000x1000")
        # self.root.state('zoomed')
        ''' '''
        # Definition of the Frames in the window
        #       Frame 1 -------- Frame 2
        #          |                |
        #          |                |
        #       Frame 3 -------- Frame 4
        # if True:
        self.frame1 = tk.Frame(self.root, width=700, height=300)
        self.frame2 = tk.Frame(self.root, width=700, height=300)
        self.frame3 = tk.Frame(self.root, width=700, height=300)
        self.frame4 = tk.Frame(self.root, width=700, height=300)

        # pad macht abstand an den Seiten
        self.frame1.grid(row=0, column=0, padx=5, pady=3)
        self.frame2.grid(row=0, column=1, padx=5, pady=3)
        self.frame3.grid(row=1, column=0, padx=5, pady=3)
        self.frame4.grid(row=1, column=1, padx=5, pady=3)

        # DIFFERENT TABS in FRAME BOTTOM LEFT (Frame 3)
        if True:
            self.frame_tabs = ttk.Notebook(self.frame3)
            self.frame_tabs.grid(row=0, column=0, sticky="N")

            def on_tab_selected(event):
                selected_tab = event.widget.select()
                tab_text = event.widget.tab(selected_tab, "text")  # get the text of the tab
                if False:  # set True to see the tab name in console
                    if tab_text == "Tab 1":
                        print("Test Tab 1")

                    if tab_text == "Tab 2":
                        print("Test Tab 2")

                    if tab_text == "Tab 3":
                        print("Test Tab 3")
                return tab_text

            self.tabs_plot = self.frame_tabs.bind("<<NotebookTabChanged>>", on_tab_selected)  # change the tab name always

            # define the tabs
            self.tab_1 = tk.Frame(self.frame_tabs)
            self.frame_tabs.add(self.tab_1, text="Tab 1")

            self.tab_2 = tk.Frame(self.frame_tabs)
            self.frame_tabs.add(self.tab_2, text="Tab 2")

            self.tab_3 = tk.Frame(self.frame_tabs)
            self.frame_tabs.add(self.tab_3, text="Tab 3")

        # Function for SUB-Frame
        def create_sub_frame(parent, frame_text, fill, expand):
            frame = tk.LabelFrame(parent, text=frame_text)
            frame.pack(fill=fill, expand=expand, pady=2, padx=5)
            return frame

        # Function for creating Space for writing
        def create_param_entry(parent, label_text, parameter, row, column):
            pad_x = 5
            pad_y = 5
            label = tk.Label(parent, text=label_text, width=20)
            label.grid(row=row, column=2 * column, padx=pad_x, pady=pad_y)
            entry = tk.Entry(parent, width=10)
            entry.insert(tk.END, str(parameter))
            entry.grid(row=row, column=2 * column + 1, padx=pad_x, pady=pad_y)

            entry.bind('<Return>', self.set_params)
            return label, entry

        self.function = create_sub_frame(self.frame1, "Function", "x", "yes")
        self.frame_conditions = create_sub_frame(self.frame1, "Conditions", "x", "yes")
        self.frame_hyperparam = create_sub_frame(self.frame1, "Hyperparameter", "x", "yes")
        self.frame_testdata = create_sub_frame(self.frame1, "Testdataset", "x", "yes")

        self.subframe_plot = create_sub_frame(self.frame2, "Plot", "x", "yes")

        create_param_entry(self.frame_conditions, " number", param.n, 0, 0)

        self.label_function, self.function = create_param_entry(self.function, " Function", 0.0, 0, 0)

        self.label_param_n, self.param_n = create_param_entry(self.frame_conditions, " number", param.n, 0, 0)
        self.label_param_sigma, self.param_sigma = create_param_entry(self.frame_conditions, " sigma", param.sigma, 1,
                                                                      0)
        self.label_param_mu, self.param_mu = create_param_entry(self.frame_conditions, " mü", param.mu, 2, 0)
        self.label_param_RESME, self.param_RESME = create_param_entry(self.frame_conditions, " mü", param.RESME, 3, 0)

        self.label_param_alpha, self.param_alpha = create_param_entry(self.frame_hyperparam, " alpha", param.alpha, 0,
                                                                      0)
        self.label_param_N_0, self.param_N_0 = create_param_entry(self.frame_hyperparam, " N_0 init datasize",
                                                                  param.N_0, 1, 0)
        self.label_param_N, self.param_N = create_param_entry(self.frame_hyperparam, " N iterations", param.N, 2, 0)

        self.label_test_data_size, self.param_test_data_size = create_param_entry(self.frame_testdata, " Test Data Size", param.test_data_size, 0,
                                                                      0)

        ########################## CREATES THE PLOT AT START OF THE GUI ###############################################
        if True:
            plot_size = (6,3)
            def create_plot():
                fig = plt.figure(figsize=plot_size, dpi=100)
                fig.set_tight_layout(True)

                self.gui_plt = fig.add_subplot(111)  # 111 means 1. row  2.column 3. number of the plot
                # self.gui_plt_histogram.cla()
                self.gui_plt.set_xlabel('x-axis')
                self.gui_plt.set_ylabel('y-axis')

                self.frame_plot = tk.Frame(self.subframe_plot)
                self.frame_plot.grid(column=0, row=0, sticky='NS')

                self.canvas_histogram = FigureCanvasTkAgg(fig, master=self.frame_plot)  # A tk.DrawingArea.
                self.canvas_histogram.draw()
                self.canvas_histogram.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        create_plot()
        # Buttons For Use of Aquisition Function
        if True:
            font_txt = tkFont.Font(size=15)

            self.aquisition_fct_ei = tk.IntVar(value=1)
            self.aquisition_fct_eigf = tk.IntVar(value=0)
            self.aquisition_fct_other = tk.IntVar(value=0)

            frame_checkbutton= self.frame4

            tk.Label(frame_checkbutton, text="Aquisition Function", width=25, font=font_txt).grid(row=0, column=0)

            button1 = tk.Checkbutton(frame_checkbutton, text="Expected Improvement",
                                     variable=self.aquisition_fct_ei,
                                     onvalue=1,
                                     offvalue=0)
            button2 = tk.Checkbutton(frame_checkbutton, text="Expected Improvement for Global Fit",
                                     variable=self.aquisition_fct_eigf,
                                     onvalue=1,
                                     offvalue=0)
            button3 = tk.Checkbutton(frame_checkbutton, text="ADDITIONAL",
                                     variable=self.aquisition_fct_other,
                                     onvalue=1,
                                     offvalue=0)
            # organize the button position and the text before
            button1.grid(row=1, column=0,sticky='W')
            button2.grid(row=2, column=0,sticky='W')
            button3.grid(row=3, column=0,sticky='W')

        # Buttons For Use of Aquisition Function
        if True:
            self.var_kernel_button = tk.IntVar(value=0)

            frame_checkbutton= self.frame4

            tk.Label(frame_checkbutton, text="Kernel Function", width=25, font=font_txt).grid(row=5, column=0)

            button1 = tk.Radiobutton(frame_checkbutton, text="Radial Basis Functions",
                                     variable=self.var_kernel_button,
                                     value= 0)
            button2 = tk.Radiobutton(frame_checkbutton, text="Periodic Kernel",
                                     variable=self.var_kernel_button,
                                     value= 1)
            button3 = tk.Radiobutton(frame_checkbutton, text="Linear",
                                     variable=self.var_kernel_button,
                                     value= 2)
            # organize the button position and the text before
            button1.grid(row=6, column=0,sticky='W')
            button2.grid(row=7, column=0,sticky='W')
            button3.grid(row=8, column=0,sticky='W')




    # Function for setting parameters
    # it only works, if there is the parameter also available
    def set_params(self, event=None):
        del event
        param.n = float(self.param_n.get())
        param.sigma = float(self.param_sigma.get())
        param.mu = float(self.param_mu.get())
        param.alpha = float(self.param_alpha.get())

        # Abbruchbedingungen
        param.N_0 = float(self.param_N_0.get())
        param.N = float(self.param_N.get())

        param.RESME = float(self.param_RESME.get())



    def show(self):
        self.root.mainloop()

    def destroy(self):
        self.root.destroy()


g = Gui()
g.show()
