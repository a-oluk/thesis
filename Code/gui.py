from GPR import GaussProcessRegression
from data_storage import Data
from PerformanceAnalysis import Acquisition, Metrics
import _params as param

import datetime
import threading
import time
import ast

import sympy as sp
import numpy as np

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.ticker import MaxNLocator


class Gui():
    """
        GUI class

        Implements a GUI to be able to adapt parameters without going in the code and visualize the results
    """

    def __init__(self):
        '''
        Initialize all Elements for the GUI
        '''
        self.root = tk.Tk()
        self.root.title("GUI for GPR")

        self.data = None
        self.metrics = None
        self.acquisition = None
        self.gpr = None

        if True:
            self.time = 0
            self.elapsed_time = tk.StringVar()
            self.elapsed_time.set("00:00:00")

        # Definition of the Frames in the window
        #       Frame 1 -------- Frame 2
        #          |                |
        #          |                |
        #       Frame 3 -------- Frame 4
        if True:
            self.frame1 = tk.Frame(self.root, width=700, height=300)
            self.frame2 = tk.Frame(self.root, width=700, height=300)
            self.frame3 = tk.Frame(self.root, width=700, height=300)
            self.frame4 = tk.Frame(self.root, width=700, height=300)

            self.frame1.grid(row=0, column=0, padx=5, pady=3, sticky="nw")
            self.frame2.grid(row=0, column=1, padx=5, pady=3, sticky="nw")
            self.frame3.grid(row=1, column=0, padx=5, pady=3, sticky="nw")
            self.frame4.grid(row=1, column=1, padx=5, pady=3, sticky="nw")

        if True:
            self.gui_plt, self.canvas_figure = None, None
            self.gui_plt_, self.canvas_figure_ = None, None

        if True:
            self.function_individuel = None
            self.dim_individuel = None

        # Helpfunctions for subframe and parameter label + entry

        if True:
            def create_sub_frame(parent, frame_text, fill="x", expand="yes", pad_y=2, pad_x=5):
                frame = tk.LabelFrame(parent, text=frame_text)
                frame.pack(fill=fill, expand=expand, pady=pad_y, padx=pad_x)
                return frame

            def create_sub_frame_grid(parent, frame_text, row_, column_, sticky_="nw", pad_y=2, pad_x=5):
                frame = tk.LabelFrame(parent, text=frame_text)
                frame.grid(row=row_, column=column_, pady=pad_y, padx=pad_x, sticky=sticky_)
                return frame

            # parameter slots
            def create_param_entry(parent, label_text, parameter, row, column=0, pad_x=5, pad_y=5, width_label=18,
                                   width_entry=15, big_entry=False):

                label = tk.Label(parent, text=label_text, width=width_label, anchor="w", justify='left')
                label.grid(row=row, column=2 * column, padx=pad_x, pady=pad_y)
                if big_entry:
                    entry = tk.Entry(parent, width=33)
                else:
                    entry = tk.Entry(parent, width=width_entry)
                entry.insert(tk.END, str(parameter))
                entry.grid(row=row, column=2 * column + 1, padx=pad_x, pady=pad_y)

                entry.bind('<Return>', self.set_params)
                return label, entry

        # CREATE SUBFRAMES
        if True:
            self.frame_function = create_sub_frame(self.frame1, "Function")
            self.frame_param_gpr = create_sub_frame(self.frame1, "GPR Parameter")
            self.frame_hyperparam = create_sub_frame(self.frame1, "Hyperparameter")
            self.frame_datasize = create_sub_frame(self.frame1, "Dataset")

            self.subframe_plot = create_sub_frame_grid(self.frame2, "Plot", row_=0, column_=0)

            self.frame_checkbutton = create_sub_frame(self.frame3, "Dataselection")

            self.frame_4top = tk.Frame(self.frame4)
            self.frame_4down = tk.Frame(self.frame4)

            self.frame_4top.grid(row=0, column=0, padx=5, sticky="nsew")
            self.frame_4down.grid(row=1, column=0, padx=5, sticky="nsew")

            self.frame_individual_function = create_sub_frame_grid(self.frame_4top, "Individual Function", row_=0,
                                                                   column_=0)
            self.frame_example = create_sub_frame_grid(self.frame_4down, "Examples", row_=1, column_=0)
            self.frame_control = create_sub_frame_grid(self.frame_4down, "Control", row_=1, column_=1)

        #  Label und Entry for the Parameter from Frame 1
        if True:
            self.label_function, self.param_function = create_param_entry(self.frame_function, " Function",
                                                                          param.example_txt, 0)

            self.label_individual_function, self.param_individual_function = create_param_entry(
                self.frame_individual_function, " Enter Individual function here:",
                "", 0, big_entry=True)
            self.label_symbol, self.param_symbol = create_param_entry(self.frame_individual_function, " Symbols:", "",
                                                                      1,big_entry=True)
            self.label_SSN, self.param_SSN = create_param_entry(self.frame_param_gpr, "start,stop,number", param.SSN, 0)

            self.label_iterations, self.param_iterations = create_param_entry(self.frame_param_gpr, "Iteration(s)",
                                                                              param.iterations, 1)

            self.label_repetitions, self.param_repetitions = create_param_entry(self.frame_param_gpr, "Repetition(s)",
                                                                                param.repetitions, 2)

            self.label_init_data_size, self.param_init_data_size = create_param_entry(self.frame_datasize,
                                                                                      "Initial datasize",
                                                                                      param.init_data_size, 0)

            self.label_test_data_size, self.param_test_data_size = create_param_entry(self.frame_datasize,
                                                                                      " Test datasize",
                                                                                      param.test_data_size, 1)

            self.label_param_alpha, self.param_alpha = create_param_entry(self.frame_hyperparam, " alpha", param.alpha,
                                                                          0)

        ''' 
        Create ITEMS of Control Window
                - Repetition / Iteration Number
                - step and start button
                - time of last iteration
        '''
        if True:
            self.label_rep_nr = tk.Label(self.frame_control, text="Repetition number", width=20)
            self.label_rep_nr.grid(row=0, column=0, padx=5)
            self.label_rep_nr_value = tk.Label(self.frame_control, text="0", width=10)
            self.label_rep_nr_value.grid(row=0, column=1, padx=5)

            self.label_iter_nr = tk.Label(self.frame_control, text="Iteration number", width=20)
            self.label_iter_nr.grid(row=1, column=0, padx=5)
            self.label_iter_nr_value = tk.Label(self.frame_control, text="0", width=10)
            self.label_iter_nr_value.grid(row=1, column=1, padx=5)

            self.label_iter_time = tk.Label(self.frame_control, text="time per iteration (h:m:s.s)", width=20)
            self.label_iter_time.grid(row=2, column=0, padx=5)
            self.iter_time_value = tk.Label(self.frame_control, text="0:00:0.000", width=10)
            self.iter_time_value.grid(row=2, column=1, padx=5)

            self.btn_gpr_step = tk.Button(self.frame_control, text='step', width=15, command=self.step_gpr)
            self.btn_gpr_step.grid(row=4, column=0, padx=5)

            self.btn_gpr_start = tk.Button(self.frame_control, text='start GPR', width=15,
                                           command=self.start_gpr_threading)
            self.btn_gpr_start.grid(row=4, column=1, sticky="W", padx=2, pady=0)

            self.btn_stop = tk.Button(self.frame_control, text='STOP', width=15,
                                      command=self.stop)
            self.btn_stop.grid(row=5, column=1, sticky="W", padx=2, pady=0)

            #def show_plot():
            #    plt.show(block=False)

            #self.btn_show_plots = tk.Button(self.frame_control, text='show plots', width=15,command=show_plot)
            #self.btn_show_plots.grid(row=6, column=1, sticky="W", padx=2, pady=0)

        # create plots
        if True:
            def create_plot():
                plot_size = (5, 3)
                fig = plt.figure(figsize=plot_size, dpi=100)
                fig.set_tight_layout(True)

                self.gui_plt = fig.add_subplot(111)
                # self.gui_plt_histogram.cla()
                self.gui_plt.set_xlabel('x-axis')
                self.gui_plt.set_ylabel('y-axis')

                self.frame_plot = tk.Frame(self.subframe_plot)
                self.frame_plot.grid(column=0, row=0, sticky='NS')

                self.canvas_figure = FigureCanvasTkAgg(fig, master=self.frame_plot)
                self.canvas_figure.draw()
                self.canvas_figure.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                toolbar = NavigationToolbar2Tk(self.canvas_figure, self.frame_plot)
                toolbar.update()

                def on_key_press(event):
                    key_press_handler(event, self.canvas_figure, toolbar)

                self.canvas_figure.mpl_connect("key_press_event", on_key_press)

            def create_plot_():
                plot_size = (5, 3)
                fig = plt.figure(figsize=plot_size, dpi=100)
                fig.set_tight_layout(True)

                self.gui_plt_ = fig.add_subplot(111)
                self.gui_plt_.cla()
                self.gui_plt_.set_xlabel('x-axis')
                self.gui_plt_.set_ylabel('y-axis')

                self.frame_plot = tk.Frame(self.subframe_plot)
                self.frame_plot.grid(column=1, row=0, sticky='NS')

                self.canvas_figure_ = FigureCanvasTkAgg(fig, master=self.frame_plot)
                self.canvas_figure_.draw()
                self.canvas_figure_.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                toolbar = NavigationToolbar2Tk(self.canvas_figure_, self.frame_plot)
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                toolbar.update()

                def on_key_press(event):
                    key_press_handler(event, self.canvas_figure_, toolbar)

                self.canvas_figure.mpl_connect("key_press_event", on_key_press)

            create_plot()
            create_plot_()
        # Buttons for plot window
        if True:
            self.frame_plt_btn = tk.Frame(self.frame2)
            self.frame_plt_btn.grid(row=1, column=0, padx=5, sticky="nw")

            self.btn_plt = tk.Button(self.frame_plt_btn, text="refresh r2 plot", command=self.plot, height = 1, width = 14)
            self.btn_plt.grid(row=0, column=0, padx=2, pady=2, sticky="w")
            self.btn_plt = tk.Button(self.frame_plt_btn, text="add plot r2-score", command=self.add_plot_r2, height = 1, width = 14)
            self.btn_plt.grid(row=1, column=0, padx=2, pady=2, sticky="w")

            padding_cell = tk.Label(self.frame_plt_btn, width=30)
            padding_cell.grid(row=0, column=2)

            self.btn_plt = tk.Button(self.frame_plt_btn, text="refresh rsme plot", command=self.plot_, height = 1, width = 14)
            self.btn_plt.grid(row=0, column=3, padx=2, pady=2, sticky="w")

            self.btn_plt = tk.Button(self.frame_plt_btn, text="add plot rsme-score", command=self.add_plot_rsme, height = 1, width = 14)
            self.btn_plt.grid(row=1, column=3, padx=2, pady=2, sticky="w")

        # (un-)show average plot
        def toggle_line(var):
            if var == self.show_average_r2_score:
                if var.get():
                    self.average_r2_line.set_visible(True)
                else:
                    self.average_r2_line.set_visible(False)
                # get the visible lines
                lines = [line for line in self.gui_plt.lines if line.get_visible()]
                # update the legend
                self.gui_plt.legend(lines, [line.get_label() for line in lines], loc="best")
                self.canvas_figure.draw()

            elif var == self.show_average_rsme_score:
                if var.get():
                    self.average_rsme_line.set_visible(True)
                else:
                    self.average_rsme_line.set_visible(False)
                # get the visible lines
                lines = [line for line in self.gui_plt_.lines if line.get_visible()]
                # update the legend
                self.gui_plt_.legend(lines, [line.get_label() for line in lines], loc="best")
                self.canvas_figure_.draw()
            elif var == self.show_iterations_r2:
                if var.get():
                    for line in self.r2_iter_lines:
                        line.set_visible(True)
                else:
                    for line in self.r2_iter_lines:
                        line.set_visible(False)
                lines = [line for line in self.gui_plt.lines if line.get_visible()]
                # update the legend
                self.gui_plt.legend(lines, [line.get_label() for line in lines], loc="best")
                self.canvas_figure.draw()

            elif var == self.show_iterations_rsme:
                if var.get():
                    for line in self.rsme_iter_lines:
                        line.set_visible(True)
                else:
                    for line in self.rsme_iter_lines:
                        line.set_visible(False)
                lines = [line for line in self.gui_plt_.lines if line.get_visible()]
                # update the legend
                self.gui_plt_.legend(lines, [line.get_label() for line in lines], loc="best")
                self.canvas_figure_.draw()
            else:
                return None

        # show scores
        if True:
            self.show_iterations_r2 = tk.BooleanVar(value=True)
            self.show_iterations_rsme = tk.BooleanVar(value=True)

            tk.Checkbutton(self.frame_plt_btn, text="show iteration R2-score",
                           variable=self.show_iterations_r2,
                           onvalue=1,
                           offvalue=0, command=lambda: toggle_line(self.show_iterations_r2)).grid(row=0, column=1,
                                                                                                  sticky='W')

            tk.Checkbutton(self.frame_plt_btn, text="show iteration rsme-score",
                           variable=self.show_iterations_rsme,
                           onvalue=1,
                           offvalue=0, command=lambda: toggle_line(self.show_iterations_rsme)).grid(row=0, column=4,
                                                                                                    sticky='W')
        # show average score Checkbutton
        if True:
            self.show_average_r2_score = tk.BooleanVar(value=False)
            self.show_average_rsme_score = tk.BooleanVar(value=False)

            tk.Checkbutton(self.frame_plt_btn, text="show average R2-score",
                           variable=self.show_average_r2_score,
                           onvalue=1,
                           offvalue=0, command=lambda: toggle_line(self.show_average_r2_score)).grid(row=1, column=1,
                                                                                                     sticky='W')
            tk.Checkbutton(self.frame_plt_btn, text="show average rsme score",
                           variable=self.show_average_rsme_score,
                           onvalue=1,
                           offvalue=0, command=lambda: toggle_line(self.show_average_rsme_score)).grid(row=1, column=4,
                                                                                                       sticky='W')
        # show legend checkbutton + log-scale for rsme-plot checkbutton
        if True:
            self.var_legend = tk.BooleanVar(value=True)
            self.var_legend_ = tk.BooleanVar(value=True)
            self.var_log_scale_ = tk.BooleanVar(value=False)
            def scale_plot_(var):
                if var.get():
                    self.gui_plt_.set_yscale("log")
                else:
                    self.gui_plt_.set_yscale("linear")
                self.canvas_figure_.draw()

            def show_hide_legend_(var):
                if var.get():
                    lines = [line for line in self.gui_plt_.lines if line.get_visible()]
                    # update the legend
                    self.gui_plt_.legend(lines, [line.get_label() for line in lines], loc="best")
                else:
                    self.gui_plt_.legend_.remove()
                self.canvas_figure.draw()

            def show_hide_legend(var):
                if var.get():
                    lines = [line for line in self.gui_plt.lines if line.get_visible()]
                    # update the legend
                    self.gui_plt.legend(lines, [line.get_label() for line in lines], loc="best")
                else:
                    self.gui_plt.legend_.remove()
                self.canvas_figure.draw()

            tk.Checkbutton(self.frame_plt_btn, text="show legend", onvalue=1,
                           offvalue=0, variable=self.var_legend,
                           command=lambda: show_hide_legend(self.var_legend)).grid(
                row=0, column=2, sticky='W')

            tk.Checkbutton(self.frame_plt_btn, text="show legend", onvalue=1,
                           offvalue=0, variable=self.var_legend_,
                           command=lambda: show_hide_legend_(self.var_legend_)).grid(
                row=0, column=5, sticky='W')
            tk.Checkbutton(self.frame_plt_btn, text="logarithmic y-axis?", onvalue=1,
                           offvalue=0, variable=self.var_log_scale_,
                           command=lambda: scale_plot_(self.var_log_scale_)).grid(
                row=0, column=6, sticky='W')


        # Individual BLOCK
        if True:
            def convert_function():
                func_str = self.param_individual_function.get()
                symbols_str = self.param_symbol.get()

                if ',' in symbols_str or ' ' in symbols_str:
                    symbols = tuple(sp.symbols(symbols_str))
                else:
                    symbols = (sp.symbols(symbols_str),)

                func = sp.lambdify(symbols, func_str, 'numpy')
                self.function_str.config(text=str(sp.sympify(func_str)))

                vectorized_func = np.vectorize(func)

                def matrix_func(X):
                    return vectorized_func(*[X[:, i] for i in range(param.dim)])

                self.function_individuel = matrix_func
                self.dim_individuel = len(symbols)

            self.label_dim, self.param_dim = create_param_entry(self.frame_function, "Dimension",
                                                                param.dim, 2)

            self.func_button = tk.Button(self.frame_individual_function, text="Convert to pyhton function",
                                         command=convert_function)
            self.func_button.grid(row=0, column=2, padx=5, pady=5)

            tk.Label(self.frame_individual_function, text="Function:").grid(row=0, column=3, padx=5, pady=5)
            self.function_str = tk.Label(self.frame_individual_function, text="type function and convert", width=25)
            self.function_str.grid(row=0, column=4, padx=5, pady=5, rowspan=2, columnspan=3, sticky="nw")

        # Buttons For Use of Aquisition Function
        if True:
            font_txt = tkFont.Font(size=15)

            self.var_aquisition_fct = tk.IntVar(value=1)  # standard is expected improvement
            self.var_aquisition_fct_normalize = tk.IntVar(value=1)  # standart is to normalize

            tk.Radiobutton(self.frame_checkbutton, text="Random",
                           variable=self.var_aquisition_fct,
                           value=2).grid(row=1, column=0, sticky='W')

            tk.Radiobutton(self.frame_checkbutton, text="Variance based ",
                           variable=self.var_aquisition_fct,
                           value=0).grid(row=2, column=0, sticky='W')
            tk.Radiobutton(self.frame_checkbutton, text="Expected Improvement for Global Fit",
                           variable=self.var_aquisition_fct,
                           value=1).grid(row=3, column=0, sticky='W')
            #tk.Label(self.frame_checkbutton, text="").grid(row=3, column=0)
            tk.Checkbutton(self.frame_checkbutton, text="Normalize",
                           variable=self.var_aquisition_fct_normalize,
                           onvalue=1,
                           offvalue=0).grid(row=4, column=0, sticky='W')
        # Buttons For Use of Kernel Function
        if True:
            self.var_kernel_button = tk.IntVar(value=0)

            self.frame_kernel = create_sub_frame(self.frame3, "Kernel Function")

            tk.Radiobutton(self.frame_kernel, text="Gauss'scher Kernel",
                           variable=self.var_kernel_button,
                           value=0).grid(row=6, column=0, sticky='W')
            tk.Radiobutton(self.frame_kernel, text="Periodic Kernel",
                           variable=self.var_kernel_button,
                           value=1).grid(row=8, column=0, sticky='W')
            tk.Radiobutton(self.frame_kernel, text="Linearer Kernel",
                           variable=self.var_kernel_button,
                           value=2).grid(row=10, column=0, sticky='W')

            self.label_lengthscale_rbf, self.param_rbf_lengthscale = create_param_entry(self.frame_kernel,
                                                                                        "lengthscale",
                                                                                        param.rbf_lengthscale, 6, 1,
                                                                                        width_label=10,
                                                                                        width_entry=3)
            self.label_variance_rbf, self.param_rbf_variance = create_param_entry(self.frame_kernel, "variance",
                                                                                  param.rbf_variance, 7, 1,
                                                                                  width_label=10,
                                                                                  width_entry=3)
            self.label_lengthscale, self.param_perio_lengthscale = create_param_entry(self.frame_kernel, "lengthscale",
                                                                                      param.perio_lengthscale, 8, 1,
                                                                                      width_label=10,
                                                                                      width_entry=3)
            self.label_periodicity, self.param_perio_periodicity = create_param_entry(self.frame_kernel, "periodicity",
                                                                                      param.perio_periodicity, 9, 1,
                                                                                      width_label=10,
                                                                                      width_entry=3)
            self.label_intercept, self.param_lin_intercept = create_param_entry(self.frame_kernel, "intercept",
                                                                                param.lin_intercept, 10, 1,
                                                                                width_label=10,
                                                                                width_entry=3)
            self.label_slope, self.param_lin_slope = create_param_entry(self.frame_kernel, "slope",
                                                                        param.lin_slope, 11, 1,
                                                                        width_label=10,
                                                                        width_entry=3)

        # Buttons For Use of Example
        if True:
            # self.var_example = tk.IntVar(value=0)
            self.var_example = tk.DoubleVar(value=1.1)

            tk.Label(self.frame_example, text="One Dimensional").grid(row=0, column=0, sticky='w')
            tk.Label(self.frame_example, text="Two Dimensional").grid(row=2, column=0, sticky='w')
            tk.Label(self.frame_example, text="Three Dimensional").grid(row=4, column=0, sticky='w')
            tk.Label(self.frame_example, text="Four Dimensional").grid(row=5, column=0, sticky='w')
            tk.Label(self.frame_example, text="Individual Function").grid(row=7, column=0, sticky='w')

            def create_btn_example(func_name, value, r, c=1):
                tk.Radiobutton(self.frame_example, text=func_name,
                               variable=self.var_example,
                               value=value).grid(row=r, column=c, sticky="W")

            create_btn_example(func_name="f₁(x)", value=1.1, r=0)
            create_btn_example(func_name="f₂(x)", value=1.2, r=0,c=2)
            create_btn_example(func_name="f₃(x)", value=1.3, r=1)
            create_btn_example(func_name="f₄(x)", value=1.4, r=1,c=2)

            create_btn_example(func_name="g₁(x₁,x₂)", value=2.1, r=2)
            create_btn_example(func_name="g₂(x₁,x₂)", value=2.2, r=2,c=2)
            create_btn_example(func_name="g₃(x₁,x₂)", value=2.3, r=3)
            create_btn_example(func_name="g₄(x₁,x₂)", value=2.4, r=3,c=2)

            create_btn_example(func_name="h(x₁,x₂,x₃)", value=3.1, r=4)

            create_btn_example(func_name="l(x₁,x₂,x₃,x₄)", value=4.1, r=5)

            tk.Radiobutton(self.frame_example, text="Type function into \nentry and convert",
                           variable=self.var_example,
                           value=999).grid(row=7, column=1, sticky="W")
            #create_btn_example(func_name="Type function into entry and convert", value=999, r=7)

            # Create a blank row with Separator widget
            #sep = ttk.Separator(self.frame_example, orient="horizontal")
            #sep.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

            #tk.Label(self.frame_example, text="", width=10).grid(row=0, column=2)
            #tk.Label(self.frame_example, text="", width=10).grid(row=1, column=2)

            self.state_frame = tk.LabelFrame(self.frame_example, text="", bg="#c0c0c0")
            self.state_frame.grid(row=3, column=3, sticky="W", padx=2, pady=0)

            self.btn_load_param = tk.Button(self.frame_example, text='Load Parameter',
                                            command=self.load_params, height = 1, width = 14)  # load params for example
            self.btn_load_param.grid(row=0, column=3, padx=2, pady=0)

            self.btn_create_gpr_model = tk.Button(self.frame_example, text='Create GPR model',
                                                  command=self.create_gpr_model, height = 1, width = 14)
            self.btn_create_gpr_model.grid(row=1, column=3, padx=2, pady=0)

            #tk.Label(self.frame_example, text="").grid(row=2, column=2)

            self.label_state = tk.Label(self.state_frame, text="", bg="#c0c0c0", width=14)
            self.label_state.grid(row=0, column=0, sticky="W", padx=2, pady=0)

            def show_true_function():
                start, stop, num = param.SSN
                num = (stop - start) * 20
                linspaces = []
                for i in range(param.dim):
                    linspaces.append(np.linspace(start, stop, num))
                grids = np.meshgrid(*linspaces)

                if param.dim == 1:
                    x = grids[0]
                    y = param.function(x)
                    plt.figure()
                    plt.plot(x, y)

                if param.dim == 2:
                    X = np.vstack([grid.flatten() for grid in grids]).T
                    Z = param.function(X)
                    plt.imshow(Z.reshape(num, num), extent=[start, stop, start, stop], origin='lower')
                    plt.colorbar()

                plt.show(block=False)

            #self.btn_load_param = tk.Button(self.frame_example, text='Show True Function', bg="#c0c0c0",command=show_true_function)
            #self.btn_load_param.grid(row=5, column=3, padx=2, pady=0)

    # create the plot windows
    def plot(self):

        self.gui_plt.cla()
        self.gui_plt.plot_size = (5, 3)
        self.gui_plt.tick_params(top=False, right=False)
        # self.gui_plt.set_title("")
        self.gui_plt.set_xlabel("Iterations")
        self.gui_plt.set_ylabel("R2-score")
        self.gui_plt.set_ylim(0, 1.05)
        self.gui_plt.axhline(y=1, ls="--", c="grey")

        self.canvas_figure.draw()

    def plot_(self):
        self.gui_plt_.cla()
        # self.gui_plt_.plot_size = (5, 3)
        # self.gui_plt_.set_title("")
        self.gui_plt_.set_xlabel("Iterations")
        self.gui_plt_.set_ylabel("RSME-Score")
        self.gui_plt_.tick_params(top=False, right=False)
        self.canvas_figure_.draw()

    # plot the scores
    def add_plot_r2(self):
        if self.data.iteration == param.iterations + 1 and self.data.repetition == param.repetitions:
            x = np.arange(self.data.iteration)

            self.r2_iter_lines = []
            for i in np.arange(np.array(self.data.r2_score__).shape[0]):
                y = np.array(self.data.r2_score__[i])
                y[y < 0] = 0
                line, = self.gui_plt.plot(x, y, color=plt.cm.Blues(0.4 + (i / param.repetitions) * 0.4),
                                          label='Repetition {}'.format(i + 1))

                self.r2_iter_lines.append(line)

            if not self.show_iterations_r2.get():
                for line in self.r2_iter_lines:
                    line.set_visible(False)

            self.average_r2_line, = self.gui_plt.plot(x, self.data.r2_score__average, label='average')

            if not self.show_average_r2_score.get():
                self.average_r2_line.set_visible(False)

            self.visible_lines_r2 = [line for line in self.gui_plt.lines if line.get_visible()]

            self.gui_plt.legend(self.visible_lines_r2, [line.get_label() for line in self.visible_lines_r2])
        self.gui_plt.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.canvas_figure.draw()
        return None

    def add_plot_rsme(self):
        # if self.gpr.iter_nr == param.iterations+1 and self.gpr.rep_nr == param.repetitions:
        if self.data.iteration == param.iterations + 1 and self.data.repetition == param.repetitions:
            x = np.arange(self.data.iteration)

            self.rsme_iter_lines = []
            for i in np.arange(np.array(self.data.rsme_score__).shape[0]):
                y = np.array(self.data.rsme_score__[i])
                y[y < 0] = 0
                line, = self.gui_plt_.plot(x, y, color=plt.cm.Blues(0.4 + (i / param.repetitions) * 0.4),
                                           label='Repetition {}'.format(i + 1))
                self.rsme_iter_lines.append(line)

            if not self.show_iterations_rsme.get():
                for line in self.rsme_iter_lines:
                    line.set_visible(False)

            self.average_rsme_line, = self.gui_plt_.plot(x, self.data.rsme_score__average, label='average')

            if not self.show_average_rsme_score.get():
                self.average_rsme_line.set_visible(False)

            self.visible_lines_rsme = [line for line in self.gui_plt_.lines if line.get_visible()]
            self.gui_plt_.legend(self.visible_lines_rsme, [line.get_label() for line in self.visible_lines_rsme])
        self.gui_plt.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.canvas_figure_.draw()
        return None

    # load parameters in the entries
    def load_params(self):
        def delete_params():
            widgets_to_clear = (self.param_function, self.param_dim, self.param_SSN, self.param_iterations,
                                self.param_repetitions, self.param_init_data_size, self.param_test_data_size,
                                self.param_alpha, self.param_rbf_lengthscale, self.param_rbf_variance,
                                self.param_perio_lengthscale, self.param_perio_periodicity)
            for widget in widgets_to_clear:
                widget.delete(0, tk.END)

        if self.var_example.get() == 999:  # INDIVIDUAL FUNCTON GET
            param.example_txt = "individual"
            self.param_function.delete(0, tk.END)
            self.param_function.insert(0, param.example_txt)
            self.param_dim.delete(0, tk.END)
            self.param_dim.insert(0, param.example_txt)
            self.label_state.config(text="Individual Loaded")  # Let User Parameter loaded
        else:
            delete_params()

            param.example_txt, param.kernel, param.function, param.SSN, param.dim, param.iterations, param.repetitions, \
            param.init_data_size, param.test_data_size, param.alpha = param.get_params(self.var_example.get())

            param.rbf_lengthscale, param.rbf_variance, param.perio_lengthscale, param.perio_periodicity, param.lin_slope, param.lin_intercept \
                = param.init_kernel_params()

            if True:
                self.param_function.insert(0, param.example_txt)
                self.param_dim.insert(0, str(param.dim))

                self.param_SSN.insert(0, str(param.SSN))

                self.param_alpha.insert(0, str(param.alpha))
                self.param_iterations.insert(0, str(param.iterations))
                self.param_repetitions.insert(0, str(param.repetitions))

                self.param_init_data_size.insert(0, str(param.init_data_size))
                self.param_test_data_size.insert(0, str(param.test_data_size))

                self.param_rbf_lengthscale.insert(0, str(param.rbf_lengthscale))
                self.param_rbf_variance.insert(0, str(param.rbf_variance))
                self.param_perio_lengthscale.insert(0, str(param.perio_lengthscale))
                self.param_perio_periodicity.insert(0, str(param.perio_periodicity))

        self.label_state.config(text="Parameter loaded")  # Let User Parameter loaded
        return None

    # set the params
    def set_params(self, event=None):
        def get_individual_function():
            func_str = self.param_individual_function.get()
            symbols_str = self.param_symbol.get()

            if ',' in symbols_str or ' ' in symbols_str:
                symbols = tuple(sp.symbols(symbols_str))
            else:
                symbols = (sp.symbols(symbols_str),)

            func = sp.lambdify(symbols, func_str, 'numpy')
            self.function_str.config(text=str(sp.sympify(func_str)))

            vectorized_func = np.vectorize(func)

            def matrix_func(X):
                return vectorized_func(*[X[:, i] for i in range(param.dim)])

            return matrix_func, len(symbols)

        del event

        if self.var_example.get() == 999:  # individual case
            if self.function_individuel is None and self.dim_individuel is None:
                param.function, param.dim = get_individual_function()
            else:
                param.function, param.dim = self.function_individuel, self.dim_individuel
        else:
            print(self.param_function.get())
            param.function, param.dim = param.get_function(self.param_function.get())

        # GPR
        param.SSN = tuple(int(x) for x in ast.literal_eval(self.param_SSN.get()))
        param.iterations = int(self.param_iterations.get())
        param.repetitions = int(self.param_repetitions.get())

        # Dataset
        param.init_data_size = int(self.param_init_data_size.get())
        param.test_data_size = int(self.param_test_data_size.get())

        # Aquisition Function
        param.alpha = float(self.param_alpha.get())

        # Kernel
        param.rbf_lengthscale = float(self.param_rbf_lengthscale.get())
        param.rbf_variance = float(self.param_rbf_variance.get())
        param.perio_lengthscale = float(self.param_perio_lengthscale.get())
        param.perio_periodicity = float(self.param_perio_periodicity.get())

    # update control window
    def set_control_values(self):
        self.label_iter_nr_value.config(text="{}/{}".format(0, param.iterations))
        self.label_rep_nr_value.config(text="{}/{}".format(0, param.repetitions))
        self.iter_time_value.config(text="0:00:0.000")

    def set_control_time(self, iteration_time):
        td = datetime.timedelta(seconds=iteration_time)
        try:
            hmsm = str(td).split('.')[0] + '.' + str(td).split('.')[1][:3]
        except IndexError:
            hmsm = "0.000"
        self.iter_time_value.config(text="{}".format(hmsm))  # configure the time value
        print("Iteration Finished - Time: {}".format(hmsm))

    # Create initial data and create Data Object
    def create_init_data(self):
        start, stop, num = param.SSN
        X_init = np.random.uniform(start, stop, (param.init_data_size, param.dim))
        Y_init = param.function(X_init)
        self.data = Data(X_init, Y_init)

    # get y value for x
    def get_y_new(self, x):
        return param.function(x)

    # reset the scores and initialize new data
    def prepare_data_for_new_repetition(self):
        self.data.reset_scores()  # reset the score and new datapoints
        X_init_new, Y_init_new = self.data.new_init_data(param.SSN, param.init_data_size, param.dim,
                                                         param.function)  # here create new Initial Dataparam.example_txt
        self.data.reset_init_data(X_init_new, Y_init_new)  # reset the Dataset in Data for new repetition
        self.gpr.reset_init_data(X_init_new, Y_init_new)  # reset the Dataset in GPR for new repetition

    # set the type of acquisition function and create object
    def init_acquisition_func(self):
        if self.var_aquisition_fct.get() == 1:
            ei_term, gf_term = True, True
        if self.var_aquisition_fct.get() == 0:
            ei_term, gf_term = True, False
        if self.var_aquisition_fct.get() == 2:
            ei_term, gf_term = False, False

        self.acquisition = Acquisition(ei=ei_term, gf=gf_term, normalize=self.var_aquisition_fct_normalize,
                                       alpha=param.alpha)

    def create_gpr_model(self):
        '''
        create gpr model and preparation for start:

                - Use all data from GUI
                - set the control window
                - create initial data
                - create acquisition object
                - create gpr Object
                - create metric Object

        '''
        self.set_params()  # If User change the values -> set the parameters for the GPR MODEL
        self.set_control_values()
        self.create_init_data()  # Data is generated
        self.init_acquisition_func()  # Eval is generated

        self.gpr = GaussProcessRegression(param.get_kernel(self.var_kernel_button.get()), param.function, param.dim,
                                          self.data.X_init, self.data.Y_init, param.SSN)
        self.metrics = Metrics()  # Metric is generated

        self.is_running = True  # if stop was pressed, change the value back to True
        self.label_state.config(text="Model is ready")
        return None

    # stop the gpr model
    def stop(self):
        self.is_running = False
        self.label_state.config(text="STOPPED")
        return None

    def start_gpr_threading(self):
        ''' usage of threading -> GUI dont freeze'''
        self.is_running = True
        self.label_state.config(text="GPR startet")
        # create thread -> tkinter window does not freeze
        self.start_gpr_thread = threading.Thread(target=self.start_gpr)
        self.start_gpr_thread.start()

    # start gpr, do step_gpr until finish
    def start_gpr(self):
        self.is_running = True
        while self.step_gpr() == 0 and self.is_running == True:
            continue

    def step_gpr(self):
        """
            Perform a single step of Gaussian Process Regression (GPR).

            Returns:
                int: Return code indicating the status of GPR.

            """
        if self.is_running and self.data.iteration <= param.iterations + 1 and self.data.repetition <= param.repetitions:
            # Check if GPR is finished
            if self.data.iteration == param.iterations + 1 and self.data.repetition == param.repetitions:
                # Save final scores and average scores
                self.data.save_scores__()

                r2__average, rsme__average = self.metrics.calculate_average_scores(self.data.r2_score__,
                                                                                   self.data.rsme_score__)
                self.data.save_average_scores__(r2__average, rsme__average)

                self.data.save_data__()
                self.label_state.config(text="GPR FINISHED")

                # used to plot finished result - can be deleted
                if param.dim == 2:
                    plt.figure()
                    plt.imshow(self.data.fstar_[-1].reshape(self.gpr.grids[0].shape), origin='lower',
                               extent=[param.SSN[0],
                                       param.SSN[1],
                                       param.SSN[0],
                                       param.SSN[1]],
                               cmap='coolwarm')
                    plt.scatter(self.data.X_obs[:, 0], self.data.X_obs[:, 1], c=self.data.Y_obs, cmap='coolwarm',
                                edgecolors='black',
                                linewidths=1)

                return 1

            elif self.data.iteration == param.iterations + 1 and self.data.repetition <= param.repetitions:
                # Move to the next repetition
                self.data.iteration = 0
                self.data.repetition += 1

                # Save data of repetition and prepare for next repetition
                self.data.save_data__()
                self.data.save_scores__()
                self.prepare_data_for_new_repetition()

            print("Start: Iteration {}/".format(self.data.iteration), param.iterations,
                  "Repetition {}/".format(self.data.repetition), param.repetitions)

            self.label_iter_nr_value.config(text="{}/{}".format(self.data.iteration, param.iterations))
            self.label_rep_nr_value.config(text="{}/{}".format(self.data.repetition, param.repetitions))

            start_time = time.time()  # Start timer for GPR

            fstar_i, Sstar_i = self.gpr.gp()  # Perform GPR

            # Create Test Data
            if self.data.iteration == 0 and self.data.repetition == 1:
                global indices_of_test_data  # global so it can used any iteration / repetition
                indices_of_test_data = self.data.create_test_data(test_data_size=param.test_data_size, dim=param.dim,
                                                                  function=param.function, Xstar=self.gpr.Xstar)

            # Calculate aquisition Function for iteration
            aquisition_func = self.acquisition.get_acquisition_function(self.gpr.Xstar, fstar_i,
                                                                        x_obs=self.gpr.X_observed,
                                                                        y_obs=self.gpr.Y_observed, Sstar=Sstar_i,
                                                                        shape_like=self.gpr.grids[
                                                                            0].shape)

            # Calculate x_new and the function value of it
            x_new = self.acquisition.get_new_x(aquisition_func, self.gpr.Xstar)
            if x_new is None:
                x_new = np.array(np.random.uniform(param.SSN[0], param.SSN[1], (1, param.dim)))

            y_new = self.get_y_new(x_new)

            # y_pred = self.gpr.get_predicted_value_for_x_test_indices(indices_of_test_data)
            y_pred = fstar_i.flatten()[indices_of_test_data]  # prediction

            # Calculate RSME and R2 score
            rsme_i, r2_i = self.metrics.get_rsme_r2_scores(y_pred, self.data.Y_test)

            # Calculate the computational time and set it for control window
            if True:
                iteration_time = time.time() - start_time  # calculate the time needed for iteration
                self.set_control_time(iteration_time)

            # Plots for each iteration -- can be deleted --
            def plot_2dim_fstar():
                plt.figure()
                plt.imshow(fstar_i.reshape(self.gpr.grids[0].shape), origin='lower', extent=[param.SSN[0],
                                                                                             param.SSN[1],
                                                                                             param.SSN[0],
                                                                                             param.SSN[1]],
                           cmap='coolwarm')
                plt.scatter(self.data.X_obs[:, 0], self.data.X_obs[:, 1], c=self.data.Y_obs, cmap='coolwarm',
                            edgecolors='black',
                            linewidths=1)
                plt.title("Repetition: {} Iterations {}".format(self.data.repetition, self.data.iteration + 1))

            def plt_1dim_fstar():
                plt.figure()
                x = self.gpr.grids[0]
                plt.plot(x, param.function(self.gpr.grids[0]), linestyle="--", c="black", linewidth=1)

                plt.scatter(self.data.X_init, self.data.Y_init, c="red", s=4)
                plt.scatter(self.data.X_obs[param.init_data_size:], self.data.Y_obs[param.init_data_size:], c="black",
                            s=4)

                plt.plot(x, fstar_i, c="blue", linewidth="1")
                # plt.errorbar(x, self.Data.fstar_[-1], yerr=2 * np.diag(self.Data.Sstar_[-1]))

                for i, txt in enumerate(range(len(self.data.X_obs))):
                    if i >= param.init_data_size:
                        plt.annotate((txt + 1) - param.init_data_size, (self.data.X_obs[i], self.data.Y_obs[i]),
                                     xytext=(self.data.X_obs[i] + 0.05, self.data.Y_obs[i] + 0.05))

            if param.dim == 1:
                plt_1dim_fstar()
            if param.dim == 2:
                plot_2dim_fstar()

            # update gpr data
            self.gpr.update_data(x_new, y_new)
            # save results, scores of iteration, and add the new points to the observed
            self.data.save_data_(fstar_i, Sstar_i, x_new, y_new)
            self.data.save_scores_(rsme_i, r2_i)

            self.data.iteration += 1
            return 0

        return 1

    # show Gui
    def show(self):
        self.root.mainloop()
