import datetime
import time
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import ast

from matplotlib import pyplot as plt

from as_class import *
from functions import *
from eval import *

import _params as param
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import sympy as sp


# TODO: AT THE END CUZ OF INDIVIDUAL FUNCTION -> GPR MATRIX need change to
#  function(*X.T)

class Gui():
    """
        GUI class.

        Implements a GUI to visualize the results
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GUI for GPR")

        self.Data = None
        self.Eval = None
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

        # DIFFERENT TABS in FRAME BOTTOM LEFT (Frame 3)

        # Function for SUB-Frame
        def create_sub_frame(parent, frame_text, fill="x", expand="yes", pad_y=2, pad_x=5):
            frame = tk.LabelFrame(parent, text=frame_text)
            frame.pack(fill=fill, expand=expand, pady=pad_y, padx=pad_x)
            return frame

        def create_sub_frame_grid(parent, frame_text, row_, column_, sticky_="nw", pad_y=2, pad_x=5):
            frame = tk.LabelFrame(parent, text=frame_text)
            frame.grid(row=row_, column=column_, pady=pad_y, padx=pad_x, sticky=sticky_)
            return frame

        # parameter slots
        def create_param_entry(parent, label_text, parameter, row, column=0, pad_x=5, pad_y=5, width_label=20,
                               width_entry=10, big_entry=False):

            label = tk.Label(parent, text=label_text, width=width_label)
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

            self.frame_checkbutton = create_sub_frame(self.frame3, "Aquisition Function")

            #  CREATE PLOT FRAME in FRAME 2
            self.subframe_plot = create_sub_frame_grid(self.frame2, "Plot", row_=0, column_=0)
            #  CREATE SUB FRAME in FRAME 3
            # self.frame_example = create_sub_frame(self.frame4, "Examples")
            # self.frame_control = create_sub_frame(self.frame4, "Control")

            self.frame_4top = tk.Frame(self.frame4)
            self.frame_4top.grid(row=0, column=0, padx=5, sticky="nsew")

            self.frame_4down = tk.Frame(self.frame4)
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
                                                                      1)

            if True:
                def get_function():
                    func_str = self.param_individual_function.get()
                    symbols_str = self.param_symbol.get()

                    if ',' in symbols_str or ' ' in symbols_str:
                        symbols = tuple(sp.symbols(symbols_str))
                    else:
                        symbols = (sp.symbols(symbols_str),)

                    func = sp.lambdify(symbols, func_str, 'numpy')
                    self.function_str.config(text=str(sp.sympify(func_str)))

                    param.function = np.vectorize(func)

                self.label_dim, self.param_dim = create_param_entry(self.frame_function, "Dimension",
                                                                    param.dim, 2)

                self.func_button = tk.Button(self.frame_individual_function, text="Convert to pyhton function",
                                             command=get_function)
                self.func_button.grid(row=0, column=2, padx=5, pady=5)

                tk.Label(self.frame_individual_function, text="Function:").grid(row=0, column=3, padx=5, pady=5)
                self.function_str = tk.Label(self.frame_individual_function, text="type function and convert")
                self.function_str.grid(row=0, column=4, padx=5, pady=5, rowspan=2, columnspan=3, sticky="nw")

                self.var_function_individual = tk.BooleanVar(value=False)

                # Create a blank row with Separator widget
                sep = ttk.Separator(self.frame_example, orient="horizontal")
                sep.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

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

        #  Create ITEMS of Control Window
        # - Repetition / Iteration Number
        # - step and start button
        # - time of last iteration
        if True:
            self.label_rep_nr = tk.Label(self.frame_control, text="Repetition number", width=20)
            self.label_rep_nr.grid(row=0, column=0, padx=5)
            self.label_rep_nr_value = tk.Label(self.frame_control, text="0", width=10)  # TODO: "0 / {}".format(5)
            self.label_rep_nr_value.grid(row=0, column=1, padx=5)

            self.label_iter_nr = tk.Label(self.frame_control, text="Iteration number", width=20)
            self.label_iter_nr.grid(row=1, column=0, padx=5)
            self.label_iter_nr_value = tk.Label(self.frame_control, text="0", width=10)  # TODO: "0 / {}".format(30)
            self.label_iter_nr_value.grid(row=1, column=1, padx=5)

            self.label_iter_time = tk.Label(self.frame_control, text="time per iteration (h:m:s.s)", width=20)
            self.label_iter_time.grid(row=2, column=0, padx=5)
            self.iter_time_value = tk.Label(self.frame_control, text="0:00:0.000", width=10)
            self.iter_time_value.grid(row=2, column=1, padx=5)

            self.btn_gpr_step = tk.Button(self.frame_control, text='step', width=15, command=self.step_gpr)
            self.btn_gpr_step.grid(row=4, column=1, padx=5)

            self.btn_gpr_start = tk.Button(self.frame_control, text='start GPR', width=15,
                                           command=self.start_gpr)
            self.btn_gpr_start.grid(row=4, column=2, sticky="W", padx=2, pady=0)

        ########################## CREATES THE PLOT AT START OF THE GUI ###############################################

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

        create_plot()
        create_plot_()

        self.frame_plt_btn = tk.Frame(self.frame2)
        self.frame_plt_btn.grid(row=1, column=0, padx=5, sticky="nw")

        self.btn_plt = tk.Button(self.frame_plt_btn, text="refresh r2 plot", command=self.plot)
        self.btn_plt.grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.btn_plt = tk.Button(self.frame_plt_btn, text="add plot r2-score", command=self.add_plot_r2)
        self.btn_plt.grid(row=1, column=0, padx=2, pady=2, sticky="w")

        padding_cell = tk.Label(self.frame_plt_btn, width=30)
        padding_cell.grid(row=0, column=2)

        self.btn_plt = tk.Button(self.frame_plt_btn, text="refresh rsme plot", command=self.plot_)
        self.btn_plt.grid(row=0, column=3, padx=2, pady=2, sticky="w")

        self.btn_plt = tk.Button(self.frame_plt_btn, text="add plot rsme-score", command=self.add_plot_rsme)
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
                self.gui_plt.legend(lines, [line.get_label() for line in lines])
                self.canvas_figure.draw()

            elif var == self.show_average_rsme_score:
                if var.get():
                    self.average_rsme_line.set_visible(True)
                else:
                    self.average_rsme_line.set_visible(False)
                # get the visible lines
                lines = [line for line in self.gui_plt_.lines if line.get_visible()]
                # update the legend
                self.gui_plt_.legend(lines, [line.get_label() for line in lines])
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
                self.gui_plt.legend(lines, [line.get_label() for line in lines])
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
                self.gui_plt_.legend(lines, [line.get_label() for line in lines])
                self.canvas_figure_.draw()



            else:
                return None

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

        # for average score
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

        # Buttons For Use of Aquisition Function
        if True:
            font_txt = tkFont.Font(size=15)

            self.var_aquisition_fct = tk.IntVar(value=1)  # standard is expected improvement
            self.var_aquisition_fct_normalize = tk.IntVar(value=1)  # standart is to normalize

            tk.Radiobutton(self.frame_checkbutton, text="Expected Improvement",
                           variable=self.var_aquisition_fct,
                           value=0).grid(row=1, column=0, sticky='W')
            tk.Radiobutton(self.frame_checkbutton, text="Expected Improvement for Global Fit",
                           variable=self.var_aquisition_fct,
                           value=1).grid(row=2, column=0, sticky='W')
            tk.Label(self.frame_checkbutton, text="").grid(row=3, column=0)
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
        # Buttons For Use of Example
        if True:
            self.var_example = tk.IntVar(value=0)

            ttk.Label(self.frame_example, text="One Dimensional", font=font_txt).grid(row=0, column=0)
            ttk.Label(self.frame_example, text="Two Dimensional", font=font_txt).grid(row=3, column=0)
            ttk.Label(self.frame_example, text="Four Dimensional", font=font_txt).grid(row=4, column=0)
            ttk.Label(self.frame_example, text="Individual Function", font=font_txt).grid(row=7, column=0)

            button1_example = tk.Radiobutton(self.frame_example, text="Example 1.1",
                                             variable=self.var_example,
                                             value=0)  # , command=toggle_function_individual)
            button1_1_example = tk.Radiobutton(self.frame_example, text="Example 1.2",
                                               variable=self.var_example,
                                               value=1)  # , command=toggle_function_individual)
            button1_2_example = tk.Radiobutton(self.frame_example, text="Example 1.3",
                                               variable=self.var_example,
                                               value=2)  # , command=toggle_function_individual)
            button2_example = tk.Radiobutton(self.frame_example, text="Example 2.1",
                                             variable=self.var_example,
                                             value=3)  # , command=toggle_function_individual)
            button3_example = tk.Radiobutton(self.frame_example, text="Example 4.2",
                                             variable=self.var_example,
                                             value=4)  # , command=toggle_function_individual)
            button_individual = tk.Radiobutton(self.frame_example, text="Type function into entry and convert",
                                               variable=self.var_example,
                                               value=10)  # , command=toggle_function_individual)

            # organize the button position and the text before
            button1_example.grid(row=0, column=1, sticky='W')
            button1_1_example.grid(row=1, column=1, sticky='W')
            button1_2_example.grid(row=2, column=1, sticky='W')

            button2_example.grid(row=3, column=1, sticky='W')
            button3_example.grid(row=4, column=1, sticky='W')
            button_individual.grid(row=7, column=1, sticky='W')

            tk.Label(self.frame_example, text="", width=10).grid(row=0, column=2)
            tk.Label(self.frame_example, text="", width=10).grid(row=1, column=2)

            self.state_frame = tk.LabelFrame(self.frame_example, text="", bg="#c0c0c0")
            self.state_frame.grid(row=3, column=3, sticky="W", padx=2, pady=0)

            self.btn_load_param = tk.Button(self.frame_example, text='Load Parameter', bg="#c0c0c0",
                                            command=self.load_params_for_example)
            self.btn_load_param.grid(row=0, column=3, padx=2, pady=0)

            self.btn_create_gpr_model = tk.Button(self.frame_example, text='Create GPR model', bg="#c0c0c0",
                                                  command=self.create_gpr_model)
            self.btn_create_gpr_model.grid(row=1, column=3, padx=2, pady=0)

            tk.Label(self.frame_example, text="").grid(row=2, column=2)

            self.label_state = tk.Label(self.state_frame, text="", bg="#c0c0c0", width=13)
            self.label_state.grid(row=0, column=0, sticky="W", padx=2, pady=0)

    def plot(self):
        self.gui_plt.cla()
        # self.gui_plt.set_title("")
        self.gui_plt.set_xlabel("Iterations")
        self.gui_plt.set_ylabel("R2-score")
        self.gui_plt.set_ylim(0, 1.05)
        self.gui_plt.axhline(y=1, ls="--", c="grey")

        self.canvas_figure.draw()

    def plot_(self):
        self.gui_plt_.cla()
        # self.gui_plt_.set_title("")
        self.gui_plt_.set_xlabel("Iterations")
        self.gui_plt_.set_ylabel("RSME-Score")
        self.canvas_figure_.draw()

    def add_plot_r2(self):
        # self.Data.r2_score__ || not defined otherwise
        if self.gpr.iter_nr == param.iterations and self.gpr.rep_nr == param.repetitions:
            x = np.arange(self.gpr.iter_nr)
            self.r2_iter_lines = []
            for i in np.arange(np.array(self.Data.r2_score__).shape[0]):
                y = np.array(self.Data.r2_score__[i])
                y[y < 0] = 0
                line, = self.gui_plt.plot(x, y, label='Repetition {}'.format(i + 1))
                self.r2_iter_lines.append(line)

            if not self.show_iterations_r2.get():
                for line in self.r2_iter_lines:
                    line.set_visible(False)

            self.average_r2_line, = self.gui_plt.plot(x, self.Data.r2_score__average, label='average')

            if not self.show_average_r2_score.get():
                self.average_r2_line.set_visible(False)

            self.visible_lines_r2 = [line for line in self.gui_plt.lines if line.get_visible()]

            self.gui_plt.legend(self.visible_lines_r2, [line.get_label() for line in self.visible_lines_r2])

        self.canvas_figure.draw()
        return None

    def add_plot_rsme(self):
        if self.gpr.iter_nr == param.iterations and self.gpr.rep_nr == param.repetitions:
            x = np.arange(self.gpr.iter_nr)

            self.rsme_iter_lines = []
            for i in np.arange(np.array(self.Data.rsme_score__).shape[0]):
                y = np.array(self.Data.rsme_score__[i])
                y[y < 0] = 0
                line, = self.gui_plt_.plot(x, y, label='Repetition {}'.format(i + 1))
                self.rsme_iter_lines.append(line)

            if self.show_iterations_rsme.get() == False:
                for line in self.rsme_iter_lines:
                    line.set_visible(False)

            self.average_rsme_line, = self.gui_plt_.plot(x, self.Data.rsme_score__average, label='average')

            if self.show_average_rsme_score.get() == False:
                self.average_rsme_line.set_visible(False)

            self.visible_lines_rsme = [line for line in self.gui_plt_.lines if line.get_visible()]
            self.gui_plt_.legend(self.visible_lines_rsme, [line.get_label() for line in self.visible_lines_rsme])

        self.canvas_figure_.draw()
        return None

    def load_params_for_example(self):
        def delete_params():
            widgets_to_clear = (self.param_function, self.param_dim, self.param_SSN, self.param_iterations,
                                self.param_repetitions, self.param_init_data_size, self.param_test_data_size,
                                self.param_alpha, self.param_rbf_lengthscale, self.param_rbf_variance,
                                self.param_perio_lengthscale, self.param_perio_periodicity)
            for widget in widgets_to_clear:
                widget.delete(0, tk.END)

        def get_function(): # aus get function and get individual function make same function
            func_str = self.param_individual_function.get()
            symbols_str = self.param_symbol.get()

            if ',' in symbols_str or ' ' in symbols_str:
                symbols = tuple(sp.symbols(symbols_str))
            else:
                symbols = (sp.symbols(symbols_str),)

            func = sp.lambdify(symbols, func_str, 'numpy')
            self.function_str.config(text=str(sp.sympify(func_str)))
            return func, len(symbols)

        if self.var_example.get() == 10:  # INDIVIDUAL FUNCTON GET
            param.function, param.dim = get_function()
            param.example_txt = "individual"
            self.param_function.delete(0, tk.END)
            self.param_function.insert(0, param.example_txt)
            self.param_dim.delete(0, tk.END)
            self.param_dim.insert(0, str(param.dim))
            self.label_state.config(text="Individual Loaded")  # Let User Parameter loaded
        else:
            delete_params()

            param.example_txt, param.kernel, param.function, param.SSN, param.dim, param.iterations, param.repetitions, \
            param.init_data_size, param.test_size, param.alpha = param.get_params(self.var_example.get())

            param.rbf_lengthscale, param.rbf_variance, param.perio_lengthscale, param.perio_periodicity \
                = param.init_kernel_params()

            if True:
                self.param_function.insert(0, param.example_txt)
                self.param_dim.insert(0, str(param.dim))

                self.param_SSN.insert(0, str(param.SSN))

                self.param_alpha.insert(0, str(param.alpha))
                self.param_iterations.insert(0, str(param.iterations))
                self.param_repetitions.insert(0, str(param.repetitions))

                self.param_init_data_size.insert(0, str(param.init_data_size))
                self.param_test_data_size.insert(0, str(param.test_size))

                self.param_rbf_lengthscale.insert(0, str(param.rbf_lengthscale))
                self.param_rbf_variance.insert(0, str(param.rbf_variance))
                self.param_perio_lengthscale.insert(0, str(param.perio_lengthscale))
                self.param_perio_periodicity.insert(0, str(param.perio_periodicity))

        self.label_state.config(text="Parameter loaded")  # Let User Parameter loaded
        return None

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
            return func

        del event

        if self.var_example.get() == 10:  # individual case
            param.function = get_individual_function()
            param.dim = int(self.param_dim.get())
        else:
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

    def reset_control_values(self):
        self.label_iter_nr_value.config(text="{}/{}".format(0, param.iterations))
        self.label_rep_nr_value.config(text="{}/{}".format(0, param.repetitions))
        self.iter_time_value.config(text="0:00:0.000")

    # TODO: GIRD / RANDOM / SELBST ERMÖGLICHEN
    def create_init_data(self):
        start, stop, num = param.SSN
        X_init = np.random.uniform(start, stop, (param.init_data_size, param.dim))
        Y_init = param.function(X_init)
        self.Data = Data(X_init, Y_init)

    def new_init_data(self):
        start, stop, num = param.SSN

        X_init_new = np.random.uniform(start, stop, (param.init_data_size, param.dim))
        Y_init_new = param.function(X_init_new)

        return X_init_new, Y_init_new

    def prepare_data_for_new_repetition(self):
        self.Data.reset_scores()  # reset the score and new datapoints

        X_init_new, Y_init_new = self.new_init_data()  # here create new Initial Data
        self.Data.reset_init_data(X_init_new, Y_init_new)  # reset the Dataset in Data for new repetition
        self.gpr.reset_init_data(X_init_new, Y_init_new)  # reset the Dataset in GPR for new repetition

    # TODO: ermöglichen ei, gf zu wählen
    def create_eval(self):
        self.Eval = Eval(ei=True, gf=True, normalize=True, alpha=param.alpha)

    def create_gpr_model(self):
        self.set_params()  # If User change the values -> set the parameters for the GPR MODEL

        self.reset_control_values()

        self.create_init_data()  # DATA MODEL # Data is generated
        self.create_eval()  # Eval Model # Eval is generated

        self.gpr = GaussianProcessRegressor(param.get_kernel(self.var_kernel_button.get(), ), param.function, param.dim,
                                            self.Data.X_init, param.SSN)

        self.label_state.config(text="Model is ready")
        return None

    def start_gpr(self):
        while self.step_gpr(stepwise=0) == 0:
            continue

    def step_gpr(self, stepwise=0):
        if not stepwise and self.gpr.iter_nr <= param.iterations and self.gpr.rep_nr <= param.repetitions:
            # TODO: HIER LOGIK ÜBERPRÜFEN (LEAST PRIORITY)

            if self.gpr.iter_nr == param.iterations and self.gpr.rep_nr == param.repetitions:

                self.label_iter_nr_value.config(text="{}/{}".format(self.gpr.iter_nr, param.iterations))
                self.label_rep_nr_value.config(text="{}/{}".format(self.gpr.rep_nr, param.repetitions))

                self.Data.save_scores__()  # before finish - save results

                r2__average, rsme__average = self.Eval.calculate_average_scores(self.Data.r2_score__,
                                                                                self.Data.rsme_score__)
                self.Data.save_average_scores(r2__average, rsme__average)

                self.label_state.config(text="GPR FINISHED")
                return 1
            elif self.gpr.iter_nr == param.iterations and self.gpr.rep_nr <= param.repetitions:
                self.gpr.iter_nr = 0
                self.gpr.rep_nr += 1

                self.Data.save_data_from_repetition()  # TODO: MUSS DOCH VERSCHIEDEEN X_INIT SEIN ?!
                self.Data.save_scores__()  # before next repetition - save results
                self.prepare_data_for_new_repetition()

            print("Start: Iteration {}/".format(self.gpr.iter_nr), param.iterations,
                  "Repetition {}/".format(self.gpr.rep_nr), param.repetitions)

            self.label_iter_nr_value.config(text="{}/{}".format(self.gpr.iter_nr + 1, param.iterations))
            self.label_rep_nr_value.config(text="{}/{}".format(self.gpr.rep_nr, param.repetitions))

            start_time = time.time()  # start timer for GPR

            fstar_i, Sstar_i = self.gpr.GPR()  # do GPR

            aquisition_func = self.Eval.get_aquisition_function(self.gpr.Xstar, fstar_i, x_obs=self.gpr.X,
                                                                y_obs=self.gpr.Y, Sstar=Sstar_i,
                                                                shape_like=self.gpr.grid_shape)  # GET Aquisition Function

            x_new = self.Eval.get_new_x_for_eigf(aquisition_func, self.gpr.Xstar)  # Calculate x_new
            # GET SCORES

            iteration_time = time.time() - start_time  # calculate the time needed for iteration
            td = datetime.timedelta(seconds=iteration_time)
            hmsm = str(td).split('.')[0] + '.' + str(td).split('.')[1][:3]
            self.iter_time_value.config(text="{}".format(hmsm))  # configure the time value

            # prepare data for evaluation
            y_pred, y_true = self.Data.data_for_evaluation(Xstar=self.gpr.Xstar, fstar=self.gpr.fstar,
                                                           function=param.function, dim=param.dim,
                                                           test_size=param.test_data_size)
            rsme_i, r2_i = self.Eval.get_rsme_r2_scores(y_pred, y_true)

            # SAVE fstar/Sstar
            self.gpr.update_data(x_new)  # add the new data point to the observed data
            # TODO: MUSS ICH FSTAR UND SSTAR BEI GPR SPEICHERN ??? ODER SPEICHER ICH DAS UNTER DATA
            # SAVE IN GPR
            self.gpr.fstar_.append(fstar_i)
            self.gpr.Sstar_.append(Sstar_i)

            # SAVE IN DATA
            self.Data.update_data_(fstar_i, Sstar_i, x_new, param.function(x_new))
            self.Data.save_scores_(rsme_i, r2_i)

            self.gpr.iter_nr += 1
            return 0

        return 1

    def show(self):
        self.root.mainloop()

    def destroy(self):
        self.root.destroy()


g = Gui()
g.show()
