import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


# Define a function to generate a random plot
def random_plot(iteration):
    x = np.linspace(-5, 5, 100)
    y = np.random.normal(size=100)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Random Plot - Iteration {iteration}')
    return fig


# Create a tkinter window
root = tk.Tk()
root.title('Random Plots')

# Create a canvas to hold the plots
canvas = tk.Canvas(root, width=800, height=500)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Create a frame to hold the scrollbar and controls
control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, fill=tk.Y)

# Create a scrollbar and link it to the canvas
scrollbar = tk.Scrollbar(control_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.LEFT, fill=tk.Y, expand=True)
canvas.configure(yscrollcommand=scrollbar.set)


# Create a "Next" button to switch to the next plot
def next_plot():
    global plot_figs, plot_index
    # Remove the current plot from the canvas
    canvas.delete('all')
    # Increment the plot index
    plot_index += 1
    if plot_index >= len(plot_figs):
        plot_index = 0
    # Add the new plot to the canvas
    canvas.create_window((0, 0), window=plot_figs[plot_index], anchor=tk.NW)
    # Update the scroll region of the canvas
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


next_button = tk.Button(control_frame, text='Next', command=next_plot)
next_button.pack(side=tk.TOP)


# Create a "Prev" button to switch to the previous plot
def prev_plot():
    global plot_figs, plot_index
    # Remove the current plot from the canvas
    canvas.delete('all')
    # Decrement the plot index
    plot_index -= 1
    if plot_index < 0:
        plot_index = len(plot_figs) - 1
    # Add the new plot to the canvas
    canvas.create_window((0, 0), window=plot_figs[plot_index], anchor=tk.NW)
    # Update the scroll region of the canvas
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


prev_button = tk.Button(control_frame, text='Prev', command=prev_plot)
prev_button.pack(side=tk.TOP)


# Create a "Reset" button to go back to the initial view
def reset_view():
    canvas.xview_moveto(0)
    canvas.yview_moveto(0)
    canvas.delete('all')
    plot_index = 0
    canvas.create_window((0, 0), window=plot_figs[plot_index], anchor=tk.NW)


reset_button = tk.Button(control_frame, text='Reset View', command=reset_view)
reset_button.pack(side=tk.TOP)

# Create the random plots and add them to the canvas
plot_figs = []
for i in range(5):
    fig = random_plot(i + 1)
    plot_figs.append(FigureCanvasTkAgg(fig, master=canvas).get_tk_widget())
    canvas.create_window((0, i * 500), window=plot_figs[-1], anchor=tk.NW)

# Set the initial plot to the first plot
plot_index = 0
canvas.create_window((0, 0), window=plot_figs[0], anchor=tk.NW)

# Update the scroll region of the canvas
canvas.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))


# Set up the scrollbar to control the canvas scrolling
def on_scroll(*args):
    canvas.yview(*args)


scrollbar.config(command=on_scroll)

# Start the tkinter event loop
root.mainloop()
