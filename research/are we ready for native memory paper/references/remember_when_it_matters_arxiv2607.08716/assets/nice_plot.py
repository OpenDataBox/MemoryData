from matplotlib import pyplot as plt
import torch

# Some sample data
x = torch.linspace(0.0, 5.0, 1025)
y1 = torch.cos(2 * torch.pi * x) * torch.exp(-0.75 * x)
y2 = torch.sin(2 * torch.pi * x) * torch.exp(-0.1 * x)

# Find in your LaTeX document using \the\textwidth
# Divide by the number of plots you would like to show in the same line
TEXT_WIDTH = 472

# Find in your LaTeX document using \expandafter\string\the\font
# or one point smaller
FONT_SIZE = 9

# Color palette should be color-blind friendly, or even better, render a
# nice figure even when printed in black and white
# https://sashamaps.net/docs/resources/20-colors/

# Find the Meta color palette at: https://www.facebook.com/brand/meta/color/
# Try to use Meta blue "#0064E0" as your primary choice,
# then pick distinct colors from the palette
COLORS = ["#0064E0", "#E29400"]

# Enable TeX support
# TODO: cluster
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=FONT_SIZE)

# Uncomment the following line to use Times New Roman for venues such as ICLR
# `preamble` also helpful to load additional LaTeX packages
# plt.rc('text.latex', preamble=r'\usepackage{times}')

# inches = points / 72
plt.figure(figsize=(TEXT_WIDTH / 72. * 0.48, 2.5))

# You possibly do not need a plot title if your caption is good enough
# plt.title("Plot title")

plt.xlabel(r'This is my $x$-axis')
plt.ylabel(r'This is my $y$-axis')
plt.plot(x, y2, lw=1.5, color=COLORS[1], ls="--",
         label=r'$\sin(2 \pi x) e^{-\frac{1}{10} x}$')
plt.plot(x, y1, lw=1.5, color=COLORS[0],
         label=r'$\cos(2 \pi x) e^{-\frac{3}{4} x}$')

plt.grid(linestyle='dotted')
plt.legend() # play with ncol
plt.tight_layout()

# Save in pdf, and specify high dpi for rasterized components (such as images)
# squeeze out every inch of your plot with bbox_inches and pad_inches
plt.savefig("plot_demo.pdf", dpi=400, bbox_inches='tight', pad_inches=0)

# Add your Figure to LaTeX. Use the `subcaption` package for multiple plots
# \begin{figure}[] % try first not to force any particular figure alignment
#   \centering
#   \includegraphics[width=\textwidth]{plot.pdf}
#   \caption{Use generous captions to self-explain the figure.}
#   \label{fig:plot}
# \end{figure}
