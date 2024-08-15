# Set the output file name
set output 'plot.png'

# Set pixel density
set terminal png size 2048,1856

# Set the title of the plot
set title 'BVH Time scaling'

# Set the labels for the x and y axes
set xlabel 'Number of primitives'
set ylabel 'Time (ms)'

set datafile separator ","

# loglog scale
set logscale x
set logscale y

# Use column title
set key autotitle columnhead

set style line 1 lw 5 
set style line 2 lw 5
set style line 3 lw 5

# Plot the data
plot 'data.csv' using 1:2 with linespoints linestyle 1, \
'data.csv' using 1:3 with linespoints linestyle 2, \
'data.csv' using 1:4 with linespoints linestyle 3