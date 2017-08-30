set terminal postscript eps enhanced color
set xlabel "x"
set ylabel "y"
set pm3d at bs
set hidden3d
set palette rgbformulae 30, 31, 32
set isosample 50
unset key

set xrange [-6:6]
set yrange [-6:6]
set title font ",30"
set title "Quadratic Function"
set zlabel "f_{Q}(x, y)"
set view ARG1, ARG2
set output "quadratic_function.eps"
splot 1.125*x**2 + 0.5*x*y + 0.75*y**2 + 2*x + 2*y

set xrange [-6:6]
set yrange [-6:6]
set title font ",30"
set title "Ridge Regularized Logistic Regression"
set zlabel "f_{LL}(x, y)"
set view ARG3, ARG4
set output "ridge_regularized_logistic_regression.eps"
splot 0.5*(x**2 + y**2) + 50*log(1 + exp(-0.5*y)) + 50*log(1 + exp(0.2*x))

set xrange [-6:6]
set yrange [-6:6]
set title font ",30"
set title "Himmelblau's Function"
set zlabel "f_{H}(x, y)"
set view ARG5, ARG6
set output "himmelblaus_function.eps"
splot 0.1*(x**2 + y - 11)**2 + 0.1*(x + y**2 - 7)**2

set xrange [-3:3]
set yrange [-6:6]
set title font ",30"
set title "Rosenbrock's Banana Function"
set zlabel "f_{R}(x, y)"
set view ARG7, ARG8
set output "rosenbrock_banana_function.eps"
splot 0.002*(1 - x)**2 + 0.2*(y - x**2)**2
