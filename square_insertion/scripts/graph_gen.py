import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

x_iterations_array = np.array([0, 1, 2, 3, 4, 5, 6])

# all neg errors
# y_average_est_errors_iterations = [[10, 10, 10, 10, 10, 20], [0.0963977244434755, 2.3642541667019956, 2.5509324332349537, 1.1578301191329956, 0.40706750750541687, 1.204400658607483], [0.5404523280201357, 1.1167475780045621, 1.7231729289167164, 0.024080032482743263, 0.43029874563217163, 1.3897593021392822], [0.10510000258068253, 0.33487433693762947, 2.0235207339399097, 0.09120841324329376, 0.27623996138572693, 1.0207074880599976], [0.8922389414844911, 1.083309372285568, 1.4927413722150562, 0.13363651931285858, 0.27440309524536133, 0.7519044280052185]]

# # all neg errors
# y_average_est_errors_iterations = [[20, 20, 20, 20, 20, 40], [16.58395219236808, 8.722024981193593, 10.176311954714357, 20.945709228515625, 2.868518590927124, 22.587900161743164], [7.146689180435706, 1.573150057606748, 1.0928025095239269, 7.068285942077637, 1.494018793106079, 2.5714409351348877], [3.5000770130773073, 1.3901563246917215, 0.9418914468511952, 1.5544592142105103, 1.0356656312942505, 0.46489962935447693], [0.960501913132239, 0.732854928560206, 0.7012595979943859, 0.37883460521698, 0.40698954463005066, 1.1232916116714478]]

# all neg errors
y_average_est_errors_iterations = [[25, 25, 25, 25, 25, 50], [23.82750204693329, 15.190804085607606, 17.329744411556046, 29.251131057739258, 5.653400421142578, 41.3417854309082], [25.215455805173036, 7.07787898081072, 8.71547253235988, 23.708044052124023, 5.22617769241333, 35.24164962768555], [20.433136736264345, 2.3812887380766092, 3.9062121640127123, 14.43774700164795, 2.737602949142456, 20.981170654296875], [7.308730398526309, 2.8617915819334208, 1.1826237906534254, 2.6150949001312256, 2.0914387702941895, 2.015946626663208], [1.5263049949313379, 1.2251169542717155, 0.6624677906911791, 0.6134580373764038, 1.2823710441589355, 2.5663223266601562], [0.1720697532986426, 1.1895087669538673, 1.4798423518259107, 0.505771815776825, 0.28914874792099, 1.3787587881088257]]


# Smoothen the curves
x_y_spline_0 = make_interp_spline(x_iterations_array, np.array([y[0] for y in y_average_est_errors_iterations]))
x0_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
y0_ = x_y_spline_0(x0_)

x_y_spline_1 = make_interp_spline(x_iterations_array, np.array([y[1] for y in y_average_est_errors_iterations]))
x1_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
y1_ = x_y_spline_1(x1_)

x_y_spline_2 = make_interp_spline(x_iterations_array, np.array([y[2] for y in y_average_est_errors_iterations]))
x2_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
y2_ = x_y_spline_2(x2_)

x_y_spline_3 = make_interp_spline(x_iterations_array, np.array([y[3] for y in y_average_est_errors_iterations]))
x3_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
y3_ = x_y_spline_3(x3_)

x_y_spline_4 = make_interp_spline(x_iterations_array, np.array([y[4] for y in y_average_est_errors_iterations]))
x4_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
y4_ = x_y_spline_4(x4_)

x_y_spline_5 = make_interp_spline(x_iterations_array, np.array([y[5] for y in y_average_est_errors_iterations]))
x5_ = np.linspace(x_iterations_array.min(), x_iterations_array.max(), 500)
y5_ = x_y_spline_5(x5_)

# Plot curves
figure, axis = plt.subplots(2, 1)

# Translation errors curves
axis[0].plot(x0_, y0_, color='r', label='x errors')
axis[0].plot(x1_, y1_, color='g', label='y errors')
axis[0].plot(x2_, y2_, color='b', label='z errors')
axis[0].set_title("Translation errors")
axis[0].set_xlabel("No. of Iterations")
axis[0].set_xticks(np.arange(min(x_iterations_array), max(x_iterations_array)+1, 1.0))
axis[0].set_ylabel("Errors (in mm)")
axis[0].legend(loc='best')

# Orientation errors curves
axis[1].plot(x3_, y3_, color='r', label='roll errors')
axis[1].plot(x4_, y4_, color='g', label='pitch errors')
axis[1].plot(x5_, y5_, color='b', label='yaw errors')
axis[1].set_title("Orientation errors")
axis[1].set_xlabel("No. of Iterations")
axis[1].set_xticks(np.arange(min(x_iterations_array), max(x_iterations_array)+1, 1.0))
axis[1].set_ylabel("Errors (in deg)")
axis[1].legend(loc='best')

plt.show()