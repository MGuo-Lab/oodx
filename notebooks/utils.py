import numpy as np
import matplotlib.pyplot as plt


def peaks(x):
    term1 = 3 * (1 - x[:, 0]) ** 2 * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)
    term2 = - 10 * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5) * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
    term3 = - 1 / 3 * np.exp(-(x[:, 0] + 1) ** 2 - x[:, 1] ** 2)
    y = sum([term1, term2, term3])
    return y.reshape(-1, 1)


def func_1d(x):
    y = 0.5 * ( - 0.2 * x + 0.5 * x * np.cos(x) - 0.3 * x * np.sin(x) )
    return y


def plot_sampling(api):

    fig = plt.figure(figsize=(4,3))
    
    ax = fig.add_subplot(111)
    ax.scatter(api.x[:, 0], api.x[:, 1], c='r', s=5)


def plot_peaks():
    space = [(-3.0, 3.0), (-3.0, 3.0)]

    # create figure and axes
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)

    # get plotting data
    x1, x2 = np.linspace(*space[0], 50), np.linspace(*space[1], 50)
    x_plot = np.c_[x1, x2]
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    inputs = np.c_[x1_grid.ravel(), x2_grid.ravel()]

    y = peaks(inputs)

    ax1.contourf(x_plot[:, 0], x_plot[:, 1], y.reshape((50, 50)))
    ax1.title.set_text('underlying model')
    plt.show()


def plot_1d(api, show_underlying=False, show_samples=False):
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)

    if show_underlying:
        x = np.linspace(*api.space[0], 100)
        y = func_1d(x)
        ax1.plot(x, y, ls='--')

    if show_samples:
        ax1.scatter(api.x, api.y)
    
    if api.regressor is not None:
        x_ = np.linspace(*api.space_[0], 100).reshape(-1, 1)
        # pred, std = api.regressor.predict(x_, return_std=True)
        pred, std = api.regressor.formulation(x_, return_std=True)
        pred =  pred * api.y_std + api.y_mean
        std = std * api.y_std
        u = 1.96 * std
        ax1.plot(x, pred)
        ax1.fill_between(x.ravel(), pred.ravel() + u.ravel(), pred.ravel() - u.ravel(), alpha=0.2)

    plt.show()


def plot_1d_old(api):
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    x_new = np.linspace(-3, 3, 100).reshape(-1, 1)
    pred, u = api.regressor.predict(x_new, return_std=True)

    ei = api._modified_ei(x_new)

    ax1.plot(x_new, pred, c='#3274B5', alpha=0.5)
    ax1.fill_between(x_new.ravel(), pred.ravel() + u.ravel(), pred.ravel() - u.ravel(), alpha=0.2)
    ax1.scatter(api.x_, api.y_, c='#3274B5', s=5)
    ax1.title.set_text('predictions')

    ax2.plot(x_new, ei, c='#3274B5', alpha=0.5)
    ax2.title.set_text('ei')

    plt.show()


def plot_model(api, show_samples=False, show_new=False, show_triangles=False, show_sol=False, show_class=False):
    # create figure and axes
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    # ax.title.set_text('Dealaunay triangulation')

    # get plotting data
    x1_scaled, x2_scaled = np.linspace(*api.space_[0], 50), np.linspace(*api.space_[1], 50)
    x1_scaled_grid, x2_scaled_grid = np.meshgrid(x1_scaled, x2_scaled)
    model_input_grid = np.c_[x1_scaled_grid.ravel(), x2_scaled_grid.ravel()]
    model_grid = api.regressor.predict(model_input_grid)
    # rescale
    if api.x_train is not None:
        x_plot = np.c_[x1_scaled, x2_scaled] * api.x_train_std + api.x_train_mean
        pred_plot = model_grid * api.y_train_std + api.y_train_mean
    else:
        x_plot = np.c_[x1_scaled, x2_scaled] * api.x_std + api.x_mean
        pred_plot = model_grid * api.y_std + api.y_mean

    ax.contourf(x_plot[:, 0], x_plot[:, 1], pred_plot.reshape((50, 50)))

    if show_samples:
        if api.x_train is not None:
            ax.scatter(api.x_train[:, 0], api.x_train[:, 1], c=api.t_train, cmap='bwr_r', s=5)
        else:
            ax.scatter(api.x[:, 0], api.x[:, 1], c=api.t, cmap='bwr_r', s=5)
    
    if show_new != False:
        ax.scatter(show_new[:, 0], show_new[:, 1], c='b', s=5)
    
    if show_triangles:
        ax.triplot(api.delaunay.points[:, 0], api.delaunay.points[:, 1], api.delaunay.simplices, c='k', lw=0.2)
    
    if show_sol:
        ax.plot(show_sol[0], show_sol[1], 'w*')
    
    if show_class:
        class_grid = api.classifier.predict(model_input_grid)
        class_grid = class_grid.reshape((50, 50))
        ax.contour(x_plot[:, 0], x_plot[:, 1], class_grid, levels=[0.5], linestyles='dashed', colors='k', linewidths=1)
    
    plt.show()


def plot_gp(api, show_samples=False, show_new=False, show_triangles=False, show_sol=False, show_class=False):
    
    # create figure and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # get plotting data
    x1_scaled, x2_scaled = np.linspace(*api.space_[0], 50), np.linspace(*api.space_[1], 50)
    x1_scaled_grid, x2_scaled_grid = np.meshgrid(x1_scaled, x2_scaled)
    model_input_grid = np.c_[x1_scaled_grid.ravel(), x2_scaled_grid.ravel()]
    pred_grid, u_grid = api.regressor.predict(model_input_grid, return_std=True)
    
    # rescale
    if api.x_train is not None:
        x_plot = np.c_[x1_scaled, x2_scaled] * api.x_train_std + api.x_train_mean
        pred_plot = pred_grid * api.y_train_std + api.y_train_mean
        u_plot = u_grid * api.y_train_std
    else:
        x_plot = np.c_[x1_scaled, x2_scaled] * api.x_std + api.x_mean
        pred_plot = pred_grid * api.y_std + api.y_mean
        u_plot = u_grid * api.y_std

    ax1.contourf(x_plot[:, 0], x_plot[:, 1], pred_plot.reshape((50, 50)))
    ax2.contourf(x_plot[:, 0], x_plot[:, 1], u_plot.reshape((50, 50)))

    if show_samples:
        if api.x_train is not None:
            ax1.scatter(api.x_train[:, 0], api.x_train[:, 1], c=api.t_train, cmap='bwr_r', s=5)
            ax2.scatter(api.x_train[:, 0], api.x_train[:, 1], c=api.t_train, cmap='bwr_r', s=5)
        else:
            ax1.scatter(api.x[:, 0], api.x[:, 1], c=api.t, cmap='bwr_r', s=5)
            ax2.scatter(api.x[:, 0], api.x[:, 1], c=api.t, cmap='bwr_r', s=5)
    
    plt.show()


def plot_activation(activation):
        
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)

    x = np.linspace(-5, 5, 100)

    if activation == 'tanh':
        y = np.tanh(x)
        ax.plot(x, y)
        ax.title.set_text('tanh')
    
    if activation == 'sigmoid':
        f = lambda x: 1 / (1 + np.exp(-x))
        ax.plot(x, f(x))
        ax.title.set_text('sigmoid')
    
    if activation == 'softplus':
        f = lambda x: np.log(1 + np.exp(x))
        ax.plot(x, f(x))
        ax.title.set_text('softplus')
    
    if activation == 'relu':
        f = lambda x: np.maximum(0, x)
        ax.plot(x, f(x))
        ax.title.set_text('relu')
    
    if activation == 'hardsigmoid':
        f = lambda x: np.maximum(0, np.minimum(x/6 + 0.5, 1))
        ax.plot(x, f(x))
        ax.title.set_text('hardsigmoid')
    
    plt.show()


def plot_triangles(api, new):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)

    ax.triplot(api.delaunay.points[:, 0], api.delaunay.points[:, 1], api.delaunay.simplices, c='k', lw=0.2)
    ax.scatter(api.x[:, 0], api.x[:, 1], c=api.y, s=5)
    ax.scatter(new[:, 0], new[:, 1], c='r', s=5)
    plt.show()
