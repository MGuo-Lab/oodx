import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


class BlackBox:
    def __init__(self):
        pass
    
    def sample_y(self, x):
        return peaks(x)
    
    def sample_t(self, x):
        return feas(x)


def peaks(x):
    term1 = 3 * (1 - x[:, 0]) ** 2 * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)
    term2 = - 10 * (x[:, 0] / 5 - x[:, 0] ** 3 - x[:, 1] ** 5) * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2)
    term3 = - 1 / 3 * np.exp(-(x[:, 0] + 1) ** 2 - x[:, 1] ** 2)
    y = sum([term1, term2, term3])
    return y.reshape(-1, 1)


def feas(x):
    t = np.ones(len(x))
    for i in range(x.shape[0]):
        if ( x[i, 0] ** 2 + x[i, 1] ** 2 > 4 ):
            t[i] = 0
    return t.reshape(-1, 1)


def plot_underlying():
    space = [(-3.0, 3.0), (-3.0, 3.0)]

    # create figure and axes
    fig = plt.figure(figsize=(4,3))
    ax1 = fig.add_subplot(111)

    # get plotting data
    x1, x2 = np.linspace(*space[0], 50), np.linspace(*space[1], 50)
    x_plot = np.c_[x1, x2]
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    inputs = np.c_[x1_grid.ravel(), x2_grid.ravel()]

    y = peaks(inputs)
    circle = plt.Circle((0, 0), 2, color='k', linestyle='--', fill=False, linewidth=1)

    c = ax1.contourf(x_plot[:, 0], x_plot[:, 1], y.reshape((50, 50)), cmap='Greys', alpha=0.8)
    ax1.add_patch(circle)
    fig.colorbar(c, ax=ax1)

    ax1.set_xlim(space[0])
    ax1.set_ylim(space[1])
    ax1.set_xticks(range(-3, 4))
    ax1.set_yticks(range(-3, 4))
    ax1.set_xlabel(r'$x_0$')
    ax1.set_ylabel(r'$x_1$')

    plt.tight_layout()


def plot_samples(db, db2):
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True)

    axs[0].scatter(db.data[db.data.t==1]['x0'], db.data[db.data.t==1]['x1'], c='k', s=10)
    axs[0].scatter(db.data[db.data.t==0]['x0'], db.data[db.data.t==0]['x1'], c='w', edgecolors='k', s=10)
    axs[1].scatter(db2.data[db2.data.t==1]['x0'], db2.data[db2.data.t==1]['x1'], c='k', s=10)
    axs[1].scatter(db2.data[db2.data.t==0]['x0'], db2.data[db2.data.t==0]['x1'], c='w', edgecolors='k', s=10)

    for ax in axs:
        ax.set_xlim(db.space[0])
        ax.set_ylim(db.space[1])
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(range(-3, 4))
        ax.set_xlabel(r'$x_0$')

    axs[0].set_ylabel(r'$x_1$')

    plt.tight_layout()


def plot_gpr(db, db2, gpr, gpr2):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]

    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.5), sharex=True, sharey=True)
    for i, row in enumerate(axs):
        if i==0:
            pred, std = gpr.predict(db.scale_inputs(x_new), return_std=True)
            pred, std = db.inv_scale_outputs(pred), std * db.data.y[db.data.t==1].std(ddof=0) * 1.96
            pred, std = pred.reshape(x1grid.shape), std.reshape(x1grid.shape)
        if i==1:
            pred, std = gpr2.predict(db2.scale_inputs(x_new), return_std=True)
            pred, std = db2.inv_scale_outputs(pred), std * db2.data.y[db2.data.t==1].std(ddof=0) * 1.96
            pred, std = pred.reshape(x1grid.shape), std.reshape(x1grid.shape)
        for j, ax in enumerate(row):
            if j==0:
                c = ax.contourf(x1, x2, pred, alpha=0.8, cmap='Greys')
            if j==1:
                c = ax.contourf(x1, x2, std, alpha=0.8, cmap='Greys')
            fig.colorbar(c, ax=ax)
            ax.set_xlim(db.space[0])
            ax.set_ylim(db.space[1])
            ax.set_xticks(range(-3, 4))
            ax.set_yticks(range(-3, 4))
            if i==0:
                ax.scatter(db.data.x0[db.data.t==1], db.data.x1[db.data.t==1], c='k', s=10)
                ax.scatter(db.data.x0[db.data.t==0], db.data.x1[db.data.t==0], c='w', edgecolors='k', s=10)
            if i==1:
                ax.scatter(db2.data.x0[db2.data.t==1], db2.data.x1[db2.data.t==1], c='k', s=10)
                ax.scatter(db2.data.x0[db2.data.t==0], db2.data.x1[db2.data.t==0], c='w', edgecolors='k', s=10)

    axs[0][0].set_ylabel(r'$x_1$')
    axs[1][0].set_xlabel(r'$x_0$')
    axs[1][0].set_ylabel(r'$x_1$')
    axs[1][1].set_xlabel(r'$x_0$')

    plt.tight_layout()


def plot_gpc(db, db2, gpc, gpc2):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]
    
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.5), sharex=True, sharey=True)
    for i, row in enumerate(axs):
        if i==0:
            pred, std = gpc.predict(db.scale_inputs(x_new), return_std=True)
            pred, std = pred.reshape(x1grid.shape), std.reshape(x1grid.shape)
        if i==1:
            pred, std = gpc2.predict(db2.scale_inputs(x_new), return_std=True)
            pred, std = pred.reshape(x1grid.shape), std.reshape(x1grid.shape)
        for j, ax in enumerate(row):
            if j==0:
                c = ax.contourf(x1, x2, pred, alpha=0.8, cmap='Greys')
            if j==1:
                c = ax.contourf(x1, x2, std, alpha=0.8, cmap='Greys')
            l = ax.contour(x1, x2, pred, levels=[0.5], linestyles='dashed', colors='k', linewidths=1)
            ax.clabel(l, l.levels, inline=True, fontsize=5)
            fig.colorbar(c, ax=ax)
            ax.set_xlim(db.space[0])
            ax.set_ylim(db.space[1])
            ax.set_xticks(range(-3, 4))
            ax.set_yticks(range(-3, 4))
            if i==0:
                ax.scatter(db.data.x0[db.data.t==1], db.data.x1[db.data.t==1], c='k', s=10)
                ax.scatter(db.data.x0[db.data.t==0], db.data.x1[db.data.t==0], c='w', edgecolors='k', s=10)
            if i==1:
                ax.scatter(db2.data.x0[db2.data.t==1], db2.data.x1[db2.data.t==1], c='k', s=10)
                ax.scatter(db2.data.x0[db2.data.t==0], db2.data.x1[db2.data.t==0], c='w', edgecolors='k', s=10)

    axs[0][0].set_ylabel(r'$x_1$')
    axs[1][0].set_xlabel(r'$x_0$')
    axs[1][0].set_ylabel(r'$x_1$')
    axs[1][1].set_xlabel(r'$x_0$')

    plt.tight_layout()


def plot_gp_opt(db, db2, gpr, gpr2, gpc, gpc2, sol, sol2):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]
    
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.5), sharex=True, sharey=True)
    for i, row in enumerate(axs):
        if i==0:
            pred, std = gpr.predict(db.scale_inputs(x_new), return_std=True)
            pred, std = db.inv_scale_outputs(pred), std * db.data.y[db.data.t==1].std(ddof=0) * 1.96
            pred, std = pred.reshape(x1grid.shape), std.reshape(x1grid.shape)
            prob = gpc.predict(db.scale_inputs(x_new)).reshape(x1grid.shape)
        if i==1:
            pred, std = gpr2.predict(db2.scale_inputs(x_new), return_std=True)
            pred, std = db2.inv_scale_outputs(pred), std * db2.data.y[db2.data.t==1].std(ddof=0) * 1.96
            pred, std = pred.reshape(x1grid.shape), std.reshape(x1grid.shape)
            prob = gpc2.predict(db.scale_inputs(x_new)).reshape(x1grid.shape)
        for j, ax in enumerate(row):
            if j==0:
                c = ax.contourf(x1, x2, pred, alpha=0.8, cmap='Greys')
            if j==1:
                c = ax.contourf(x1, x2, std, alpha=0.8, cmap='Greys')
            l = ax.contour(x1, x2, prob, levels=[0.25, 0.5, 0.75], linestyles='dashed', colors='k', linewidths=1)
            ax.clabel(l, l.levels, inline=True, fontsize=5)
            fig.colorbar(c, ax=ax)
            ax.set_xlim(db.space[0])
            ax.set_ylim(db.space[1])
            ax.set_xticks(range(-3, 4))
            ax.set_yticks(range(-3, 4))
            if i==0:
                ax.scatter(*sol, marker='*', c='w', edgecolors='k', zorder=10)
            if i==1:
                ax.scatter(*sol2, marker='*', c='w', edgecolors='k', zorder=10)

    axs[0][0].set_ylabel(r'$x_1$')
    axs[1][0].set_xlabel(r'$x_0$')
    axs[1][0].set_ylabel(r'$x_1$')
    axs[1][1].set_xlabel(r'$x_0$')

    plt.tight_layout()


def plot_gp_adaptive_opt(db, gpr, gpc, sol, asol, space):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]
    
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), sharex=True, sharey=True)
    for i, ax in enumerate(axs):
        pred, std = gpr.predict(db.scale_inputs(x_new), return_std=True)
        pred, std = db.inv_scale_outputs(pred), std * db.data.y[db.data.t==1].std(ddof=0) * 1.96
        pred, std = pred.reshape(x1grid.shape), std.reshape(x1grid.shape)
        prob = gpc.predict(db.scale_inputs(x_new)).reshape(x1grid.shape)
        if i==0:
            c = ax.contourf(x1, x2, pred, alpha=0.8, cmap='Greys')
        if i==1:
            c = ax.contourf(x1, x2, std, alpha=0.8, cmap='Greys')
        l = ax.contour(x1, x2, prob, levels=[0.25, 0.5, 0.75], linestyles='dashed', colors='k', linewidths=1)
        ax.clabel(l, l.levels, inline=True, fontsize=5)
        fig.colorbar(c, ax=ax)
        
        corner = (space[0][0], space[1][0])
        width = space[0][1] - space[0][0]
        height = space[1][1] - space[1][0]
        box = patches.Rectangle(corner, width, height, linewidth=0.5, edgecolor='k', facecolor='none')
        ax.add_patch(box)
        
        ax.scatter(*sol, marker='*', c='w', edgecolors='k', zorder=10)
        ax.scatter(*asol, marker='o', c='w', edgecolors='k', zorder=10, s=10)
        
        ax.set_xlim(db.space[0])
        ax.set_ylim(db.space[1])
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(range(-3, 4))

    axs[0].set_ylabel(r'$x_1$')
    axs[0].set_xlabel(r'$x_0$')
    axs[1].set_xlabel(r'$x_0$')

    plt.tight_layout()


def plot_nnr(db, db2, nnr, nnr2):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]

    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), sharex=True, sharey=True)
    for i, ax in enumerate(axs):
        if i==0:
            pred = nnr.predict(db.scale_inputs(x_new))
            pred = db.inv_scale_outputs(pred)
            pred = pred.reshape(x1grid.shape)
        if i==1:
            pred = nnr2.predict(db2.scale_inputs(x_new))
            pred = db2.inv_scale_outputs(pred)
            pred = pred.reshape(x1grid.shape)
        c = ax.contourf(x1, x2, pred, alpha=0.8, cmap='Greys')
        fig.colorbar(c, ax=ax)
        ax.set_xlim(db.space[0])
        ax.set_ylim(db.space[1])
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(range(-3, 4))
        if i==0:
            ax.scatter(db.data.x0[db.data.t==1], db.data.x1[db.data.t==1], c='k', s=10)
            ax.scatter(db.data.x0[db.data.t==0], db.data.x1[db.data.t==0], c='w', edgecolors='k', s=10)
        if i==1:
            ax.scatter(db2.data.x0[db2.data.t==1], db2.data.x1[db2.data.t==1], c='k', s=10)
            ax.scatter(db2.data.x0[db2.data.t==0], db2.data.x1[db2.data.t==0], c='w', edgecolors='k', s=10)

    axs[0].set_ylabel(r'$x_1$')
    axs[0].set_xlabel(r'$x_0$')
    axs[1].set_xlabel(r'$x_0$')

    plt.tight_layout()


def plot_nnc(db, db2, nnc, nnc2):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]
    
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.5), sharex=True, sharey=True)
    for i, row in enumerate(axs):
        if i==0:
            lgt, prb = nnc.predict(db.scale_inputs(x_new), return_proba=True)
            lgt, prb = lgt.reshape(x1grid.shape), prb.reshape(x1grid.shape)
        if i==1:
            lgt, prb = nnc2.predict(db2.scale_inputs(x_new), return_proba=True)
            lgt, prb = lgt.reshape(x1grid.shape), prb.reshape(x1grid.shape)
        for j, ax in enumerate(row):
            if j==0:
                c = ax.contourf(x1, x2, lgt, alpha=0.8, cmap='Greys')
            if j==1:
                c = ax.contourf(x1, x2, prb, alpha=0.8, cmap='Greys')
            l = ax.contour(x1, x2, prb, levels=[0.5], linestyles='dashed', colors='k', linewidths=1)
            ax.clabel(l, l.levels, inline=True, fontsize=5)
            fig.colorbar(c, ax=ax)
            ax.set_xlim(db.space[0])
            ax.set_ylim(db.space[1])
            ax.set_xticks(range(-3, 4))
            ax.set_yticks(range(-3, 4))
            if i==0:
                ax.scatter(db.data.x0[db.data.t==1], db.data.x1[db.data.t==1], c='k', s=10)
                ax.scatter(db.data.x0[db.data.t==0], db.data.x1[db.data.t==0], c='w', edgecolors='k', s=10)
            if i==1:
                ax.scatter(db2.data.x0[db2.data.t==1], db2.data.x1[db2.data.t==1], c='k', s=10)
                ax.scatter(db2.data.x0[db2.data.t==0], db2.data.x1[db2.data.t==0], c='w', edgecolors='k', s=10)

    axs[0][0].set_ylabel(r'$x_1$')
    axs[1][0].set_xlabel(r'$x_0$')
    axs[1][0].set_ylabel(r'$x_1$')
    axs[1][1].set_xlabel(r'$x_0$')

    plt.tight_layout()


def plot_nn_opt(db, db2, nnr, nnr2, nnc, nnc2, sol, sol2):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]
    
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), sharex=True, sharey=True)
    for i, ax in enumerate(axs):
        if i==0:
            pred = nnr.predict(db.scale_inputs(x_new))
            pred = db.inv_scale_outputs(pred)
            pred = pred.reshape(x1grid.shape)
            prob = nnc.predict(db.scale_inputs(x_new), return_proba=True)[1].reshape(x1grid.shape)
        if i==1:
            pred = nnr2.predict(db2.scale_inputs(x_new))
            pred = db2.inv_scale_outputs(pred)
            pred = pred.reshape(x1grid.shape)
            prob = nnc2.predict(db2.scale_inputs(x_new), return_proba=True)[1].reshape(x1grid.shape)
        c = ax.contourf(x1, x2, pred, alpha=0.8, cmap='Greys')
        l = ax.contour(x1, x2, prob, levels=[0.01, 0.5, 0.99], linestyles='dashed', colors='k', linewidths=1)
        ax.clabel(l, l.levels, inline=True, fontsize=5)
        fig.colorbar(c, ax=ax)
        ax.set_xlim(db.space[0])
        ax.set_ylim(db.space[1])
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(range(-3, 4))
        if i==0:
            ax.scatter(*sol, marker='*', c='w', edgecolors='k', zorder=10)
        if i==1:
            ax.scatter(*sol2, marker='*', c='w', edgecolors='k', zorder=10)

    axs[0].set_ylabel(r'$x_1$')
    axs[0].set_xlabel(r'$x_0$')
    axs[1].set_xlabel(r'$x_0$')

    plt.tight_layout()


def plot_nn_adaptive_opt(db, nnr, nnc, sol, asol, space, ads):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]
    
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), sharex=True, sharey=True)
    for i, ax in enumerate(axs):
        pred = nnr.predict(db.scale_inputs(x_new))
        pred = db.inv_scale_outputs(pred)
        pred = pred.reshape(x1grid.shape)
        prob = nnc.predict(db.scale_inputs(x_new), return_proba=True)[1].reshape(x1grid.shape)
        corner = (space[0][0], space[1][0])
        width = space[0][1] - space[0][0]
        height = space[1][1] - space[1][0]
        if i==0:
            c = ax.contourf(x1, x2, pred, alpha=0.8, cmap='Greys')
            l = ax.contour(x1, x2, prob, levels=[0.01, 0.5, 0.99], linestyles='dashed', colors='k', linewidths=1)
            ax.clabel(l, l.levels, inline=True, fontsize=5)
            fig.colorbar(c, ax=ax)
            box = patches.Rectangle(corner, width, height, linewidth=0.5, edgecolor='k', facecolor='none')
            ax.add_patch(box)
            ax.scatter(*sol, marker='*', c='w', edgecolors='k', zorder=10)
        if i==1:
            x_points = db.inv_scale_inputs(ads.delaunay.points)
            ax.triplot(x_points[:, 0], x_points[:, 1], ads.delaunay.simplices, c='k', lw=0.5)
            l = ax.contour(x1, x2, prob, levels=[0.5], linestyles='dashed', colors='k', linewidths=1)
            ax.clabel(l, l.levels, inline=True, fontsize=5)
            ax.scatter(*asol, marker='o', c='w', edgecolors='k', zorder=10, s=10)
        ax.set_xlim(db.space[0])
        ax.set_ylim(db.space[1])
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(range(-3, 4))
    
    axs[0].set_ylabel(r'$x_1$')
    axs[0].set_xlabel(r'$x_0$')
    axs[1].set_xlabel(r'$x_0$')

    plt.tight_layout()


def plot_bayesian_opt(db, gpr, gpc, sol, asol, ystar): 
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]

    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5), sharex=True, sharey=True)
    for i, ax in enumerate(axs):
        if i == 0:
            pred = gpr.predict(db.scale_inputs(x_new))
            pred = db.inv_scale_outputs(pred).reshape(x1grid.shape)
            c = ax.contourf(x1, x2, pred, alpha=0.8, cmap='Greys')
            fig.colorbar(c, ax=ax)
            ax.scatter(*sol, marker='*', c='w', edgecolors='k', zorder=10)
        if i == 1:
            pred, std = gpr.predict(db.scale_inputs(x_new), return_std=1)
            std = std.reshape(-1, 1)
            ei = std / np.sqrt(2 * 3.1416) * np.exp(-(ystar - pred) ** 2 / (2 * std ** 2))
            ei = ei.reshape(x1grid.shape)
            c = ax.contourf(x1, x2, ei, alpha=0.8, cmap='Greys')
            fig.colorbar(c, ax=ax)
            ax.scatter(*asol, marker='o', c='w', edgecolors='k', zorder=10, s=10)
        prob = gpc.predict(db.scale_inputs(x_new)).reshape(x1grid.shape)
        l = ax.contour(x1, x2, prob, levels=[0.5], linestyles='dashed', colors='k', linewidths=1)
        ax.clabel(l, l.levels, inline=True, fontsize=5)
        ax.set_xlim(db.space[0])
        ax.set_ylim(db.space[1])
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(range(-3, 4))
        ax.set_xlabel(r'$x_0$')  
    axs[0].set_ylabel(r'$x_1$')
    plt.tight_layout()


def plot_direct_search(db, clf, sol, asol, ads):
    x1, x2 = np.linspace(*db.space[0], 50), np.linspace(*db.space[1], 50)
    x1grid, x2grid = np.meshgrid(x1, x2)
    x_new = np.c_[x1grid.ravel(), x2grid.ravel()]
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    prob = clf.predict(db.scale_inputs(x_new), return_proba=True)[1].reshape(x1grid.shape)
    x_points = db.inv_scale_inputs(ads.delaunay.points)
    ax.triplot(x_points[:, 0], x_points[:, 1], ads.delaunay.simplices, c='k', lw=0.5)
    l = ax.contour(x1, x2, prob, levels=[0.5], linestyles='dashed', colors='k', linewidths=1)
    ax.clabel(l, l.levels, inline=True, fontsize=5)
    ax.scatter(*sol, marker='*', c='w', edgecolors='k', zorder=10)
    ax.scatter(*asol, marker='o', c='w', edgecolors='k', zorder=10, s=10)
    ax.set_xlim(db.space[0])
    ax.set_ylim(db.space[1])
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(-3, 4))
    ax.set_ylabel(r'$x_1$')
    ax.set_xlabel(r'$x_0$')
    plt.tight_layout()
    