import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import re
import shutil
from pod_neural_network_rom import PODNeuralNetworkROM

def plot_surface_pressure(points, parameters, solutions, solution_names, path):
    points = points.reshape(solutions.shape[0], -1)
    data = np.concatenate((points, solutions), axis=1)

    upper = data[np.argwhere(data[:, 1] > 0).reshape(-1), :]
    upper_sort_args = np.argsort(upper[:, 0])
    upper = upper[upper_sort_args, :]

    lower = data[np.argwhere(data[:, 1] < 0).reshape(-1), :]
    lower_sort_args = np.argsort(lower[:, 0])
    lower = lower[lower_sort_args, :]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax1.plot(np.append(upper[:, 0], lower[:, 0][::-1]), np.append(upper[:, 3], lower[:, 3][::-1]), color="tab:red",
             label=solution_names[1])
    ax1.plot(np.append(upper[:, 0], lower[:, 0][::-1]), np.append(upper[:, 2], lower[:, 2][::-1]), color="k",
             linestyle=(0, (5, 10)), label=solution_names[0])
    ax2.plot(np.append(upper[:, 0], lower[:, 0][::-1]), np.append(upper[:, 4], lower[:, 4][::-1]), color="tab:green",
             label=solution_names[2])
    ax2.plot(np.append(upper[:, 0], lower[:, 0][::-1]), np.append(upper[:, 2], lower[:, 2][::-1]), color="k",
             linestyle=(0, (5, 10)), label=solution_names[0])
    fig.suptitle(f"Solutions at Mach: {parameters[0]:.3f}, AoA: {np.rad2deg(parameters[1]):.3f} deg")
    ax1.legend()
    ax2.legend()
    fig.savefig(os.path.join(path, f"mach{parameters[0]:.3f}_AoA{np.rad2deg(parameters[1]):.3f}.pdf"))

root_dir = r"/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order"
plots_dir = r"/home/alex/Desktop/Thesis_plots"

snapshot_sweep = 1
dof_sweep = 0

#### Snapshots sweep ####
if snapshot_sweep:
    run_number = 0
    n_eval_pts = 5
    plot_solutions = 0

    nn_L2_error = []
    nn_Linf_error = []
    lspg_L2_error = []
    lspg_Linf_error = []
    timings = []
    n_snapshots = []

    for dir in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, dir)):
            continue
        if "snapshots" not in dir:
            continue
        dir_id = [int(x) for x in re.findall(r'\d+', dir)]
        if dir_id[-1] != run_number or dir_id[1] != n_eval_pts:
            continue
        n_snapshots.append(dir_id[0])

        if plot_solutions:
            sol_plots_dir = os.path.join(plots_dir, dir + "_solutions")
            if os.path.exists(sol_plots_dir):
                shutil.rmtree(sol_plots_dir)
            os.mkdir(sol_plots_dir)

        for file in os.listdir(os.path.join(root_dir, dir)):
            if not file.endswith('.txt'):
                continue

            if "testing_matrix" in file:
                fom_sol = np.genfromtxt(os.path.join(root_dir, dir, file), delimiter="  ")
            elif file == "pod_nnrom_solution.txt":
                nn_sol = np.genfromtxt(os.path.join(root_dir, dir, file), delimiter="  ")
            elif file == "lspg_rom_solution_matrix.txt":
                lspg_sol = np.genfromtxt(os.path.join(root_dir, dir, file), delimiter="  ")
            elif file == "timings.txt":
                timings.append(np.genfromtxt(os.path.join(root_dir, dir, file), delimiter=":")[:, 1])
            elif file == "point_locations.txt":
                points = []
                with open(os.path.join(root_dir, dir, file)) as f:
                    for line in f:
                        points.append([float(x) for x in line.strip().split()])
                f.close()
                points = np.array(points)
            elif "testing_parameters" in file:
                params = np.genfromtxt(os.path.join(root_dir, dir, file), delimiter="  ")

        if plot_solutions:
            for i in range(n_eval_pts):
                plot_surface_pressure(points, params[:, i],
                                      np.vstack((fom_sol[:, i], nn_sol[:, i], lspg_sol[:, i])).T,
                                      ["FOM", "NN-ROM", "LSPG-ROM"], sol_plots_dir)

        nn_L2 = la.norm(fom_sol - nn_sol, axis=0) / la.norm(fom_sol, axis=0)
        nn_Linf = la.norm(fom_sol-nn_sol, ord=np.inf, axis=0) / la.norm(fom_sol, ord=np.inf, axis=0)
        lspg_L2 = la.norm(fom_sol - lspg_sol, axis=0) / la.norm(fom_sol, axis=0)
        lspg_Linf = la.norm(fom_sol - lspg_sol, ord=np.inf) / la.norm(fom_sol, ord=np.inf, axis=0)

        nn_L2_error.append(np.mean(nn_L2))
        nn_Linf_error.append(np.mean(nn_Linf))
        lspg_L2_error.append(np.mean(lspg_L2))
        lspg_Linf_error.append(np.mean(lspg_Linf))

        print(f"\nRun: {dir}"
              f"NN L2 error: {nn_L2_error[-1]}\n"
              f"NN Linf error: {nn_Linf_error[-1]}\n"
              f"LSPG L2 error: {lspg_L2_error[-1]}\n"
              f"LSPG Linf error: {lspg_Linf_error[-1]}")

    timings = np.array(timings)
    nn_L2_error = np.array(nn_L2_error)
    nn_Linf_error = np.array(nn_Linf_error)
    lspg_L2_error = np.array(lspg_L2_error)
    lspg_Linf_error = np.array(lspg_Linf_error)
    n_snapshots = np.array(n_snapshots)

    order = np.argsort(n_snapshots)
    n_snapshots = n_snapshots[order]

    fig1, ax1 = plt.subplots()
    ax1.plot(n_snapshots, nn_L2_error[order], label=r"NN-ROM $L_2$ error")
    ax1.plot(n_snapshots, nn_Linf_error[order], label=r"NN-ROM $L_\infty$ error")
    ax1.plot(n_snapshots, lspg_L2_error[order], label=r"LSPG-ROM $L_2$ error")
    ax1.plot(n_snapshots, lspg_Linf_error[order], label=r"LSPG-ROM $L_\infty$ error")
    ax1.set_xlabel("Number of snapshots")
    ax1.set_ylabel("Error")
    ax1.set_title("ROM Error VS Number of Snapshots")
    ax1.legend()
    ax1.set_yscale('log')
    fig1.savefig(os.path.join(plots_dir, f"errors{dir_id[-1]}.pdf"))

    fig2, ax2 = plt.subplots()
    ax2.plot(n_snapshots, timings[:, 2][order], label="NN-ROM")
    ax2.plot(n_snapshots, timings[:, 3][order], label="LSPG-ROM")
    ax2.set_title("Evaluation Time Scaling")
    ax2.set_xlabel("Number of Snapshots")
    ax2.set_ylabel("Time [s]")
    ax2.legend()
    ax2.set_yscale('log')
    fig2.savefig(os.path.join(plots_dir, f"timings{dir_id[-1]}.pdf"))

#### DoF sweep ####
if dof_sweep:
    n_snapshots = 10
    n_eval_pts = 5
    plot_solutions = True

    nn_L2_error = []
    nn_Linf_error = []
    lspg_L2_error = []
    lspg_Linf_error = []
    timings = []
    run_parameters = []

    for dir in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, dir)):
            continue
        if "snapshots" not in dir:
            continue
        if "dof_sweep" not in dir:
            continue
        dir_id = [int(x) for x in re.findall(r'\d+', dir)]
        if dir_id[0] != n_snapshots or dir_id[1] != n_eval_pts:
            continue

        if plot_solutions:
            sol_plots_dir = os.path.join(plots_dir, dir + "_solutions")
            if os.path.exists(sol_plots_dir):
                shutil.rmtree(sol_plots_dir)
            os.mkdir(sol_plots_dir)

        for file in os.listdir(os.path.join(root_dir, dir)):
            if not file.endswith('.txt'):
                continue

            if "testing_matrix" in file:
                fom_sol = np.genfromtxt(os.path.join(root_dir, dir, file), delimiter="  ")
            elif file == "pod_nnrom_solution.txt":
                nn_sol = np.genfromtxt(os.path.join(root_dir, dir, file), delimiter="  ")
            elif file == "lspg_rom_solution_matrix.txt":
                lspg_sol = np.genfromtxt(os.path.join(root_dir, dir, file), delimiter="  ")
            elif file == "timings.txt":
                timings.append(np.genfromtxt(os.path.join(root_dir, dir, file), delimiter=":")[:, 1])
            elif file == "point_locations.txt":
                points = []
                with open(os.path.join(root_dir, dir, file)) as f:
                    for line in f:
                        points.append([float(x) for x in line.strip().split()])
                f.close()
                points = np.array(points)
            elif "testing_parameters" in file:
                params = np.genfromtxt(os.path.join(root_dir, dir, file), delimiter="  ")
            elif "run_parameters" in file:
                params_list = []
                with open(os.path.join(root_dir, dir, file)) as f:
                    for line in f:
                        params_list.append([x for x in line.strip().split(":")])
                f.close()
                run_parameters.append(np.array(params_list, dtype=object)[:, 1])


        if plot_solutions:
            for i in range(n_eval_pts):
                plot_surface_pressure(points, params[:, i],
                                       np.vstack((fom_sol[:, i], nn_sol[:, i], lspg_sol[:, i])).T,
                                       ["FOM", "NN-ROM", "LSPG-ROM"], sol_plots_dir)

        nn_L2 = la.norm(fom_sol - nn_sol, axis=0) / la.norm(fom_sol, axis=0)
        nn_Linf = la.norm(fom_sol - nn_sol, ord=np.inf, axis=0) / la.norm(fom_sol, ord=np.inf, axis=0)
        lspg_L2 = la.norm(fom_sol - lspg_sol, axis=0) / la.norm(fom_sol, axis=0)
        lspg_Linf = la.norm(fom_sol - lspg_sol, ord=np.inf) / la.norm(fom_sol, ord=np.inf, axis=0)

        nn_L2_error.append(np.mean(nn_L2))
        nn_Linf_error.append(np.mean(nn_Linf))
        lspg_L2_error.append(np.mean(lspg_L2))
        lspg_Linf_error.append(np.mean(lspg_Linf))

        print(f"\nRun: {dir}"
              f"NN L2 error: {nn_L2_error[-1]}\n"
              f"NN Linf error: {nn_Linf_error[-1]}\n"
              f"LSPG L2 error: {lspg_L2_error[-1]}\n"
              f"LSPG Linf error: {lspg_Linf_error[-1]}")

    timings = np.array(timings)
    nn_L2_error = np.array(nn_L2_error)
    nn_Linf_error = np.array(nn_Linf_error)
    lspg_L2_error = np.array(lspg_L2_error)
    lspg_Linf_error = np.array(lspg_Linf_error)
    run_parameters = np.array(run_parameters, dtype=object)
    n_dofs = np.array(run_parameters[:, 2].astype(float))
    order = np.argsort(n_dofs)
    n_dofs = n_dofs[order]

    fig1, ax1 = plt.subplots()
    ax1.plot(n_dofs, nn_L2_error[order], label=r"NN-ROM $L_2$ error")
    ax1.plot(n_dofs, nn_Linf_error[order], label=r"NN-ROM $L_\infty$ error")
    ax1.plot(n_dofs, lspg_L2_error[order], label=r"LSPG-ROM $L_2$ error")
    ax1.plot(n_dofs, lspg_Linf_error[order], label=r"LSPG-ROM $L_\infty$ error")
    ax1.set_xlabel("Number of DoFs")
    ax1.set_ylabel("Error")
    ax1.set_title("ROM Error VS Number of DoFs")
    ax1.legend()
    ax1.set_yscale('log')
    fig1.savefig(os.path.join(plots_dir, f"dof_sweep_errors{dir_id[0]}.pdf"))

    fig2, ax2 = plt.subplots()
    ax2.plot(n_dofs, timings[:, 2][order], label="NN-ROM")
    ax2.plot(n_dofs, timings[:, 3][order], label="LSPG-ROM")
    ax2.set_title("Evaluation Time Scaling")
    ax2.set_xlabel("Number of DoFs")
    ax2.set_ylabel("Time [s]")
    ax2.legend()
    ax2.set_yscale('log')
    fig2.savefig(os.path.join(plots_dir, f"dof_sweep_timings{dir_id[0]}.pdf"))




