from mbpert.simhelper import pert_mat, get_ode_params, mbpert_split, mbpert_writer

if __name__ == '__main__':
    # Simulation Example: 10 speices, all single-node perturbations + 50% of each of the
    # k-node combo for k = 2,...,5
    n_species = 10
    n_conds_list = [10] + [
        round(0.5 * math.comb(n_species, k)) for k in range(2, 6)
    ]
    p = pert_mat(n_species, list(range(1, 6)), n_conds_list)
    r, A, eps, X_ss = get_ode_params(n_species, p, seed=0)

    n_conds = sum(n_conds_list)
    x0 = 0.1 * np.ones(n_species * n_conds)  # initial state chosen arbitrary

    # Split into train and test set (70-30)
    # Set `shuffle=False` to keep all single-node conditions in training set
    split_outputs = mbpert_split(x0, X_ss, p, test_size=0.3, shuffle=False)

    # Save results
    mbpert_writer(split_outputs, ode_params=dict(A=A, r=r, eps=eps))
