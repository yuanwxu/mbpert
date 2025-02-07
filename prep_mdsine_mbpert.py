# Prepare mice data from MDSINE to be run on MBPert

import numpy as np

if __name__ == "__main__":

    DATA_DIR = "data/mdsine/cdiff/"
    OUT_DIR = "output/mdsine/cdiff/"

    counts = np.loadtxt(DATA_DIR + "counts.txt", usecols=range(1, 131))
    biomass = np.loadtxt(DATA_DIR + "biomass.txt", skiprows=1)
    meta = np.loadtxt(DATA_DIR + "metadata.txt", usecols=(2, 3), skiprows=1)

    species_names = np.loadtxt(DATA_DIR + "counts.txt", usecols=0, dtype=str)
    
    # Filter out species with low abundance across all time points in all samples
    mask = np.mean(counts == 0, axis=1) < 0.8
    counts = counts[mask]
    species_names = species_names[mask]

    # Use average biomass across the three qPCR replicates
    biomass = np.mean(biomass, axis=1)

    # Get species relative abundance
    counts_rel = counts / counts.sum(axis=0)

    # Get estimates of absolute abundance
    counts_abs = counts_rel * biomass

    # Common perturbation matrix for all mice
    T = 56
    P = np.zeros((T+1, 2))
    P[0,0] = 1
    P[28, 1] = 1

    np.savetxt(OUT_DIR + "X.txt", counts_abs / 1e9)
    np.savetxt(OUT_DIR + "P.txt", P, fmt="%i")
    np.savetxt(OUT_DIR + "meta.txt", meta)
    np.savetxt(OUT_DIR + "species_names.txt", species_names, fmt="%s")
