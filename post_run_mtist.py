import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from mtist.infer_mtist import calculate_es_score as mtist_es_score

DATA_DIR = "mtist/mtist1.0/"
OUT_DIR = "output/mtist/"
MTIST_RESULT_DIR = DATA_DIR + 'mtist_datasets/'

df_meta = pd.read_csv(DATA_DIR + f"mtist_datasets/mtist_metadata.csv")

# Read ES score results spread across different computing tasks. 
# `TASK_MAX` the number of tasks used in the job script to divide the workload into.
# Alternatively, provide a list TASK_IDS of task IDs.
# `n_runs` the number of runs of MBPert per dataset, default 1. If greater than 1, will
# add a new column indicating the run number
def read_ES_scores(run_dir="run/", TASK_MAX=24, TASK_IDS = None, n_runs=1, n_datasets=648):
    es_scores = []
    task_ids = range(1, TASK_MAX+1) if not TASK_IDS else TASK_IDS
    for i in task_ids:
        es_scores.append(pd.read_csv(OUT_DIR + run_dir + f"es_score_task{i}.csv"))
    es_scores_df = pd.concat(es_scores)

    assert es_scores_df.shape == (n_datasets*n_runs, 2)

    if n_runs > 1:
        es_scores_df['run_number'] = np.tile(range(1, n_runs + 1), n_datasets)
    return es_scores_df

# Read ES score results directly from mtist repo
def read_ES_scores2():
    result_files = {'LR': MTIST_RESULT_DIR + 'default_inference_result/default_es_scores.csv',
                    'RR': MTIST_RESULT_DIR + 'ridge_CV_inference_result/ridge_CV_es_scores.csv',
                    'ENR': MTIST_RESULT_DIR + 'elasticnet_CV_inference_result/elasticnet_CV_es_scores.csv',
                    'MK': MTIST_RESULT_DIR + 'mkseqspike_inference_result/mkseqspike_es_scores.csv'}

    es_dict = {}
    for label, f in result_files.items():
        es = pd.read_csv(f, index_col=0)
        es = es.reset_index().rename(columns={'index': 'did', 'raw': 'ES_score'})
        es_dict[label] = es
    
    return es_dict


# Compute median ES scores grouped by species. 
def es_score_by_species(df, agg_func=np.median):
    return df.groupby('n_species').agg({'ES_score': agg_func})

# ES score for 100 species constrained to strong interacting species, as described  
# in Hussey et al, bioRxiv 2022
def es_score_100species_subset(run_dir, TASK_IDS, last_did_100species=35):
    # Get ground truth ids corresponding to all datasets
    df_gt = pd.read_csv(DATA_DIR + f"mtist_datasets/mtist_metadata.csv")
    gts = dict(zip(df_gt['did'], df_gt['ground_truth'])) # dict mapping dataset id to ground truth id

    es_score_100 = {'did': [], 'ES_score': [], 'run_number': []}
    for i in TASK_IDS:
        with open(OUT_DIR + run_dir + f'inferred_aij_dict_task{i}.pkl', 'rb') as f:
            inferred_aij = pickle.load(f)
        # Loop over dataset ids for each task
        for did in inferred_aij.keys():
            if did > last_did_100species:
                return pd.DataFrame(es_score_100)
                
            gt = gts[did].replace('gt', 'aij')
            true_aij = np.loadtxt(DATA_DIR + f"ground_truths/interaction_coefficients/{gt}.csv", delimiter=',')
            # Apply the mask
            mask = np.abs(true_aij) < 0.25
            true_aij[mask] = 0

            if isinstance(inferred_aij[did], list): # Multiple runs per did
                # Unpack runs within each dataset id
                for k, aij_k in enumerate(inferred_aij[did]):
                    # Apply the mask
                    aij_k[mask] = 0
                    # Compute ES_score on the subset 
                    sc = mtist_es_score(true_aij, aij_k)
                    es_score_100['ES_score'].append(sc)
                    es_score_100['run_number'].append(k+1)
                    es_score_100['did'].append(did)
            else: # assume only one inferred aij per did
                aij = inferred_aij[did]
                # Apply the mask
                aij[mask] = 0
                # Compute ES_score on the subset 
                sc = mtist_es_score(true_aij, aij)
                es_score_100['ES_score'].append(sc)
                es_score_100['run_number'].append(1)
                es_score_100['did'].append(did)
    
    return pd.DataFrame(es_score_100)

# Read and compute subset ES scores for mtist paper methods directly from mtist repo
def es_score_100species_subset2(last_did_100species=35):
    # Get ground truth ids corresponding to all datasets
    df_gt = pd.read_csv(DATA_DIR + f"mtist_datasets/mtist_metadata.csv")
    gts = dict(zip(df_gt['did'], df_gt['ground_truth'])) # dict mapping dataset id to ground truth id

    out = {}
    for did in range(last_did_100species + 1):
        gt = gts[did].replace('gt', 'aij')
        true_aij = np.loadtxt(DATA_DIR + f"ground_truths/interaction_coefficients/{gt}.csv", delimiter=',')
        # Apply the mask
        mask = np.abs(true_aij) < 0.25
        true_aij[mask] = 0
        
        for method in ['LR', 'RR', 'ENR', 'MK']:
            if did == 0:
                out[method] = {'did': [], 'ES_score': [], 'run_number': []}

            if method == 'LR':
                aij = np.loadtxt(MTIST_RESULT_DIR + f'default_inference_result/default_inferred_aij_{did}.csv', delimiter=',')
            
            if method == 'RR':
                aij = np.loadtxt(MTIST_RESULT_DIR + f'ridge_CV_inference_result/ridge_CV_inferred_aij_{did}.csv', delimiter=',')
          
            if method == 'ENR':
                aij = np.loadtxt(MTIST_RESULT_DIR + f'elasticnet_CV_inference_result/elasticnet_CV_inferred_aij_{did}.csv', delimiter=',')
          
            if method == 'MK':
                aij = np.loadtxt(MTIST_RESULT_DIR + f'mkseqspike_inference_result/mkseqspike_inferred_aij_{did}.csv', delimiter=',')

            # Apply the mask
            aij[mask] = 0
            # Compute ES_score on the subset 
            sc = mtist_es_score(true_aij, aij)
            out[method]['ES_score'].append(sc)
            out[method]['run_number'].append(1)
            out[method]['did'].append(did)

    for k in out.keys():
        out[k] = pd.DataFrame(out[k])

    return out


# Get a tidy data frame of ES scores for all methods and species configs
def get_tidy_es_scores(es_full_dict, es_restricted_dict, df_meta):
    # To hold es scores for individual methods evaluated on all species configs
    esf_out = [] 
    for label, esf in es_full_dict.items():
        if label == 'MBPert': # multiple runs per did
            esf = esf.groupby('did')\
                    .agg({'ES_score': 'median'})\
                    .reset_index()

        esf['method'] = label
        esf_out.append(esf)
    
    # Combine ES scores for all methods on all species configs
    esf_out = pd.concat(esf_out)
    esf_out = pd.merge(esf_out, df_meta[['did', 'n_species']], on='did')
    esf_out['config'] = esf_out['n_species'].astype('str') + ' species'
    esf_out = esf_out.drop('n_species', axis=1)

    # To hold es scores for individual methods evaluated on a subset of 
    # interactions of 100 speces configs
    esr_out = []
    for label, esr in es_restricted_dict.items():
        if label == 'MBPert':
            esr = esr.groupby('did')\
                    .agg({'ES_score': 'median'})\
                    .reset_index()
        else:
            esr = esr.drop('run_number', axis=1)
        
        esr['method'] = label
        esr['config'] = '100 species subset'
        esr_out.append(esr)
    
    # Combine ES scores for all methods on subset of 100 species interactions
    esr_out = pd.concat(esr_out)

    # Combine everything
    out = pd.concat([esf_out, esr_out])

    return out

# ES scores of MBPert on all datasets across multiple runs
es_scores_mbp = read_ES_scores(run_dir='old_25June2024/', TASK_MAX=216, n_runs=50, n_datasets=648)

# ES scores of other methods in MTIST paper
es_scores_others = read_ES_scores2()

# MBPert ES scores restricted to a subset of interactions of 100 species configurations
es_scores_100s_sub_mbp = es_score_100species_subset(run_dir='old_25June2024/', TASK_IDS=range(1, 13))

# Other methods' ES scores restricted to a subset of interactions of 100 species configurations
es_scores_100s_sub_others = es_score_100species_subset2()

es_full_dict = {'MBPert': es_scores_mbp, **es_scores_others}
es_restricted_dict = {'MBPert': es_scores_100s_sub_mbp, **es_scores_100s_sub_others}

es_scores = get_tidy_es_scores(es_full_dict, es_restricted_dict, df_meta)

# Plot
g = sns.FacetGrid(es_scores, col='config', hue='method', sharex=False, sharey=False)
g.map_dataframe(sns.kdeplot, x='ES_score', fill=True)
g.add_legend()
g.savefig(OUT_DIR + "mtist_es_score.pdf")





# # Median ES score by number of species, for each run
# es_scores_agg = es_scores.groupby('run_number').apply(es_score_by_species)
# print(es_scores_agg.head(10))

# # ES score statistics by number of species
# print(es_scores_agg.reset_index().groupby('n_species').ES_score.describe())

# # Comparing MBPert ES scores with methods in the MTIST paper for 10 species communities
# ax_10s = sns.kdeplot(data=es_scores_agg.reset_index().query('n_species == 10'), 
#                      x='ES_score', fill=True, color='grey')
# ax_10s.axvline(x=0.78, ymin=0, ymax=0.2, color='orangered')
# ax_10s.text(0.78-0.006, 15, "LR", fontdict={'color': 'orangered'})
# ax_10s.axvline(x=0.7804, ymin=0, ymax=0.2, color='green')
# ax_10s.text(0.78-0.006, 18, "RR", fontdict={'color': 'green'})
# ax_10s.axvline(x=0.71, ymin=0, ymax=0.2, color='blue')
# ax_10s.text(0.71+0.002, 15, "ENR", fontdict={'color': 'blue'})
# ax_10s.axvline(x=0.81, ymin=0, ymax=0.2, color='purple')
# ax_10s.text(0.81-0.006, 15, "MK", fontdict={'color': 'purple'})

# ax_10s.get_figure().savefig(OUT_DIR + "ES_score_compare_10species.png", dpi=300)


# # Comparing MBPert ES scores with methods in the MTIST paper for 100 species communities
# ax_100s = sns.kdeplot(data=es_scores_agg.reset_index().query('n_species == 100'), 
#                       x='ES_score', fill=True, color='grey')
# ax_100s.axvline(x=0.51, ymin=0, ymax=0.2, color='orangered')
# ax_100s.text(0.51-0.002, 50, "LR", fontdict={'color': 'orangered'})
# ax_100s.axvline(x=0.52, ymin=0, ymax=0.2, color='green')
# ax_100s.text(0.52+0.001, 50, "RR", fontdict={'color': 'green'})
# ax_100s.axvline(x=0.5102, ymin=0, ymax=0.2, color='blue')
# ax_100s.text(0.51-0.002, 60, "ENR", fontdict={'color': 'blue'})
# ax_100s.axvline(x=0.50, ymin=0, ymax=0.2, color='purple')
# ax_100s.text(0.50+0.001, 50, "MK", fontdict={'color': 'purple'})

# ax_100s.get_figure().savefig(OUT_DIR + "ES_score_compare_100species.png", dpi=300)


# # Comparing the subset of strong interactions in the 100 species case
# es_scores_100 = es_score_100species_subset(run_dir="", TASK_IDS=range(1, 13))

# ax_100s_subset = sns.kdeplot(data=es_scores_subset_100species\
#                                    .groupby('run_number')\
#                                    .agg({'ES_score': np.median}),
#                              x='ES_score', fill=True)
# ax_100s_subset.set(title="ES score for 100 species community\n   restricting to strong interacting species")
# ax_100s_subset.get_figure().save(OUT_DIR + "ES_score_100species_subset.png", dpi=300)
