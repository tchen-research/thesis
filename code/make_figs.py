import subprocess

files = [
    'ch1_unif_vs_spec',
    'ch2_sample_dist',
    'ch3_fig1-4',
    'ch4_CESMwCESM',
    'ch4_fig2',
    'ch4_RM_spike',
    'ch4_RM_split',
    'ch4_spin',
    'ch5_OR_est',
    'ch6_CGsquared',
    'ch6_sign',
    'ch6_sign_rat',
    'ch6_sign_spec',
    'ch7_fig1-2',
    'ch7_fig_pcr',
    'ch7_fig_sqrt',
    'ch7_step_qf',
    'ch8_fig1',
]

for file in files:
    subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 {file}.ipynb',shell=True)

