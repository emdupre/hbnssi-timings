import os
import glob
import pandas as pd
from shutil import copyfile

for seq in range(1, 4):
    event_onsets = []
    run_time = 0  # track current time in miliseconds
    df = pd.read_csv('flanker_sequence{}.txt'.format(seq), sep='\t')

    for i in df.index:
        if df.iloc[i]['Stimtype'] == 'nostim':
            event_onsets.append(run_time)
            run_time += df.iloc[i]['Duration[ms]']
        else:
            run_time += 500  # to acount for fixation
            event_onsets.append(run_time)
            run_time += df.iloc[i]['Duration[ms]']
            run_time += 1150  # to account for null stimuli

    event_onsets = [e / 1000 for e in event_onsets]
    duration = [d / 1000 for d in df['Duration[ms]']]

    tsv = pd.DataFrame(data={'onset': event_onsets,
                             'duration': duration,
                             'trial_type': df['Stimtype'],
                             'stim_file': df['Img']})
    tsv.to_csv('task-FLANKERTASK{}_events.tsv'.format(seq),
               sep='\t', index=False)

subjects = glob.glob('sub-*sessions.tsv')
data_dir = '/media/emdupre/emdupre-ext4/hbn-ssi/fmriprep'

for s in subjects:
    df = pd.read_csv(s, sep='\t')
    for i in df.index:
        subid, sesid, _, fseq, _, _ = df.iloc[i].values
        fname = f'sub-00{subid}_ses-SSV{sesid}_task-FLANKERTASK_events.tsv'
        try:
            copyfile(f'task-FLANKER{fseq}_events.tsv',
                     os.path.join(data_dir, f'sub-00{subid}',
                                  f'ses-SSV{sesid}', 'func',  fname))
        except FileNotFoundError:  # at least one subject is missing BOLD files
            pass
