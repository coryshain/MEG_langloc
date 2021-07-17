import sys
import os
import re
import numpy as np
import pandas as pd
import nltk
from scipy import io, signal

def stderr(x):
    sys.stderr.write(x)
    sys.stderr.flush()

def tag(x):
    if 'stim' in x:
        col = 'stim'
    else:
        col = 'Sentence'
    s = x[col].lower().split()
    tags = tuple([x[1] for x in nltk.pos_tag(s)])

    return tags

def is_noun(x):
    return x.startswith('NN') or x.startswith('PRP')

def is_verb(x):
    return x.startswith('VB') or x == 'MD'

def is_nonword(x):
    return x == 'NONW'

def simplify_tags(x):
    if is_noun(x):
        return 'N'
    if is_verb(x):
        return 'V'
    if is_nonword(x):
        return x
    return 'O'

def select_tag(x):
    pos = x['position'] - 1
    tag = x['postag'][pos]
    return tag

parse_subject = re.compile('.*FED([0-9]+)$')
parse_file = re.compile('data_(s|nw)([0-9]+)_trial([0-9]+).mat$')
parse_run = re.compile('.*run([0-9]).csv$')
parse_condpos = re.compile('(s|nw|stim)([0-9]+)')
SUBJ = 0
COND = 0
POS = 1
TRIAL = 2
S_FILES = ['stim%d.mat' % i for i in range(1, 13)]
NW_FILES = ['stim%d.mat' % i for i in range(31, 43)]

topdir = 'data/english/timing_langloc_Eng/behavioral_langloc/'

stderr('Loading stimulus data...\n')
dirs = [topdir + x for x in os.listdir(topdir) if os.path.isdir(topdir + x)]
Y = []
timestamps = []
for _dir in dirs:
    subject = os.path.basename(_dir)[:3]
    if int(subject) != 8 and int(subject) != 14: # Filter bugged subjects
        subject = 's' + subject

        # Process stimulus tables
        _Y = []
        for behmat in [_dir + '/' + x for x in os.listdir(_dir) if x.endswith('.csv')]:
            __Y = pd.read_csv(behmat)
            run = int(parse_run.search(behmat).group(1))
            if 'stim' in __Y:
                col = 'stim'
            else:
                col = 'Sentence'

            # Normalize column names and add missing data
            if 'Set' in __Y:
                del __Y['Set']
            if 'Fixation Time' in __Y:
                del __Y['Fixation Time']
            if 'Response time' in __Y:
                __Y['RT'] = __Y['Response time']
                del __Y['Response time']
            if 'Condition' in __Y:
                __Y['cond'] = __Y['Condition']
                del __Y['Condition']
            if 'Run' in __Y:
                __Y['run'] = __Y['Run']
                del __Y['Run']
            else:
                __Y['run'] = run
            if 'Trial' in __Y:
                __Y['trial'] = __Y['Trial']
                del __Y['Trial']
            else:
                __Y['trial'] = np.arange(len(__Y))
            __Y['response'] = __Y['Response']
            del __Y['Response']
            __Y['subject'] = subject
            __Y['postag'] = __Y.apply(tag, axis=1)

            # Tile out by word
            words = __Y[col].str.split(' ').apply(pd.Series, 1).stack().str.lower()
            words.index = words.index.droplevel(-1)
            words.name = 'word'
            __Y = __Y.join(words)
            del __Y[col]

            __Y['position'] = __Y.groupby('trial').cumcount() + 1
            _Y.append(__Y)
        _Y = pd.concat(_Y, axis=0).reset_index(drop=True)
        _Y['postag'] = _Y.apply(select_tag, axis=1)
        _Y['postagsimp'] = _Y['postag'].apply(simplify_tags)
        _Y['wlen'] = _Y['word'].str.len()
        _Y['trial'] = _Y.groupby(['cond', 'position']).cumcount() + 1
        Y.append(_Y)

        # Process timing tables
        # Try tables named 'sN.mat'/'nwN.mat'
        for basename in [x for x in os.listdir(_dir) if (parse_condpos.match(x) and parse_condpos.match(x).group(1) in ['nw', 's'])]:
            cond, position = parse_condpos.match(basename).groups()
            path = _dir + '/' + basename
            _timestamps = io.loadmat(path)[cond + position][0]
            _timestamps = pd.DataFrame({'time': _timestamps})
            _timestamps['trial'] = np.arange(len(_timestamps)) + 1
            if cond == 'nw':
                _timestamps['cond'] = 'N'
            else:
                _timestamps['cond'] = 'S'
            _timestamps['position'] = int(position)
            _timestamps['subject'] = subject
            timestamps.append(_timestamps)

        # Now try tables named 'stimN.mat'
        for basename in [x for x in os.listdir(_dir) if (x.startswith('stim') and (x in S_FILES or x in NW_FILES))]:
            cond, position = parse_condpos.match(basename).groups()
            position = int(position)
            path = _dir + '/' + basename
            _timestamps = io.loadmat(path)['stimtime'][0]
            _timestamps = pd.DataFrame({'time': _timestamps})
            _timestamps['trial'] = np.arange(len(_timestamps)) + 1
            if position > 12:
                _timestamps['cond'] = 'N'
                position -= 30
            else:
                _timestamps['cond'] = 'S'
                cond = 'S'
            _timestamps['position'] = int(position)
            _timestamps['subject'] = subject
            timestamps.append(_timestamps)


Y = pd.concat(Y, axis=0).reset_index(drop=True)
timestamps = pd.concat(timestamps, axis=0).reset_index(drop=True)
Y = Y.merge(timestamps, on=['subject', 'cond', 'trial', 'position'])
Y = Y.sort_values(['subject', 'run', 'time']).reset_index(drop=True)

stderr('Loading MEG data...\n')
topdir = 'data/english/'
dirs = [topdir + x for x in os.listdir(topdir) if x.startswith('FED')]

# Iterate subjects
X = []
subjects = set(Y['subject'].unique())
for _dir in dirs:
    # Get subject ID
    subject = 's' + parse_subject.match(_dir).groups()[SUBJ]
    if subject in subjects: # Only get MEG timecourses for subjects with valid behavioral data
        # Iterate condition (S,NW) + sequence position (word 1-8)
        for subdir in os.listdir(_dir):
            __dir = _dir + '/' + subdir
            stderr('  %s\n' % __dir)

            # Get indices and names of magnetometer channels
            channels = io.loadmat(__dir + '/channel_vectorview306_acc1.mat')['Channel']
            mag_channel_ix = [i for i, x in enumerate(channels['Type'][0]) if x[0] == 'MEG MAG']
            grad1_channel_ix = [i for i, (x, y) in enumerate(zip(channels['Type'][0], channels['Name'][0])) if (x[0] == 'MEG GRAD' and y[0].endswith('2'))]
            grad2_channel_ix = [i for i, (x, y) in enumerate(zip(channels['Type'][0], channels['Name'][0])) if (x[0] == 'MEG GRAD' and y[0].endswith('3'))]
            mag_channel_names = [channels['Name'][0][i][0] for i in mag_channel_ix]
            grad1_channel_names = [channels['Name'][0][i][0] for i in grad1_channel_ix]
            grad2_channel_names = [channels['Name'][0][i][0] for i in grad2_channel_ix]
            gradnorm_channel_names = ['MEGGN' + x[3:6] for x in mag_channel_names]

            # Get names of bad files to drop
            bad_trials = io.loadmat(__dir + '/brainstormstudy.mat')['BadTrials']
            if len(bad_trials):
                bad_trials = {x[0] for x in bad_trials[0]}
            else:
                bad_trials = {}

            # Iterate trials (individual presentations of the Nth word of condition C)
            for path in os.listdir(__dir):
                if path.startswith('data'):
                    # Get the name of the condition, the position index, and the trial index
                    matches = parse_file.match(path).groups()
                    cond, position, trial = matches[COND], int(matches[POS]), int(matches[TRIAL])
                    if cond == 'nw':
                        cond = 'N'
                    else:
                        cond = 'S'

                    if cond == 'S': # For now, we're just using the sentences condition
                        if path in bad_trials:
                            print(path)
                            Y = Y[~(
                                (Y.subject == subject) &
                                (Y.cond == cond) &
                                (Y.position == position) &
                                (Y.trial == trial)
                            )]
                        else:
                            # Load the timecourses (each 1.4s long measured every 2ms)
                            src = io.loadmat(__dir + '/' + path)
                            F = src['F']
                            F_mag = F[mag_channel_ix].T
                            F_grad1 = F[grad1_channel_ix].T
                            F_grad2 = F[grad2_channel_ix].T
                            F_gradnorm = np.sqrt(F_grad1 ** 2 + F_grad2 ** 2)

                            # Currently using gradnorm
                            # F = np.concatenate([F_mag, F_grad1, F_grad2, F_gradnorm], axis=1)
                            # channel_names = mag_channel_names + grad1_channel_names + grad2_channel_names + gradnorm_channel_names
                            F = F_gradnorm
                            channel_names = gradnorm_channel_names
                            time = src['Time'][0]

                            # Resample to 10ms chunks, i.e. 1/5 resolution
                            F = signal.resample(F, 140, axis=0)
                            time_ix = np.arange(3, len(time), 5) # starts at 3, midpoint of 10ms frame
                            time_base = Y[
                                (Y.subject == subject) &
                                (Y.cond == cond) &
                                (Y.position == position) &
                                (Y.trial == trial)
                            ].time.values
                            if len(time_base):
                                time_base = time_base[0]
                            else:
                                time_base = 0.
                            time = time[time_ix] + time_base

                            # Construct the output dataframe
                            _X = pd.DataFrame(F, columns=channel_names)

                            _X['subject'] = subject
                            _X['cond'] = cond
                            _X['position'] = position
                            _X['trial'] = trial
                            _X['time'] = time
                            X.append(_X)

Y.to_csv('data/english/Y.csv', sep=' ', index=False, na_rep='NaN')
X = pd.concat(X, axis=0)
X.to_csv('data/english/X.csv', sep=' ', index=False, na_rep='NaN')
