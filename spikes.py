# -*- coding: utf-8 -*-
'''
Class to read Kilosorted & Phy-curated units into Yggdrasil format
Use:
    (1) call init with all parameters defined except 'name' to generate new object
    (2) call init with only 'name' to load object for further processing
'''

from os.path import exists, join, isfile
from os import listdir
import re
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore


class Spikes:

    '''Spikes and related functions from cells from a single session

    Attributes:
        name (string):                       path to spikes file
        spike_channel (1xunits array):       mapping between unit index and electrode site
        spikes (1xunits array of 1xspikes):  ragged array of spike times, each key is a unit
        mua_channel (1xunits array):         mapping between mua index and electrode site
        mua (1xunits array of 1xspikes):     ragged array of mua times, each key is a unit
        spike_classification (1xunits array): putative cell type of good units   

    Methods:
        load
        save
        subset_by_channel
        subset_by_time
        assign_classification

    Functions:
        calc_fr
        calc_binned_spikes
        calc_mua_score
    '''

    def __init__(self, name="", rec_file_path="", ks_file_path="", is_phy_curated=True,
                 start=None, end=None, include_mua=False):
        '''Loads or builds spiketimes and relevant variables

        Parameters:
            name (string):              optional, path to spikes file
            rec_file_path (string):     optional, path to imec0 directory with ap.bin
            ks_file_path (string):      optional, path to directory with Kilosort outputs
            is_phy_curated (bool):      optional, reads KS labels if false
            start (float):              optional, start of gate (seconds)
            end (float):                optional, end of gate (seconds)
            include_mua (bool):         optional, stores MUA channels and spikes if true
        '''

        self.name = name

        # if instance already defined
        if exists(self.name):
            self.load()
        else:
            if name:
                raise Exception(f"Cannot find file {self.name}")
            self.rec_file_path = rec_file_path
            self.ks_file_path = ks_file_path
            self.is_phy_curated = is_phy_curated
            self.rate = 30000

            # get cluster labels
            # npyx package requires KS output & raw data to be in same folder,
            # so load these values manually
            if is_phy_curated:
                group = pd.read_table(self.ks_file_path+'/cluster_group.tsv')
                good_units = group['cluster_id'][group['group']
                                                 == 'good'].to_numpy()
                mua_units = group['cluster_id'][group['group']
                                                == 'mua'].to_numpy()
            else:
                group = pd.read_table(self.ks_file_path+'/cluster_KSLabel.tsv')
                good_units = group['cluster_id'][group['KSLabel']
                                                 == 'good'].to_numpy()
                mua_units = group['cluster_id'][group['KSLabel']
                                                == 'mua'].to_numpy()

            # read data from Kilosort output
            spike_clusters = np.load(
                join(self.ks_file_path, 'spike_clusters.npy'))
            spike_times = np.load(
                join(self.ks_file_path, 'spike_times_sec_adj.npy'))
            #best_channels = np.load(join(self.ks_file_path, 'clus_Table.npy'))
            best_channels = pd.read_csv(join(self.ks_file_path, 'cluster_info.tsv'), sep='\t')
            best_channels = best_channels.set_index('cluster_id')

            # get spike trains and peak channels for each unit
            self.spikes = []
            self.spike_channel = []
            self.cid = []
            for u in good_units:
                try:
                    self.spikes.append(spike_times[spike_clusters == u])
                    self.spike_channel.append(best_channels.ch[u]) #best_channels[u, 1]
                    self.cid.append(u)
                except:
                    print(f'Good unit {u} not in dataset.')

            # repeat for MUA
            self.mua = []
            self.mua_channel = []
            if include_mua:
                for u in mua_units:
                    try:
                        self.mua.append(spike_times[spike_clusters == u])
                        self.mua_channel.append(best_channels[u, 1])
                    except:
                        print(f'MUA unit {u} not in dataset.')

            if (start is not None) and (end is not None):  # if gate start/end provided
                self.start = start
                self.end = end
                intervals = np.array([start, end])
                # get spikes in [start, end] and shift spikes so start = 0 seconds
                self.subset_by_time(intervals)
                for u in range(len(self.spikes)):
                    self.spikes[u] -= start
                for u in range(len(self.mua)):
                    self.mua[u] -= start

    def load(self):
        '''Load spikes definition file'''
        name = self.name
        with open(self.name, 'rb') as file:
            temp_dict = pickle.load(file)
        self.__dict__.update(temp_dict)
        self.name = name

    def save(self):
        '''Save spikes to file'''
        with open(self.name, 'wb') as file:
            pickle.dump(self.__dict__, file)

    # %% Subsetting functions
    def subset_by_channel(self, subset_channels):
        '''Subset spikes on specific channels, eg after running electrodes.subset_by_location()'''
        subset_spikes = []
        subset_spike_channel = []
        subset_spike_indices = []
        subset_mua = []
        subset_mua_channel = []

        for u in subset_channels:
            spike_idx = [i for i, x in enumerate(self.spike_channel) if x == u]
            for i in spike_idx:
                subset_spike_indices.append(i)
            for unit in [self.spikes[i] for i in spike_idx]:
                subset_spikes.append(unit)
                subset_spike_channel.append(u)
            mua_idx = [i for i, x in enumerate(self.mua_channel) if x == u]
            for unit in [self.mua[i] for i in mua_idx]:
                subset_mua.append(unit)
                subset_mua_channel.append(u)

        self.spikes = subset_spikes#np.array(subset_spikes, dtype=object)
        self.spike_channel = subset_spike_channel
        self.mua = subset_mua#np.array(subset_mua, dtype=object)
        self.mua_channel = subset_mua_channel
        if hasattr(self, 'spike_classification'):
            self.spike_classification = self.spike_classification.iloc[subset_spike_indices]

    def subset_by_time(self, intervals):
        '''Subset spikes within windows, e.g. during epochs or running'''
        if isinstance(intervals, list):
            intervals = np.asarray(intervals)
        if intervals.ndim == 1:  # if just 1 interval
            # convert vector to 2D array with 1 row
            intervals = intervals[np.newaxis]

        subset_spikes = []
        for u in range(len(self.spikes)):
            i = intervals[0]
            subset_spikes.append(
                self.spikes[u][(self.spikes[u] >= i[0]) & (self.spikes[u] <= i[1])])
            if len(intervals) > 1:
                for i in intervals[1:]:
                    subset_spikes[u].append(self.spikes[u][(self.spikes[u] >= i[0])
                                                           & (self.spikes[u] <= i[1])])

        # repeat for mua
        subset_mua = []
        for u in range(len(self.mua)):
            i = intervals[0]
            subset_mua.append(
                self.mua[u][(self.mua[u] >= i[0]) & (self.mua[u] <= i[1])])
            if len(intervals) > 1:
                for i in intervals[1:]:
                    subset_mua[u].append(self.mua[u][(self.mua[u] >= i[0])
                                                     & (self.mua[u] <= i[1])])
        
        self.spikes = subset_spikes
        self.mua = subset_mua
        
    def assign_classification(self, search_string):
        '''Store list of putative cell types
        Input: all or part of the name of a .csv file inside rec_file_path
        which has a header row 'cell_type' and 1 row per unit
        see classifyCellTypes in emilyasterjones fork of CellExplorer
        for how to generate this file
        eg: spikes.assign_classification('monoSynType')'''
        
        # find file
        files = [f for f in listdir(self.rec_file_path) if isfile(join(self.rec_file_path,f))]
        for f in files:
            if re.search('.*'+search_string+'.*',f):
                cell_types = pd.read_csv(join(self.rec_file_path,f))
                break
        self.spike_classification = cell_types
        
    def subset_by_classification(self, cell_types):
        '''Subset spikes by putative cell type(s)'''
        if not hasattr(self, 'spike_classification'):
            raise Exception('No cell types defined yet. Use assign_classification() to do this first.')
        subset_spikes = []
        subset_spike_channel = []
        subset_spike_indices = []

        for c in cell_types:
            for u in range(len(self.spikes)):
                if self.spike_classification.iloc[u].item()==c:
                    subset_spikes.append(self.spikes[u])
                    subset_spike_channel.append(self.spike_channel[u])
                    subset_spike_indices.append(u)

        self.spikes = subset_spikes
        self.spike_channel = subset_spike_channel
        self.spike_classification = self.spike_classification.iloc[subset_spike_indices]


# %% External functions


def calc_fr(spikes, start, end):
    '''Calculate firing rate within an interval'''
    # if just 1 unit, convert to same format as multiple units
    if type(spikes[0]) == np.float64:
        init_spikes = spikes
        spikes = []
        spikes.append(init_spikes)
        spikes = np.asarray(spikes)
    fr = np.empty(len(spikes))
    for u in range(len(spikes)):
        fr[u] = np.count_nonzero(spikes[u][(spikes[u] >= start) &
                                           (spikes[u] <= end)])/((end-start))
    return fr


def calc_binned_spikes(spikes, start, end, window=0.002):
    '''Calculate firing rate across session binned by windows'''
    intervals = np.arange(start, end, window)
    binned_spikes = np.empty((len(spikes), len(intervals)-1), dtype=np.uint16)
    for u in range(len(spikes)):
        hist = np.histogram(spikes[u], intervals)[0]
        binned_spikes[u, :] = hist.astype(np.uint16)
    return binned_spikes, intervals[:-1]


def calc_mua_score(spikes, start, end, window=0.002):
    '''Calculate Z-scored MUA trace, timestamps, mean, and SD
    To include MUA channels, use spikes=np.append(spikes, mua)'''
    fr, timestamps = calc_binned_spikes(spikes, start, end, window)
    fr_mean = np.mean(fr, axis=0)
    mua_trace = zscore(fr_mean)
    mua_mean = np.mean(fr_mean)
    mua_sd = np.std(fr_mean)
    return mua_trace, timestamps, mua_mean, mua_sd
