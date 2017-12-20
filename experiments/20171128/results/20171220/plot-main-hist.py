import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.stats import ttest_ind
from numpy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
from qvalues import qvalues
from multiprocessing import Pool
sns.set(color_codes=True)

METHYLATION_FILE = 'data/GSE76399_data_with_probe_ann.txt'
PATIENT_FILE = 'data/samples_clinical_data.txt'


def csvToDataFrame(filename):
    '''Reads tab-separated data from filename and returns a dataframe'''
    with open(filename,'r') as f:
        data = pd.read_csv(f,sep='\t')
    return data


def cleanData(methylation_data,patient_data):
    '''Takes raw methylation and patient data. Removes non-SAT data, removes probes without genes,
    and converts gene information from csv to sets.
    Returns cleaned methylation and patient data.'''
    SAT_patient_data = patient_data[patient_data['Tissue_type']=='subcutaneous adipose tissue']

    # Removing non-SAT data
    no_SAT_patient_data = patient_data[patient_data['Tissue_type']!='subcutaneous adipose tissue']
    SAT_methylation_data = methylation_data.drop(no_SAT_patient_data['GEO_accession'],axis=1)

    # Removing probes without genes
    gene_SAT_methylation_data = SAT_methylation_data.drop(SAT_methylation_data[SAT_methylation_data['UCSC_RefGene_Accession'].isnull()].index)
    # Converting genes to a set instead of semi-colon separated strings
    gene_SAT_methylation_data.loc[:,'UCSC_RefGene_Accession'] = gene_SAT_methylation_data['UCSC_RefGene_Accession'].str.split(';').apply(set)
    return gene_SAT_methylation_data, SAT_patient_data


def betaToM(beta):
    '''Converts a beta-value to a M-value.'''
    return np.log2(beta/(1-beta))


def columnsFromBetaToM(df, SAT_geo_accession):
    '''Turns the columns containing beta values into columns containing M-values.'''
    df.loc[:,SAT_geo_accession] = df.loc[:,SAT_geo_accession].applymap(betaToM)
    return df


def dropBadRows(df,SAT_geo_accession,eps=0.1):
    '''Drop all rows that contain values outside of (eps,1-eps) range.'''
    idxs = df[((df.loc[:,SAT_geo_accession] < eps) | (df.loc[:,SAT_geo_accession] >(1- eps))).any(1)].index
    df = df.drop(idxs)
    return df


def HistPlot(df,SAT_geo_accession,insulin_geo_dict):
    data = df[insulin_geo_dict['resistant'].append(insulin_geo_dict['sensitive'])].values.flatten() 

    g = sns.distplot(data, kde=False, bins = 150)
    plt.show()


def HistPlotM(df,SAT_geo_accession,insulin_geo_dict):
    data = df[insulin_geo_dict['resistant'].append(insulin_geo_dict['sensitive'])].values.flatten() 

    sns.distplot(data, kde=False, bins = 150)
    plt.show()
    

def main():
    # Reading data from file
    methylation_data = csvToDataFrame(METHYLATION_FILE)
    patient_data = csvToDataFrame(PATIENT_FILE)

    SAT_methylation_data, SAT_patient_data = cleanData(methylation_data, patient_data)

    # Picks out SAT samples (columns)
    SAT_geo_accession = SAT_patient_data['GEO_accession']
    insulin_geo_dict = {x:SAT_patient_data[SAT_patient_data['Insulin_state']==x]['GEO_accession'] for x in ['resistant','sensitive']}

	
	#Drop bad rows
    #SAT_methylation_data = dropBadRows(SAT_methylation_data,SAT_geo_accession)
    
    ## Plot input data from beta-values
    HistPlot(SAT_methylation_data,SAT_geo_accession,insulin_geo_dict)

	## Converts beta to m-values
    #SAT_methylation_data = columnsFromBetaToM(SAT_methylation_data,SAT_geo_accession)

    ## Plot input data from M values
    #HistPlotM(SAT_methylation_data,SAT_geo_accession,insulin_geo_dict)

    
if __name__=='__main__':
    main()




