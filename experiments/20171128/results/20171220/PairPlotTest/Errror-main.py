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
#sns.set(style="ticks")

METHYLATION_FILE = 'data/GSE76399_data_with_probe_ann.txt'
#METHYLATION_FILE = 'data/short_sample_data.txt'

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
    print(data)
    print("hej")
    #g = sns.distplot(data, kde=False, bins = 150)
    #plt.show()
    return data

def HistPlotM(df,SAT_geo_accession,insulin_geo_dict):
    data = df[insulin_geo_dict['resistant'].append(insulin_geo_dict['sensitive'])].values.flatten() 

    #sns.distplot(data, kde=False, bins = 150)
    #plt.show()
    return data

def HistPlotPair(data_non_drop, data_drop):
    fig, ax = plt.subplots()
    print(data_non_drop)
    print(len(data_non_drop))
    print("jek")
    print(data_drop)
    print(len(data_drop)) 
    sns.distplot(data_non_drop, kde=False, bins = 150, ax=ax)
    sns.distplot(data_drop, kde=False, bins = 150, ax=ax)
    plt.show()


def HistPlotPairShit(df,SAT_geo_accession,insulin_geo_dict):
    data = df.values.flatten()
    print(len(data)) 
    #insulin_geo_dict = {x:SAT_patient_data[SAT_patient_data['Name']==x]["beta-values for every patient for probe in probe_db"] for x in probe_df}
    sns.distplot(data, kde=False, bins = 150)
    plt.show()
   
    


def buildGeneSet(clean_methylation_data):
    '''Takes clean methylation data, with gene accessions stored in sets, and returns the set of all genes.'''
    gene_set = set()
    for i in clean_methylation_data['UCSC_RefGene_Accession']:
        gene_set |= i # Union of gene_set and i
    return gene_set

def buildMatrix(clean_methylation_data, SAT_geo_accessions, gene_accession):
    '''Picks out all probes and samples for a given gene.
    Returns a numpy array of probes and samples.'''
    # Picking rows
    rows = clean_methylation_data['UCSC_RefGene_Accession'].map(lambda probe_gene_set: gene_accession in probe_gene_set)
    df = clean_methylation_data[rows]
    if df.shape[0] == 1:
        return None,None
    probes = df['Name']
    # Picking columns
    df = df[SAT_geo_accessions]
    # Converts to numpy
    df = df.values
    return df, probes


def main():
    probe_list =[]
    short_SAT_methylation_data_list = []
    # Reading data from file
    methylation_data = csvToDataFrame(METHYLATION_FILE)
    patient_data = csvToDataFrame(PATIENT_FILE)

    SAT_methylation_data, SAT_patient_data = cleanData(methylation_data, patient_data)

    # Picks out SAT samples (columns)
    SAT_geo_accession = SAT_patient_data['GEO_accession']
    insulin_geo_dict = {x:SAT_patient_data[SAT_patient_data['Insulin_state']==x]['GEO_accession'] for x in ['resistant','sensitive']}

	## Plot input data from beta-values
    data_non_drop = HistPlot(SAT_methylation_data,SAT_geo_accession,insulin_geo_dict)

	#Drop bad rows
    SAT_methylation_data = dropBadRows(SAT_methylation_data,SAT_geo_accession)
    
	## Converts beta to m-values
    #SAT_methylation_data = columnsFromBetaToM(SAT_methylation_data,SAT_geo_accession)

    ## Plot input data from M values
    #data_drop = HistPlotM(SAT_methylation_data,SAT_geo_accession,insulin_geo_dict)
    
    ##Plots both pre and post removal of poor probes
    #HistPlotPair(data_non_drop, data_drop)

    gene_set = buildGeneSet(SAT_methylation_data)
    
    num_genes = len(gene_set)


    for igene, gene in enumerate(gene_set):
        matrix,probes = buildMatrix(SAT_methylation_data,SAT_geo_accession,gene)
        if matrix is None:
            print('Too few probes found for gene {0} ({1}/{2})'.format(gene,igene+1,num_genes))
            continue
        probe_list.append(probes) 
        print('Working on gene {0} ({1}/{2})'.format(gene,igene+1,num_genes))
               
	##Remove duplicate probes
    probe_df=pd.concat(probe_list) 
    probe_df.drop_duplicates(inplace=True)
   
    ##takes only genes that have more than 1 probe per gene
    counter = 0
    #print(probe_df)
    for i in probe_df:
        short_SAT_methylation_data = (SAT_methylation_data[SAT_methylation_data.Name.isin([i])])
        short_SAT_methylation_data = short_SAT_methylation_data.iloc[:,13:89]
        short_SAT_methylation_data_list.append(short_SAT_methylation_data)
      	print(counter)
        counter +=1
        #short_SAT_methylation_data_list = short_SAT_methylation_data_list.iloc[:,13:89]    
	#print(short_SAT_methylation_data_list[0])
	#plots shit
    HistPlotPairShit(short_SAT_methylation_data,SAT_geo_accession,insulin_geo_dict)


if __name__=='__main__':
    main()




