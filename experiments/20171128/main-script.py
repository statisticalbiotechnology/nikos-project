import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.stats import ttest_ind
from numpy.linalg import svd
from qvalues import qvalues
METHYLATION_FILE = './data/GSE76399_data_with_probe_ann.txt'
PATIENT_FILE = './data/samples_clinical_data.txt'

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
    gene_SAT_methylation_data = SAT_methylation_data[SAT_methylation_data['UCSC_RefGene_Accession'].notnull()]
    # Converting genes to a set instead of semi-colon separated strings
    gene_SAT_methylation_data['UCSC_RefGene_Accession'] = gene_SAT_methylation_data['UCSC_RefGene_Accession'].str.split(';').apply(set)
    return gene_SAT_methylation_data, SAT_patient_data

def buildGeneSet(clean_methylation_data):
    '''Takes clean methylation data, with gene accessions stored in sets, and returns the set of all genes.'''
    gene_set = set()
    for i in gene_SAT_methylation_data['UCSC_RefGene_Accession']:
        gene_set |= i # Union of gene_set and i
    return gene_set

def buildMatrix(clean_methylation_data, SAT_geo_accessions, gene_accession):
    '''Picks out all probes and samples for a given gene.
    Returns a numpy array of probes and samples.'''
    # Picking rows
    rows = clean_methylation_data['UCSC_RefGene_Accession'].map(lambda probe_gene_set: gene_accession in probe_gene_set)
    df = clean_methylation_data[rows]
    probes = df['Name']
    # Picking columns
    df = df[SAT_geo_accession]
    # Converts to numpy
    df = df.values
    return df, probes

def rebuildDataFrame(matrix, SAT_geo_accessions,probes):
    '''Rebuilds a dataframe from a matrix.'''
    df = pd.DataFrame(matrix,colums=SAT_geo_accessions)
    probedf = pd.DataFrame(probes,columns=['probe_name'])
    df=df.join(probedf)
    return df

def ttestDataframe(dataframe, insulin_geo_dict):
    '''Performs a t-test on the probes in the dataframe. Returns a dataframe with p-values joined on.'''
    # Splits dataframe by group
    group_dict = {x:dataframe[insuling_geo_dict[x]] for x in ['resistant','sensitive']}

    # Runs ttest
    test_res = ttest_ind(group_dict['resistant'].values,group_dict['sensitive'],axis=1)

    # Save and return results
    ttestframe = pd.DataFrame(test_res[1],columns=['p_value'])
    dataframe = dataframe.join(ttestframe)
    return dataframe

def denoiseMatrixWithSVD(matrix):
    u,s,v = svd(matrix)
    u1 = u[:,0].reshape(-1,1)
    v1 = v[0,:].reshape(1,-1)
    rank1matrix = u1 @ v1 * s[0]
    return rank1matrix


def main():
    # Reading data from file
    methylation_data = csvToDataFrame(METHYLATION_FILE)
    patient_data = csvToDataFrame(PATIENT_FILE)

    SAT_methylation_data, SAT_patient_data = cleanData(methylation_data, patient_data)
    SAT_geo_accession = SAT_patient_data['GEO_accession']
    insulin_geo_dict = {x:SAT_patient_data[SAT_patient_data['Insulin_state']==x]['GEO_accession'] for x in ['resistant','sensitive']}

    gene_set = buildGeneSet(SAT_methylation_data)
    results_df = pd.DataFrame(columns=['UCSC_RefGene_Accession','probe_name','p_value'])

    for gene in gene_set:
        matrix,probes = buildMatrix(SAT_methylation_data,SAT_geo_accession,gene)
        matrix = denoiseMatrixWithSVD(matrix)
        df = rebuildDataFrame(matrix,SAT_geo_accession,probes)
        df = ttestDataframe(df,insulin_geo_dict)
        gene_result_df = df[['probe_name','p_value']]
        gene_result_df['UCSC_RefGene_Accession']=gene
        results_df.append(gene_results_df)

    # Formatting as list of tuples to pass to qvalues function
    ptuples = [(x[0],(x[1],x[2]) for x in results_df[['p_values','UCSC_RefGene_Accession','probe_name']].values]
    qtuple = qvalues(ptuples)
    new_results_df = pd.DataFrame([(q,p,ident[0],ident[1]) for q,p,ident in qtuple], columns=['q_value','p_value','UCSC_RefGene_Accession','probe_name'])

    #Displaying the results
    #Consider having other options for saving them
    print(new_results_df)






