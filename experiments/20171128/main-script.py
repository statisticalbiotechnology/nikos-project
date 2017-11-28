import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
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
    df = clean_methylation_data[clean_methylation_data['UCSC_RefGene_Accession'].map(lambda probe_gene_set: gene_accession in probe_gene_set)]
    # Picking columns
    df = df[SAT_geo_accession]
    # Converts to numpy
    df = df.values
    return df

def main():
    # Reading data from file
    methylation_data = csvToDataFrame(METHYLATION_FILE)
    patient_data = csvToDataFrame(PATIENT_FILE)

    SAT_methylation_data, SAT_patient_data = cleanData(methylation_data, patient_data)
    SAT_geo_accession = SAT_patient_data['GEO_accession']
    gene_set = buildGeneSet(SAT_methylation_data)


