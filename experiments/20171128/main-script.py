import sys
import numpy as np
#np.seterr(all='raise')
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.stats import ttest_ind
from numpy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
from qvalues import qvalues
from multiprocessing import Pool
from wpca import WPCA

#CHECK_FOR_OUTLIERS = False
METHYLATION_FILE = '../../data/GSE76399_data_with_probe_ann.txt'
#METHYLATION_FILE = '../../data/NM_178822_data.txt'
#METHYLATION_FILE = '../../data/test_data.txt'
#METHYLATION_FILE = '../../data/test_30k.txt'
PATIENT_FILE = '../../data/samples_clinical_data.txt'

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

def rebuildDataFrame(matrix, SAT_geo_accessions,probes):
    '''Rebuilds a dataframe from a matrix.'''
    df = pd.DataFrame(matrix,columns=SAT_geo_accessions)
    df['probe_name'] = probes.values
    return df

def ttestDataframe(dataframe, insulin_geo_dict):
    '''Performs a t-test on the probes in the dataframe. Returns a dataframe with p-values joined on.'''
    # Splits dataframe by group
    group_dict = {x:dataframe[insulin_geo_dict[x]] for x in ['resistant','sensitive']}

    # Runs ttest
    test_res = ttest_ind(group_dict['resistant'].values,group_dict['sensitive'].values,axis=1)

    # Save and return results
    ttestframe = pd.DataFrame(test_res[1],columns=['p_value'])
    dataframe = dataframe.join(ttestframe)
    return dataframe

def denoiseMatrixWithSVD(matrix):
    '''Run SVD and build matrix for first singular value.'''
    u,s,v = svd(matrix)
    u1 = u[:,0].reshape(-1,1)
    v1 = v[0,:].reshape(1,-1)
    rank1matrix = u1 @ v1 * s[0]
    return rank1matrix

def betaToM(beta):
    '''Converts a beta-value to a M-value.'''
    return np.log2(beta/(1-beta))

def MToBeta(M):
    '''Converts a M-value to a beta-value.'''
    return 2**M/(2**M + 1)

def columnsFromBetaToM(df, SAT_geo_accession):
    '''Turns the columns containing beta values into columns containing M-values.'''
    df.loc[:,SAT_geo_accession] = df.loc[:,SAT_geo_accession].applymap(betaToM)
    return df

def normalizeBeta(beta):
    '''Ensures that beta is in the range (0,1).'''
    e = 1e-9
    return np.minimum(np.maximum(beta,e),1-e)

def normalizeBetaColumns(df, SAT_geo_accession):
    '''Ensures that all beta-values are in range (0,1).'''
    df.loc[:,SAT_geo_accession] = df.loc[:,SAT_geo_accession].applymap(normalizeBeta)
    return df

def calcEigenSample(matrix):
    '''Calculate the first eigensample of the probe-sample matrix.'''
    svd = TruncatedSVD(1)
    svd.fit(matrix)
    return svd.components_

def dfFromEigenSample(matrix,SAT_geo_accession):
    '''Build dataframe from the eigensample.'''
    df = pd.DataFrame(matrix,columns=SAT_geo_accession)
    return df

def checkForOutliers(df,SAT_geo_accession,insulin_geo_dict):
    '''Print the sum of the eigensamples. Also display heatmap of the eigensamples.'''
    df[insulin_geo_dict['resistant'].append(insulin_geo_dict['sensitive'])].sum().to_csv(sys.stdout,sep='\t')
    showOutliers(df,SAT_geo_accession,insulin_geo_dict)

def showOutliers(df,SAT_geo_accession,insulin_geo_dict):
    ax=sns.heatmap(df.sort_values(by=['p_value'])[insulin_geo_dict['resistant'].append(insulin_geo_dict['sensitive'])],xticklabels=1,yticklabels=False)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.show()

def showInputData(df,SAT_geo_accession,insulin_geo_dict):
    data = df[insulin_geo_dict['resistant'].append(insulin_geo_dict['sensitive'])]
    ax=sns.heatmap(data.sort_values(by=['GSM1986071']),xticklabels=1,yticklabels=False)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.show()

def normalizeColumns(df,SAT_geo_accession):
    '''Subtract mean and divide by std dev for all columns.'''
    df.loc[:,SAT_geo_accession] = df.loc[:,SAT_geo_accession].subtract(df.loc[:,SAT_geo_accession].mean(axis=0))
    df.loc[:,SAT_geo_accession] = df.loc[:,SAT_geo_accession].divide(df.loc[:,SAT_geo_accession].std(axis=0))
    return df

def replaceMissingDataPoint(beta, eps = 0):
    '''Replace 'bad' values with NaN.'''
    if (beta <= eps) or (beta >= (1 - eps)):
        return np.nan
    return beta

def handleMissingData(df,SAT_geo_accession):
    '''Replace 'bad' values with NaN in all columns.'''
    df.loc[:,SAT_geo_accession] = df.loc[:,SAT_geo_accession].applymap(replaceMissingDataPoint)
    return df

def eigensampleFromWPCA(matrix):
    '''Find a representation of each sample using wPCA to exclude NaNs.'''
    weights = 1.0 - np.isnan(matrix.T)
    pc = WPCA(n_components=1).fit_reconstruct(matrix.T,weights=weights)
    return pc.T


def dropBadRows(df,SAT_geo_accession,eps=0.05):
    '''Drop all rows that contain values outside of (eps,1-eps) range.'''
    idxs = df[((df.loc[:,SAT_geo_accession] < eps) | (df.loc[:,SAT_geo_accession] >(1- eps))).any(1)].index
    df = df.drop(idxs)
    return df

    



def main():
    # Reading data from file
    methylation_data = csvToDataFrame(METHYLATION_FILE)
    patient_data = csvToDataFrame(PATIENT_FILE)

    SAT_methylation_data, SAT_patient_data = cleanData(methylation_data, patient_data)

    # Picks out SAT samples (columns)
    SAT_geo_accession = SAT_patient_data['GEO_accession']
    insulin_geo_dict = {x:SAT_patient_data[SAT_patient_data['Insulin_state']==x]['GEO_accession'] for x in ['resistant','sensitive']}

    ## Replace all elements smaller than eps or larger than 1 - eps with np.nan
    #SAT_methylation_data = handleMissingData(SAT_methylation_data,SAT_geo_accession)
    ## Plot input data
    #showInputData(SAT_methylation_data,SAT_geo_accession,insulin_geo_dict)
    # Drop all rows with elements smaller than eps or larger than 1 - eps
    SAT_methylation_data = dropBadRows(SAT_methylation_data,SAT_geo_accession)
    # Convert beta values to M values
    SAT_methylation_data = columnsFromBetaToM(SAT_methylation_data,SAT_geo_accession)
    # Normalize columns
    #SAT_methylation_data = normalizeColumns(SAT_methylation_data,SAT_geo_accession)

    gene_set = buildGeneSet(SAT_methylation_data)
    gene_set = {'NM_018284'}
    num_genes = len(gene_set)


    results_df = pd.DataFrame(columns=['UCSC_RefGene_Accession','p_value'])

    for igene, gene in enumerate(gene_set):
        matrix,probes = buildMatrix(SAT_methylation_data,SAT_geo_accession,gene)
        if matrix is None:
            print('Too few probes found for gene {0} ({1}/{2})'.format(gene,igene+1,num_genes),file=sys.stderr)
            continue
        print('Working on gene {0} ({1}/{2})'.format(gene,igene+1,num_genes),file=sys.stderr)
        eigensample = calcEigenSample(matrix)
#        eigensample = eigensampleFromWPCA(matrix)
        df = dfFromEigenSample(eigensample,SAT_geo_accession)
        df.to_csv(sys.stdout,sep='\t',index=False)
        return 0
        df = ttestDataframe(df,insulin_geo_dict)
        df['UCSC_RefGene_Accession'] = gene
        results_df = results_df.append(df)


    #if CHECK_FOR_OUTLIERS:
    #    # Checking for outliers
    #    checkForOutliers(results_df,SAT_geo_accession,insulin_geo_dict)
    #else:
    # Formatting as list of tuples to pass to qvalues function
    ptuples = [(x[0],x[1]) for x in results_df[['p_value','UCSC_RefGene_Accession']].values]
    qtuple = qvalues(ptuples)
    new_results_df = pd.DataFrame([(q,p,ident) for q,p,ident in qtuple], columns=['q_value','p_value','UCSC_RefGene_Accession'])

    #Displaying the results
    #Consider having other options for saving them
    new_results_df.to_csv(sys.stdout,sep='\t',index=False)
    showOutliers(results_df,SAT_geo_accession,insulin_geo_dict)



if __name__=='__main__':
    main()



