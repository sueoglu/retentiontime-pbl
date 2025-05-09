from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import re

data = pd.read_csv('dataset_paper.csv')
data['sequence'] = data['sequence'].str.strip('_')

def remove_modifications(sequence):
    return re.sub(r'\(.*?\)', '', sequence)


data['sequence'] = data['sequence'].apply(remove_modifications)
print("Modifications removed.")

sequence_data_list = []
molecular_weight_list = []
instability_index_list = []
isoelectric_point_list = []
sequence_length_list = []

for sequence in data['sequence']:
    try:
        sequence_data = ProteinAnalysis(sequence)
        sequence_data_list.append(sequence_data)

        molecular_weight = sequence_data.molecular_weight()
        molecular_weight_list.append(molecular_weight)

        instability_index = sequence_data.instability_index()
        instability_index_list.append(instability_index)

        isoelectric_point = sequence_data.isoelectric_point()
        isoelectric_point_list.append(isoelectric_point)

        sequence_length = len(sequence)
        sequence_length_list.append(sequence_length)

    except KeyError as e:
        print(f"Error processing {sequence}: {e}")
        molecular_weight_list.append(None)
        instability_index_list.append(None)
        isoelectric_point_list.append(None)
        sequence_length_list.append(None)

data['molecular_weight'] = molecular_weight_list
data['instability_index'] = instability_index_list
data['isoelectric_point'] = isoelectric_point_list
data['sequence_length'] = sequence_length_list

data.to_csv('dataset_paper_features', index=False)
print("Data saved")
print(data.head())
