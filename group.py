import pandas as pd

data = pd.read_csv('/Users/oykusuoglu/PythonProjects/RetentionTimePBL/.venv/peptides_no_duplicates.csv')

grouped_data = data.groupby('sequence').agg({
    'retention_time' : 'mean',
    'is_modified' : 'first',
    'hydrophobicity' : 'first',
    'aromaticity': 'first',
    'cystein_count' : 'first',
    'sequence_length' : 'first',
    'gravy' : 'first',
    'aliphatic_index' : 'first',
    'net_charge_at_pH_7.0' : 'first',
    'count_hydrophobic' : 'first',
    'fraction_hydrophobic' : 'first',
    'count_aromatic' : 'first',
    'fraction_aromatic' : 'first',
    'fraction_ILVFW' : 'first',
    'fraction_DEKR' : 'first',
    'fraction_STY' : 'first',
    'cysteine_count' : 'first',
    'cysteine_count_x_GRAVY' : 'first',
    'sequence_length_x_aromaticity' : 'first'
}).reset_index()

print(grouped_data.head())

grouped_data.to_csv('peptides_no_duplicates_grouped.csv', index=False)
print("Grouped data saved")
