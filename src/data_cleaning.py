import pandas as pd
import math

def data_clean(data):
    # Drop null and duplicate values
    data = data.dropna().drop_duplicates()

    # Drop unnamed column
    data = data.drop(columns=['Unnamed: 0'])

    # Clean Ram column
    data['Ram'] = data['Ram'].str.replace('GB', '').astype(int)

    # Simplify CPU column
    data['Cpu'] = data['Cpu'].apply(lambda x: ' '.join(x.split()[:3]))

    # Clean Weight column
    data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)

    # Extract resolution
    data['ScreenResolution'] = data['ScreenResolution'].str.extract(r'(\d+x\d+)')
    data[['Width', 'Height']] = data['ScreenResolution'].str.split('x', expand=True).astype(int)
    data = data.drop(columns=['ScreenResolution'])  # Drop after extracting

    # Calculate DPI
    def calculate_dpi(row):
        pixel_diagonal = math.sqrt(row['Width']**2 + row['Height']**2)
        return pixel_diagonal / row['Inches']
    data['DPI'] = data.apply(calculate_dpi, axis=1)

    # Simplify GPU
    data['Gpu'] = data['Gpu'].apply(lambda x: x.split(' ')[0])

    # Simplify OS types
    data['OpSys'] = data['OpSys'].apply(lambda x: 'Window' if 'Window' in x else ('Mac' if 'mac' in x.lower() else 'Others'))

    # Standardize CPU labels
    cpu_processed = []
    for i in data['Cpu']:
        if 'i3' in i:
            cpu_processed.append('Intel i3')
        elif 'i5' in i:
            cpu_processed.append('Intel i5')
        elif 'i7' in i:
            cpu_processed.append('Intel i7')
        elif 'AMD' in i:
            cpu_processed.append('AMD Processor')
        else:
            cpu_processed.append('Others')
    data['Cpu'] = cpu_processed

    # Drop processed resolution columns
    data = data.drop(columns=['Width', 'Height'])

    # Drop unused column
    data = data.drop(columns=['Memory'])

    return data
