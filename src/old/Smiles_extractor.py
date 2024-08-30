import pandas as pd

def main():
    input_file_path = "fatdata/IC_71772.sdf"
    output_file_path = "test_NC.csv"
    smiles = []
    
    with open(input_file_path, 'r', encoding='latin1') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>  <Smile>'):
                current_smiles = next(file).strip()
                smiles.append(current_smiles)
    
    smiles_df = pd.DataFrame(smiles, columns=['SMILES'])
    smiles_df.to_csv(output_file_path, index=False)
    
    print(f"Extracted {len(smiles)} SMILES strings and saved to {output_file_path}")

if __name__ == "__main__":
    main()