import pandas as pd

class CleanFile:
    def __init__(self, file_path, header=0):
        # Load CSV with specified header row
        self.df = pd.read_csv(file_path, encoding='utf-8-sig', engine='python', header=header)
        self.df.reset_index(drop=True, inplace=True)

        self.df.columns = self.df.columns.str.lower().str.strip()
        self.df = self.df.drop(index=range(1,13)).reset_index(drop=True)
        self.df = self.df.drop(columns=['period name', 'main account code', 'fund', 'budget line item', 'bali', 'ba bsa bli', 'limit'])
        self.df = self.df.drop(index=[0]).reset_index(drop=True)
        # Remove Unnamed columns (often leftover from extra headers)

        self.df.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_clean_cluster.csv', index=False)
        print("File loaded and cleaned:")
        print(self.df.head(20).to_string())

if __name__ == "__main__":
    cf = CleanFile('c:/Users/anton/OneDrive/gov_finance/gov_load_file.csv', header=3)

