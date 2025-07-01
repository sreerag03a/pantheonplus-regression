import os
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

dwnld_file = {
    "Pantheon+SH0ES.dat" : "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
    "DES-data.csv" : "https://raw.githubusercontent.com/des-science/DES-SN5YR/main/4_DISTANCES_COVMAT/DES-SN5YR_HD%2BMetaData.csv"

}

for filename, url in dwnld_file.items():
    save_path = os.path.join(DATA_DIR, filename)

    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved to {save_path}")
    except Exception as e:
        print(f"Failed to download {filename}:\n{e}")