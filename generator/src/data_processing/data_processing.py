import src.utils.utils as utils

if __name__ == "__main__":
    df = utils.unpack_and_read('../../data/single-cell-tracks_exp1-6_noErbB2.csv.gz')
    print(df['Image_Metadata_T'].max())
