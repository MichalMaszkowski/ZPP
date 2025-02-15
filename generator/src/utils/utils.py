import gzip
import pandas as pd
import gdown


def unpack_and_read(f_p=None) -> pd.DataFrame:
    """
    Unpacks and reads a gzipped CSV file into a Pandas DataFrame.

    Args:
    - f_p (str): The file path to the gzipped CSV file. Defaults to a sample file. If None, downloads the sample file.

    Returns:
    - pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    if f_p is None:
        file_id = "1xsDZF8PZUNNzYAVrfuNxGoLMhoEFgAuN"
        url = f"https://drive.google.com/uc?id={file_id}"

        f_p = "data.csv.gz"
        gdown.download(url, f_p, quiet=False)

    with gzip.open(f_p, 'rt') as f:
        df = pd.read_csv(f)
    return df
