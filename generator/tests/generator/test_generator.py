from src.generator.generator import Generator
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

@pytest.fixture
def sample_generator():
    """Fixture creating an instance of the Generator class with a sample DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame({
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 10),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 10),
        'other_column': np.random.randint(0, 5, 10)
    })
    return Generator(df)  # Tworzymy instancję klasy z przekazanym DataFrame

@pytest.fixture
def sample_generator_100_records():
    """Fixture creating an instance of the Generator class with a sample DataFrame holding 100 records."""
    np.random.seed(42)
    df = pd.DataFrame({
        'objNuclei_Location_Center_X': np.random.uniform(0, 256, 100),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 256, 100),
        'other_column': np.random.randint(0, 5, 100)
    })
    return Generator(df)  # Tworzymy instancję klasy z przekazanym DataFrame

@pytest.fixture
def sample_generator_extra_columns():
    """Fixture tworząca instancję klasy Generator z przykładowym DataFrame, zawierającym dodatkowe nieistotne kolumny."""
    np.random.seed(42)
    df = pd.DataFrame({
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 10),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 10),
        'other_column': np.random.randint(0, 5, 10),
        'irrelevant_column_1': np.random.choice(['A', 'B', 'C'], 10),  # Kolumna tekstowa
        'irrelevant_column_2': np.random.uniform(0, 1, 10),  # Kolumna numeryczna
        'irrelevant_column_3': np.random.randint(100, 200, 10)  # Kolejna liczba całkowita
    })
    return Generator(df)  # Tworzymy instancję klasy z rozszerzonym DataFrame

@pytest.fixture
def sample_data():
    """Fixture tworzący przykładowy DataFrame dla testów."""
    return pd.DataFrame({
        'track_id': np.arange(5),
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 5),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 5),
        'ERKKTR_ratio': np.random.uniform(0.5, 2.5, 5)
    })

@pytest.fixture
def adjacency_matrix():
    """Fixture dla przykładowej macierzy sąsiedztwa."""
    return torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]
    ], dtype=torch.float32, device=DEVICE)

@pytest.fixture
def disconnected_adjacency():
    """Macierz sąsiedztwa, w której żadne jądra nie są połączone."""
    return torch.zeros((5, 5), dtype=torch.float32, device=DEVICE)

@pytest.fixture
def sample_points():
    """Fixture tworzący przykładowy DataFrame z pozycjami jąder."""
    return pd.DataFrame({
        'track_id': np.arange(5),
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 5),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 5),
        'objNuclei_Location_Center_Z': np.random.uniform(0, 50, 5)
    })

@pytest.fixture
def simple_case():
    """Fixture dla prostych danych, w których sąsiedztwo jest oczywiste."""
    return pd.DataFrame({
        'track_id': [1, 1, 1, 1, 1, 1],
        'objNuclei_Location_Center_X': [0, 1, 2, 3, 5, 7],  # Punkty leżące na linii
        'objNuclei_Location_Center_Y': [0, 0, 0, 0, 0, 0]
    })

def test_return_type(sample_generator):
    """Testuje, czy funkcja zwraca obiekt typu `pd.DataFrame`."""
    result = sample_generator.generate_next_move(sample_generator.df_first_frame.copy())
    assert isinstance(result, pd.DataFrame), "Funkcja nie zwraca DataFrame"

def test_dataframe_shape(sample_generator):
    """Sprawdza, czy liczba wierszy i kolumn nie zmienia się po wywołaniu funkcji."""
    original_shape = sample_generator.df_first_frame.shape
    result = sample_generator.generate_next_move(sample_generator.df_first_frame.copy())
    assert result.shape == original_shape, "Zmieniła się liczba wierszy lub kolumn"

def test_x_y_columns_change(sample_generator):
    """Sprawdza, czy wartości w kolumnach X i Y zmieniają się po wywołaniu funkcji."""
    original_x = sample_generator.df_first_frame['objNuclei_Location_Center_X'].copy()
    original_y = sample_generator.df_first_frame['objNuclei_Location_Center_Y'].copy()

    result = sample_generator.generate_next_move(sample_generator.df_first_frame.copy())

    assert not np.allclose(original_x, result['objNuclei_Location_Center_X']), "Kolumna X nie zmieniła wartości"
    assert not np.allclose(original_y, result['objNuclei_Location_Center_Y']), "Kolumna Y nie zmieniła wartości"

def test_other_columns_unchanged(sample_generator_extra_columns):
    """Sprawdza, czy inne kolumny nie są zmieniane przez funkcję."""

    irrelevant_column_1 = sample_generator_extra_columns.df_first_frame['irrelevant_column_1'].copy()
    irrelevant_column_2 = sample_generator_extra_columns.df_first_frame['irrelevant_column_2'].copy()
    irrelevant_column_3 = sample_generator_extra_columns.df_first_frame['irrelevant_column_3'].copy()
    
    result = sample_generator_extra_columns.generate_next_move(sample_generator_extra_columns.df_first_frame.copy())

    assert np.array_equal(irrelevant_column_1, result['irrelevant_column_1']), "Irrelevan column_1 has changed!"
    assert np.array_equal(irrelevant_column_2, result['irrelevant_column_2']), "Irrelevan column_2 has changed!"
    assert np.array_equal(irrelevant_column_3, result['irrelevant_column_3']), "Irrelevan column_3 has changed!"

def test_random_distribution(sample_generator_100_records):
    """Sprawdza, czy różnice w X i Y są zgodne z rozkładem normalnym."""
    result = sample_generator_100_records.generate_next_move(sample_generator_100_records.df_first_frame.copy())

    diffs_x = result['objNuclei_Location_Center_X'] - sample_generator_100_records.df_first_frame['objNuclei_Location_Center_X']
    diffs_y = result['objNuclei_Location_Center_Y'] - sample_generator_100_records.df_first_frame['objNuclei_Location_Center_Y']

    # Średnia powinna być w okolicach 0, a odchylenie bliskie 0.43
    print(f"gówno srednie x {np.mean(diffs_x)}")
    print(f"gówno srednie y {np.mean(diffs_y)}")
    print(f"gówno std x {np.std(diffs_x)}")
    print(f"gówno std y {np.std(diffs_y)}")
    assert np.isclose(np.mean(diffs_x), 0, atol=0.1), "Średnia różnic w X odbiega od oczekiwanej"
    assert np.isclose(np.mean(diffs_y), 0, atol=0.1), "Średnia różnic w Y odbiega od oczekiwanej"
    assert np.isclose(np.std(diffs_x), 0.43, atol=0.1), "Odchylenie w X odbiega od oczekiwanej wartości"
    assert np.isclose(np.std(diffs_y), 0.43, atol=0.1), "Odchylenie w Y odbiega od oczekiwanej wartości"

def test_output_dataframe_shape(adjacency_matrix, sample_data, sample_generator):
    """Sprawdza, czy liczba wierszy w zwróconym DataFrame się zgadza.""" # Zamień na właściwą klasę
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=1)
    
    assert isinstance(result, pd.DataFrame), "Funkcja powinna zwracać DataFrame"
    assert result.shape[0] == sample_data.shape[0], "Liczba wierszy w DataFrame powinna pozostać taka sama"

def test_ERKKTR_ratio_range(sample_data, adjacency_matrix, sample_generator):
    """Sprawdza, czy wartości ERKKTR_ratio są ograniczone do zakresu [0.4, 2.7]."""
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=1)
    
    assert result['ERKKTR_ratio'].between(0.4, 2.7).all(), "Wartości ERKKTR_ratio powinny być ograniczone do [0.4, 2.7]"

def test_track_id_sorted(sample_data, adjacency_matrix, sample_generator):
    """Sprawdza, czy wyniki są posortowane według track_id."""
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=1)
    
    assert (result['track_id'].values == sorted(result['track_id'].values)).all(), "track_id powinien być posortowany"

def test_ERKKTR_ratio_changes(sample_data, adjacency_matrix, sample_generator):
    """Sprawdza, czy wartości ERKKTR_ratio zmieniają się po zastosowaniu funkcji."""
    original_values = sample_data['ERKKTR_ratio'].copy()
    result = sample_generator.generate_next_ERK(sample_data, adjacency_matrix, T=1)

    assert not np.allclose(original_values, result['ERKKTR_ratio']), "ERKKTR_ratio powinno ulec zmianie"

def test_disconnected_graph(sample_data, disconnected_adjacency, sample_generator):
    """Sprawdza, czy wartości ERKKTR_ratio nie zmieniają się, gdy nie ma sąsiadów."""
    original_values = sample_data['ERKKTR_ratio'].copy()
    result = sample_generator.generate_next_ERK(sample_data, disconnected_adjacency, T=1)

    assert np.allclose(original_values, result['ERKKTR_ratio'], atol=0.1), "ERKKTR_ratio nie powinno się znacząco zmieniać"

def test_output_tensor_shape(sample_points, sample_generator):
    """Sprawdza, czy zwracana macierz sąsiedztwa ma poprawne wymiary."""
    result = sample_generator.calculate_neighbors(sample_points)

    assert isinstance(result, torch.Tensor), "Funkcja powinna zwracać torch.Tensor"
    assert result.shape == (len(sample_points), len(sample_points)), "Macierz sąsiedztwa powinna mieć rozmiar NxN"

def test_adjacency_symmetry(sample_points, sample_generator):
    """Sprawdza, czy macierz sąsiedztwa jest symetryczna."""
    adjacency_matrix = sample_generator.calculate_neighbors(sample_points)

    assert torch.equal(adjacency_matrix, adjacency_matrix.T), "Macierz powinna być symetryczna"

def test_no_self_connections(sample_points, sample_generator):
    """Sprawdza, czy macierz sąsiedztwa nie ma połączeń własnych."""
    adjacency_matrix = sample_generator.calculate_neighbors(sample_points)

    assert torch.all(torch.diag(adjacency_matrix) == 0), "Nie powinno być połączeń własnych"

# TODO
def test_simple_neighbors(simple_case, sample_generator):
    """Sprawdza, czy sąsiedztwo dla uporządkowanych punktów jest poprawne."""
    adjacency_matrix = sample_generator.calculate_neighbors(simple_case)

    print(f" gówno  matrix: {adj_matrix}")
    expected = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.uint8, device=DEVICE)

    assert torch.equal(adjacency_matrix, expected), "Macierz sąsiedztwa dla prostego przypadku powinna być poprawna"


@pytest.mark.parametrize("num_points", [10, 50, 100])
def test_various_sizes(num_points, sample_generator):
    """Sprawdza działanie funkcji dla różnych liczby punktów."""
    np.random.seed(42)
    data = pd.DataFrame({
        'track_id': np.arange(num_points),
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, num_points),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, num_points)
        # 'objNuclei_Location_Center_Z': np.random.uniform(0, 50, num_points)
    })

    adjacency_matrix = sample_generator.calculate_neighbors(data)

    assert adjacency_matrix.shape == (num_points, num_points), f"Zły rozmiar macierzy dla {num_points} punktów"


@pytest.fixture
def sample_initial_frame():
    """Fixture: przykładowa pierwsza klatka symulacji"""
    return pd.DataFrame({
        'track_id': np.arange(5),
        'objNuclei_Location_Center_X': np.random.uniform(0, 100, 5),
        'objNuclei_Location_Center_Y': np.random.uniform(0, 100, 5),
        'ERKKTR_ratio': np.random.uniform(0.5, 2.5, 5)
    })

def test_generate_video_returns_dataframe(sample_initial_frame, sample_generator):
    """Sprawdza, czy funkcja `generate_video` zwraca DataFrame."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    # Mockowanie funkcji pomocniczych
    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    assert isinstance(result, pd.DataFrame), "Funkcja powinna zwracać `pd.DataFrame`"


def test_generate_video_columns(sample_initial_frame, sample_generator):
    """Sprawdza, czy zwrócony DataFrame zawiera wszystkie wymagane kolumny."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    expected_columns = {'track_id', 'objNuclei_Location_Center_X', 'objNuclei_Location_Center_Y', 
                        'ERKKTR_ratio'}
    assert expected_columns.issubset(result.columns), "Brakuje wymaganych kolumn w wyniku"

def test_generate_video_frame_count(sample_initial_frame, sample_generator):
    """Sprawdza, czy wynikowy DataFrame zawiera odpowiednią liczbę klatek."""

    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    num_tracks = len(sample_initial_frame)
    expected_rows = num_tracks * sample_generator.number_of_frames
    assert len(result) == expected_rows, f"Nieprawidłowa liczba wierszy: {len(result)}, oczekiwano {expected_rows}"

def test_generate_video_track_id_integrity(sample_initial_frame, sample_generator):
    """Sprawdza, czy identyfikatory `track_id` pozostają stałe w czasie."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    grouped = result.groupby("track_id")
    assert all(len(group) == sample_generator.number_of_frames for _, group in grouped), "Nie wszystkie `track_id` mają poprawną liczbę klatek"

# TODO
def test_generate_video_time_metadata(sample_initial_frame, sample_generator):
    """Sprawdza, czy `Image_Metadata_T` jest poprawnie zwiększane w czasie."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = MagicMock(return_value=sample_initial_frame.copy())
    sample_generator.generate_next_move = MagicMock(return_value=sample_initial_frame.copy())

    result = sample_generator.generate_video()

    unique_times = sorted(result["Image_Metadata_T"].unique())
    assert unique_times == list(range(instance.number_of_frames)), "Błędne wartości `Image_Metadata_T`"

def test_generate_video_changes_values(sample_initial_frame, sample_generator):
    """Sprawdza, czy `ERKKTR_ratio` oraz współrzędne X, Y faktycznie się zmieniają w czasie."""
    sample_generator.df_first_frame = sample_initial_frame
    sample_generator.number_of_frames = 10

    def fake_next_move(df):
        df = df.copy()
        df["objNuclei_Location_Center_X"] += np.random.normal(0, 1, len(df))
        df["objNuclei_Location_Center_Y"] += np.random.normal(0, 1, len(df))
        return df

    def fake_next_ERK(df, adj_matrix, T):
        df = df.copy()
        df["ERKKTR_ratio"] += np.random.normal(0, 0.1, len(df))
        return df

    sample_generator.calculate_neighbors = MagicMock(return_value=torch.zeros((5, 5), dtype=torch.uint8, device=DEVICE))
    sample_generator.generate_next_ERK = fake_next_ERK
    sample_generator.generate_next_move = fake_next_move

    result = sample_generator.generate_video()

    assert result["objNuclei_Location_Center_X"].nunique() > len(sample_initial_frame), "Wartości X nie zmieniają się"
    assert result["objNuclei_Location_Center_Y"].nunique() > len(sample_initial_frame), "Wartości Y nie zmieniają się"
    assert result["ERKKTR_ratio"].nunique() > len(sample_initial_frame), "ERK nie zmienia się w czasie"
