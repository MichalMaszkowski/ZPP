import os
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from clearml import Task
from src.trainer.trainer import Trainer
import src.model.model as model
import src.data_processing.data_processing as data_processing

# Set up ClearML
@pytest.fixture
def setup_clearml():
    access_key = os.getenv('CLEARML_ACCESS_KEY')
    secret_key = os.getenv('CLEARML_SECRET_KEY')

    Task.set_credentials(
        web_host='https://app.clear.ml',
        api_host='https://api.clear.ml',
        files_host='https://files.clear.ml',
        key=access_key,
        secret=secret_key
    )

# Data generation (updated with hidden_dim support)
def generate_recursive(n_first, vocab_size, next_prob):
    assert 0 < vocab_size
    initial = np.random.randint(0, vocab_size, n_first)
    coeffs = np.random.randint(0, vocab_size, n_first)

    return initial, coeffs, vocab_size, next_prob

class SeqGen:
    """
    For generating recurrent sequences with stochastically repeating terms.
    """
    def __init__(self, initial, coeffs, size, next_prob):
        assert len(coeffs) == len(initial)
        self.initial = initial
        self.coeffs = coeffs
        self.size = size
        self.next_prob = next_prob
        self.current = initial

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.random() < self.next_prob:
          new = self.current[-1] + 1
        else:
          new = (self.current @ self.coeffs)

        new %= self.size
        self.current = np.append(self.current, new)[1:]

        return new

# Generating artificial datasets for training/evaluating
def generate_dataset(gen_factory, seq_len, num_entries, dim, exclude=[], correlation_factor=0.0):
    """
    For generating datasets with num_entries elements each of length seq_len.

    gen_factory is a procedure that returns an instance of SeqGen when called.

    seq_len is the length of the sequence to generate.

    num_entries is the number of sequences to generate.

    exclude is the set of sequences that aren't to be used in training
    """
    entries = []
    generators = []

    for e in range(num_entries):
        # Generate dim separate sequences for each entry
        seq = []
        for i in range(dim):
            while True:
                seq_gen = gen_factory()  # Generate a new sequence generator
                if seq_gen in exclude:
                    continue  # Skip if the generator is already in the exclude set
                
                dim_seq = []
                for s in range(seq_len):
                    val = next(seq_gen)  # Generate the next value for this dimension
                    # Optionally correlate with the previous dimensions
                    if i > 0:
                        val += correlation_factor * next(generators[i-1])  # Introduce correlation
                    dim_seq.append(val)
                
                # Append the generated sequence for the dimension
                seq.append(dim_seq)
                generators.append(seq_gen)  # Keep track of the generator
                break  # Exit the while loop once a valid sequence is generated

        # Stack all dim sequences (each of shape seq_len) to form a seq_len x dim tensor
        final_seq = torch.tensor(seq).T  # Transpose to get shape (seq_len, dim)
        entries.append(final_seq)  # Add to the list of entries

    data = torch.stack(entries)  # Stack all sequences (batch_size, seq_len, dim)

    # Split the data into input (x) and target (y)
    x = data[:, :-1]  # Input
    y = data[:, 1:]   # Target
    
    return torch.utils.data.TensorDataset(x, y), set(generators)


# Test with different hyperparameters (ModelArgs values)
@pytest.mark.parametrize("dim, n_layers, n_heads, lr, batch_size, n_epochs", [
    (256, 128, 8, 2e-4, 32, 10),
    (128, 64, 4, 1e-3, 64, 5),
    (512, 256, 16, 1e-5, 128, 20),
])

def test_train_with_different_hyperparameters(dim, n_layers, n_heads, lr, batch_size, n_epochs, setup_clearml):
    train_loader, test_loader = setup_data(dim, batch_size)
    
    # Set up the model args
    args = model.ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads)
    model_instance = model.Transformer(args)

    # Set up the trainer with the current hyperparameters
    trainer = Trainer(lr=lr, n_epochs=n_epochs)
    
    # Create a new ClearML task for each run
    params = {
        'dim': dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'lr': lr,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
    }
    
    task = Task.init(
        project_name='Test Transformer',
        task_name=f'Run {n_epochs} - ' + ', '.join([f'{key}: {value}' for key, value in params.items()]),
        task_type=Task.TaskTypes.optimizer
    )
    
    # Log hyperparameters for this task
    task.connect(params)
    logger = task.get_logger()

    # Train the model
    trainer.train(model_instance, train_loader, test_loader, logger=logger)
    
    # Optionally - assert training progress
    assert task.state == 'completed'


def example_generator(initial, vocab_size, next_prob):
    """
    A procedure that returns a generator function
    for generating sequences based on specific parameters.
    """
    def example_gen():
        return SeqGen(*generate_recursive(initial, vocab_size, next_prob))
    return example_gen

# Initialize the data and model
def setup_data(dim, batch_size):
    # Generate artificial data based on the `dim` argument
    seq_len = 64  # Define sequence length
    vocab_size = 7
    next_prob = 0.1
    initial = 2

    perm_example_generator = example_generator(initial, vocab_size, next_prob)

    test_dataset, generators = generate_dataset(
        gen_factory=perm_example_generator, seq_len=seq_len, num_entries=1000, dim=dim)
    train_dataset, _ = generate_dataset(
        gen_factory=perm_example_generator, seq_len=seq_len, num_entries=10000, dim=dim, exclude=generators)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


# Test ClearML integration
def test_clearml_integration(setup_clearml):
    task = Task.init(project_name='TestProject', task_name='TestTask')
    
    # Check if the task is initialized correctly
    assert task.id is not None, "ClearML task was not initialized properly"


# Test Data Generation
def test_data_generation():
    
    perm_example_generator = example_generator(initial=1, vocab_size=3, next_prob=0.5)
    # Generate dataset
    gen_dataset, _ = generate_dataset(
            gen_factory=perm_example_generator, seq_len=30, num_entries=10, dim=10)
    
    # Verify the shape of the data
    assert gen_dataset[0].shape == (30, 10), f"Expected shape (seq_len, dim), got {gen_dataset[0].shape}"
    
    # Print the first sequence to verify the content
    print(f"First generated sequence: {gen_dataset[0]}")
    print(f"Second generated sequence: {gen_dataset[1]}")


# Run the tests
if __name__ == "__main__":
    pytest.main()