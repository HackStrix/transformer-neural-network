# Neural Machine Translation Project

This project implements a Neural Machine Translation (NMT) system using PyTorch. The system is designed to translate text from English to Hindi using a Transformer model.

## Project Structure

## Files and Directories

- `dev_test/`: Contains development and test datasets.
- `env/`: Python virtual environment directory.
- `getModel.py`: Contains the `get_model` function and `TransformerModel` class.
- `model.py`: Defines the neural network architecture including `FeedforwardNeuralNetModel`, `Encoder`, `Decoder`, and `Transformer`.
- `process.py`: Contains data processing functions including `create_dataset`.
- `train.py`: Main training script that defines the `CustomDataset` class and training loop.
- `train_old.py`: Previous version of the training script.

## Setup

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv env
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        .\env\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source env/bin/activate
        ```

4. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the data:**
    Ensure that your data files are placed in the `dev_test/` directory.

2. **Train the model:**
    Run the training script:
    ```sh
    python train.py
    ```

## Key Components

### Model Architecture

The model architecture is defined in [model.py](model.py). It includes:
- `FeedforwardNeuralNetModel`
- `Encoder`
- `Decoder`
- `Transformer`

### Data Processing

Data processing functions are defined in [process.py](process.py). The `create_dataset` function is used to prepare the dataset for training.

### Training

The main training script is [train.py](train.py). It defines the `CustomDataset` class and the training loop.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.