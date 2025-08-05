import numpy as np
import pickle
import time
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from pathlib import Path
from src.model_utils import MolFeatureExtractor
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")  # type: ignore
# Optional Ray import
try:
    import ray

    RAY_AVAILABLE = True
    # remove ray logging

except ImportError:
    RAY_AVAILABLE = False

# Ray remote function for parallel fingerprint generation (must be at module level)
if RAY_AVAILABLE:

    @ray.remote
    def process_smiles_batch(smiles_batch, fp_rad, fp_len, data_path):
        """Process a batch of SMILES in parallel"""
        # Import numpy with threading already configured
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from src.model_utils import MolFeatureExtractor
        from pathlib import Path

        # Initialize generators for this worker
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_rad, fpSize=fp_len, countSimulation=False
        )
        feature_gen = MolFeatureExtractor(Path(data_path))

        results = []
        for smi in smiles_batch:  # Remove tqdm to avoid overhead
            try:
                if not smi:
                    fp = np.zeros(fp_len, dtype=np.float32)
                else:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        fp = np.zeros(fp_len, dtype=np.float32)
                    else:
                        fp = fp_gen.GetFingerprintAsNumPy(mol).astype(np.float32)

                # Add features
                features = feature_gen.encode(smi)
                features = feature_gen.standardise_features(features)
                features = features.reshape(-1, 1).squeeze()
                fp = np.concatenate((fp, features), axis=0).astype(np.float32)
                results.append(fp)
            except:
                # Return zero vector on error
                results.append(np.zeros(fp_len + 10, dtype=np.float32))

        return results


class NumpyFingerprints:
    def __init__(self, weights_path=None, FP_rad=3, FP_len=4096, debug=False):
        """
        Standalone numpy implementation of Fingerprints model.

        Args:
            weights_path: Path to saved weights (.pickle or .pkl)
            FP_rad: Morgan fingerprint radius
            FP_len: Morgan fingerprint length
            debug: Enable debug output
        """
        self.FP_rad = FP_rad
        self.FP_len = FP_len
        self._restored = False
        self.debug = debug

        # Model weights storage
        self.nn_weights = []
        self.nn_biases = []
        self.final_weight = None
        self.final_bias = None

        # Initialize fingerprint generator
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.FP_rad, fpSize=self.FP_len, countSimulation=False
        )
        # Initialize feature extractor
        self.feature_gen = MolFeatureExtractor(
            Path(__file__).parent.parent / "data/features"
        )

        if weights_path:
            self.restore(weights_path)

    def restore(self, weights_path):
        """
        Load model weights from pickle file.

        Args:
            weights_path: Path to weights file (.pickle or .pkl)
        """
        with open(weights_path, "rb") as f:
            weights_dict = pickle.load(f)

        # Extract neural network layers and pre-transpose for efficiency
        layer_indices = [0, 3, 6, 9]  # Based on your model structure
        for idx in layer_indices:
            weight_key = f"neural_network.{idx}.weight"
            bias_key = f"neural_network.{idx}.bias"
            if weight_key in weights_dict and bias_key in weights_dict:
                # Pre-transpose weights to avoid doing it every forward pass
                self.nn_weights.append(weights_dict[weight_key].T)
                self.nn_biases.append(weights_dict[bias_key])

        # Final linear layer (also pre-transpose)
        if "linear.weight" in weights_dict and "linear.bias" in weights_dict:
            self.final_weight = weights_dict["linear.weight"].T
            self.final_bias = weights_dict["linear.bias"]

        self._restored = True
        print(f"Model restored from {weights_path}")
        return self

    def mol_to_fp(self, mol):
        """
        Convert RDKit molecule to Morgan fingerprint using new generator.

        Args:
            mol: RDKit molecule object

        Returns:
            numpy array fingerprint
        """
        if mol is None:
            return np.zeros(self.FP_len, dtype=np.float32)

        # Use the new fingerprint generator to get numpy array directly
        fp = self.fp_gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.float32)

    def smi_to_fp(self, smi):
        """
        Convert SMILES string to Morgan fingerprint.

        Args:
            smi: SMILES string

        Returns:
            numpy array fingerprint
        """
        if not smi:
            return np.zeros(self.FP_len, dtype=np.float32)

        mol = Chem.MolFromSmiles(smi)
        fp = self.mol_to_fp(mol)
        features = self.feature_gen.encode(smi)
        features = self.feature_gen.standardise_features(features)
        features = features.reshape(-1, 1)  # Ensure 2D shape
        features = features.squeeze()
        fp = np.concatenate((fp, features), axis=0)
        return fp.astype(np.float32)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input numpy array (fingerprint or batch of fingerprints)

        Returns:
            output: Final price
            z: Latent representation
        """
        if not self._restored:
            raise ValueError("Must restore model weights first!")

        start_time = time.time()
        z = x

        # Neural network layers
        for i, (weight, bias) in enumerate(zip(self.nn_weights, self.nn_biases)):
            z = z @ weight + bias  # Using @ operator instead of np.dot
            if i < len(self.nn_weights) - 1:  # Apply ReLU except for last layer
                z = self.relu(z)

        # Store intermediate representation
        intermediate = z.copy()

        # Final linear layer
        output = z @ self.final_weight + self.final_bias

        forward_time = time.time() - start_time
        if self.debug:
            print(f"Forward pass time: {forward_time:.4f} seconds")

        return output, intermediate

    def predict_from_smiles(self, smi, return_intermediate=False):
        """
        Get prediction directly from SMILES string.

        Args:
            smi: SMILES string
            return_intermediate: Whether to return intermediate representation

        Returns:
            price of molecule (and intermediate if requested)
        """
        fp = self.smi_to_fp(smi)
        if np.sum(fp) == 0:
            print("Warning: Could not generate fingerprint for SMILES")
            return 0.0 if not return_intermediate else (0.0, np.zeros(10))

        # Add batch dimension if single molecule
        if fp.ndim == 1:
            fp = fp.reshape(1, -1)

        output, intermediate = self.forward(fp)

        # Remove batch dimension if single molecule
        if output.shape[0] == 1:
            output = output.squeeze(0)
            intermediate = intermediate.squeeze(0)

        if return_intermediate:
            return output, intermediate
        return output[0]

    def predict_batch_from_smiles(self, smiles_list, return_intermediate=False):
        """
        Get predictions for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            return_intermediate: Whether to return intermediate representations
            use_ray: Use Ray for parallel processing if batch > 100 (auto-detect if None)

        Returns:
            price of molecule (and intermediates if requested)
        """
        batch_size = len(smiles_list)
        print("-" * 50)
        print("FEATURE GENERATION")
        print("-" * 50)
        if RAY_AVAILABLE and batch_size > 100:
            print("Using Ray for parallel processing...")
            if not ray.is_initialized():
                # disable Ray logging
                ray.init(ignore_reinit_error=True, log_to_driver=False)
            fps = self._parallel_fingerprint_generation(smiles_list)
        else:
            fps = np.array(
                [
                    self.smi_to_fp(smi)
                    for smi in tqdm(smiles_list, desc="Processing SMILES")
                ]
            )

        print("-" * 50)
        print("MODEL PREDICTION")
        print("-" * 50)
        starting_time = time.time()
        # Dynamic chunking based on memory constraints
        max_memory_gb = 8
        fp_size = fps.shape[1] if len(fps) > 0 else self.FP_len + 10
        bytes_per_fp = fp_size * 4 # float 32
        max_batch_size = max(1, int((max_memory_gb * 1024**3) // (bytes_per_fp * 8)))  # Factor of 8 for safety
        
        outputs = []
        intermediates = []
        
        for i in tqdm(range(0, len(fps), max_batch_size), desc="Processing batches"):
            chunk = fps[i:i + max_batch_size]
            chunk_output, chunk_intermediate = self.forward(chunk)
            outputs.append(chunk_output)
            if return_intermediate:
                intermediates.append(chunk_intermediate)
        
        # Concatenate all results
        output = np.concatenate(outputs, axis=0)
        if return_intermediate:
            intermediate = np.concatenate(intermediates, axis=0)
        forward_time = time.time() - starting_time
        print(f"Batch forward pass time: {forward_time:.4f} seconds")
        print(f"Average per molecule: {forward_time*1000 / len(smiles_list):.4f} ms")

        if return_intermediate:
            return output, intermediate
        return output

    def _parallel_fingerprint_generation(self, smiles_list):
        """Generate fingerprints in parallel using Ray"""
        # Split into batches for parallel processing
        num_cpus = int(ray.available_resources().get("CPU", 4))
        batch_size = max(1, len(smiles_list) // (num_cpus * 2))

        batches = [
            smiles_list[i : i + batch_size]
            for i in range(0, len(smiles_list), batch_size)
        ]

        print(
            f"Processing {len(smiles_list)} SMILES in {len(batches)} parallel batches..."
        )

        # Submit parallel tasks
        data_path = str(Path(__file__).parent.parent / "data/features")

        # Use the module-level Ray remote function
        futures = [
            process_smiles_batch.remote(batch, self.FP_rad, self.FP_len, data_path)
            for batch in batches
        ]

        # Collect results with progress bar
        batch_results = []
        for future in tqdm(futures, desc="Collecting Ray results"):
            batch_results.append(ray.get(future))

        # Flatten results
        all_fps = []
        for batch_fps in batch_results:
            all_fps.extend(batch_fps)

        return np.array(all_fps)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Numpy Fingerprints Inference")
    parser.add_argument(
        "--mol",
        help="Path to the molecule file (.csv) or singular SMILES string",
        required=True,
    )

    parser.add_argument(
        "--smiles-col",
        help="Column name for SMILES string",
        type=str,
        default="smi_can",
    )
    args = parser.parse_args()

    weight_path = "models/Numpy/MP_Morgan_hybrid.pkl"
    model = NumpyFingerprints(weights_path=weight_path)
    if ".csv" in args.mol:
        print("-" * 50)
        print("DATA LOADING")
        print("-" * 50)
        print(f"Loading SMILES from {args.mol}")
        df = pd.read_csv(args.mol)
        smiles = df[args.smiles_col].tolist()
        print(f"Loaded {len(smiles)} SMILES from CSV")

        prediction = model.predict_batch_from_smiles(smiles)
        prediction = [pred[0] for pred in prediction]

        print("-" * 50)
        print("SAVING RESULTS")
        print("-" * 50)
        # Save predictions to file
        output_df = pd.DataFrame({"smi_can": smiles, "price": prediction})
        # replace 0 prices with "Error"
        output_df["price"] = output_df["price"].replace(0, "Error")
        output_df.to_csv("prices.csv", index=False)
        print(f"Predictions saved to prices.csv")
        print("-" * 50)
    else:
        print("-" * 50)
        print("SINGLE MOLECULE PREDICTION")
        print("-" * 50)
        prediction = model.predict_from_smiles(args.mol)
        print(f"Prediction for {args.mol}: {prediction:.2f}")
        print("-" * 50)
