from pyquil import Program, get_qc
from pyquil.gates import H, MEASURE
import numpy as np
import wave
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumNoiseGenerator:
    def __init__(self, qvm_type="9q-square-noisy-qvm"):
        """Initialize the quantum noise generator with specified QVM type."""
        self.qvm_type = qvm_type
        self.qc = None
        
    def _initialize_qvm(self):
        """Initialize the quantum virtual machine."""
        if self.qc is None:
            try:
                self.qc = get_qc(self.qvm_type)
                logger.info(f"Successfully initialized {self.qvm_type}")
            except Exception as e:
                logger.error(f"Failed to initialize QVM: {str(e)}")
                raise

    def generate_random_numbers(self, batch_size=100):
        """
        Generates random bits using 9 qubits.
        
        Args:
            batch_size (int): Number of random number batches to generate
            
        Returns:
            np.ndarray: Array of random bits
        """
        self._initialize_qvm()
        
        program = Program()
        ro = program.declare("ro", "BIT", batch_size * 9)  # Reserve classical bits for 9 qubits
        
        # Apply Hadamard gate to all 9 qubits
        for i in range(9):
            program += H(i)
        
        # Measure all 9 qubits
        program += [MEASURE(i, ro[i + j * 9]) for j in range(batch_size) for i in range(9)]
        
        try:
            executable = self.qc.compile(program)
            result = self.qc.run(executable)  # Run the program
            result_array = result.readout_data["ro"]  # Extract measurement results
            return result_array.flatten()  # Flatten to 1D array
        except Exception as e:
            logger.error(f"Error generating quantum random numbers: {str(e)}")
            raise

    def generate_white_noise(self, duration_seconds=10, batch_size=100, sampling_rate=44100, output_file="quantum_white_noise.wav"):
        """
        Generates and saves quantum white noise as a WAV file.
        
        Args:
            duration_seconds (int): Duration of the audio in seconds
            batch_size (int): Number of quantum random numbers to generate per batch
            sampling_rate (int): Audio sampling rate in Hz
            output_file (str): Output WAV file path
        """
        num_samples = duration_seconds * sampling_rate
        noise_data = np.zeros(num_samples, dtype=np.float32)
        bits_per_batch = batch_size * 9
        chunks = num_samples // bits_per_batch
        
        logger.info(f"Generating {duration_seconds}s of quantum white noise at {sampling_rate}Hz using 9 qubits.")
        
        try:
            # Use tqdm for progress tracking
            for chunk in tqdm(range(chunks), desc="Generating noise"):
                start_idx = chunk * bits_per_batch
                random_bits = self.generate_random_numbers(batch_size)
                noise_data[start_idx:start_idx + len(random_bits)] = random_bits * 2 - 1
                
            # Handle remaining samples if any
            remaining = num_samples - chunks * bits_per_batch
            if remaining > 0:
                remaining_batches = (remaining + 8) // 9
                random_bits = self.generate_random_numbers(remaining_batches)
                noise_data[-remaining:] = random_bits[:remaining] * 2 - 1
                
            # Scale to 16-bit PCM range
            scaled_noise = (noise_data * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open(output_file, "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sampling_rate)
                wav_file.writeframes(scaled_noise.tobytes())
                
            logger.info(f"Successfully saved white noise to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating white noise: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    generator = QuantumNoiseGenerator()
    generator.generate_white_noise(
        duration_seconds=10,
        batch_size=100,
        sampling_rate=44100,
        output_file="quantum_white_noise.wav"
    )