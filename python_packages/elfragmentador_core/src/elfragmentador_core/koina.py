import requests
import json
from typing import List, Tuple, Dict, Any


def predict_spectra_koina(
    peptide_charge_pairs: List[Tuple[str, int, int]],
    model_name: str = "ms2pip_timsTOF2023",
) -> Dict[str, Any]:
    """
    Predict MS/MS spectra using the koina.

    Args:
        peptide_charge_pairs: List of tuples containing
            (peptide_sequence, charge, collision_energy)
            Example: [("ACDEK", 2, 23), ("AAAAAAAAAAAAA", 3, 24)]
        model_name: Name of the model to use for prediction
            Default: "ms2pip_timsTOF2023"

    Returns:
        Dictionary containing the parsed API response with predictions
    """
    # API endpoint
    url = f"https://koina.wilhelmlab.org/v2/models/{model_name}/infer"
    # "https://koina.wilhelmlab.org/v2/models//infer"

    # Prepare the request payload
    payload = {
        "id": "0",
        "inputs": [
            {
                "name": "peptide_sequences",
                "shape": [len(peptide_charge_pairs), 1],
                "datatype": "BYTES",
                "data": [pair[0] for pair in peptide_charge_pairs],
            },
            {
                "name": "precursor_charges",
                "shape": [len(peptide_charge_pairs), 1],
                "datatype": "INT32",
                "data": [pair[1] for pair in peptide_charge_pairs],
            },
            {
                "name": "collision_energies",
                "shape": [len(peptide_charge_pairs), 1],
                "datatype": "FP32",
                "data": [pair[2] for pair in peptide_charge_pairs],
            },
        ],
    }

    # Make the API request
    headers = {"content-type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


# Example usage:
if __name__ == "__main__":
    # Example peptide-charge pairs
    peptides = [("ACDEK", 2), ("AAAAAAAAAAAAA", 3)]

    try:
        result = predict_spectra_koina(peptides)
        print("Prediction successful!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
