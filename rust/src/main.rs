use anyhow::{Context, Result};
use ndarray::{Array, Array1, Array2, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder};
use std::path::Path;

pub struct PeptideTransformer {
    session: Session,
}

impl PeptideTransformer {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = SessionBuilder::new()?
            // .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)
            .context("Failed to create session from ONNX model")?;

        Ok(Self { session })
    }

    /// Converts a peptide sequence into input IDs
    fn sequence_to_input_ids(&self, sequence: &str) -> Vec<i64> {
        let mut ids = Vec::with_capacity(sequence.len());
        // Add start token (^)
        ids.push(94);
        // Add sequence
        ids.extend(sequence.chars().map(|c| c as i64));
        // Add end token ($)
        ids.push(36);
        ids
    }

    /// Creates position IDs for the sequence
    fn create_position_ids(&self, length: usize) -> Vec<f32> {
        (0..length).map(|i| i as f32).collect()
    }

    /// Creates the padding mask for the sequence
    fn create_padding_mask(&self, input_ids: &[i64]) -> Vec<f32> {
        input_ids
            .iter()
            .map(|&id| if id == 32 { f32::NEG_INFINITY } else { 0.0 })
            .collect()
    }

    /// Runs inference on a peptide sequence
    pub fn predict(
        &self,
        sequence: &str,
        charge: u8,
    ) -> Result<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>> {
        let input_ids = self.sequence_to_input_ids(sequence);
        let seq_len = input_ids.len();
        let position_ids = self.create_position_ids(seq_len);
        let padding_mask = self.create_padding_mask(&input_ids);

        // Reshape inputs to match the model's expected shapes
        let input_ids = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let position_ids = Array2::from_shape_vec((1, seq_len), position_ids)?;
        let padding_mask = Array2::from_shape_vec((1, seq_len), padding_mask)?;
        let charge = Array2::from_shape_vec((1, 1), vec![charge as f32])?;

        // Create input tensors
        let inputs = ort::inputs![
            "input_ids_ns" => input_ids.view(),
            "position_ids_ns" => position_ids.view(),
            "src_key_padding_mask_ns" => padding_mask.view(),
            "charge_n1" => charge.view(),
        ]?;

        // Run inference
        let outputs = self
            .session
            .run(inputs)
            .context("Failed to run inference")?;

        // Extract the output tensor
        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        // Convert to Vec<f32>
        Ok(output.to_owned())
    }
}

// Example usage and tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_inference() -> Result<()> {
        let model = PeptideTransformer::new("test_model.onnx")?;
        let result = model.predict("MYPEPTIDEK", 2)?;

        // The output should be a vector of length 12 (as per the Python model)
        assert_eq!(result.len(), 12);
        Ok(())
    }

    #[test]
    fn test_sequence_conversion() {
        let model = PeptideTransformer::new("test_model.onnx").unwrap();
        let ids = model.sequence_to_input_ids("MYPEPTIDEK");

        // Check start and end tokens
        assert_eq!(ids[0], 94); // ^
        assert_eq!(ids[ids.len() - 1], 36); // $

        // Check sequence conversion
        assert_eq!(ids[1], 'M' as i64);
        assert_eq!(ids[2], 'Y' as i64);
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Create model instance
    let model = PeptideTransformer::new("../test_model.onnx")?;

    // Run inference
    let result = model.predict("MYPEPTIDEK", 2)?;
    println!("Prediction: {:?}", result);

    Ok(())
}
