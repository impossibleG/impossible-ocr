use std::{collections::BTreeSet, fmt};

use impossible_ocr_domain::{Confidence, OcrError, OcrErrorCode};

/// Decoded CTC sequence with mean emitted-character confidence.
#[derive(Clone, PartialEq)]
pub struct DecodedSequence {
    /// Text after blank removal and repeated-class collapse.
    pub text: String,
    /// Arithmetic mean of selected probabilities for emitted classes, or zero when empty.
    pub confidence: Confidence,
}

impl fmt::Debug for DecodedSequence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodedSequence")
            .field("text", &"[REDACTED]")
            .field("confidence", &self.confidence)
            .finish()
    }
}

/// CTC decoder bound to one exact injected model dictionary.
#[derive(Clone)]
pub struct CtcDecoder {
    dictionary: Vec<String>,
    dictionary_sha256: Option<[u8; 32]>,
}

impl fmt::Debug for CtcDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CtcDecoder")
            .field("dictionary", &"[REDACTED]")
            .field("dictionary_entries", &self.dictionary.len())
            .field("dictionary_sha256", &self.dictionary_sha256)
            .finish()
    }
}

impl CtcDecoder {
    /// Constructs a decoder. Class zero is always the blank class; dictionary entry zero maps to
    /// model class one. When requested, one space entry is appended. The supplied dictionary must
    /// not already contain a space in that mode because silently reusing it could hide a model
    /// class-index mismatch.
    ///
    /// # Errors
    /// Returns `invalid_request` for an empty dictionary, duplicate entries, entries that are not
    /// exactly one Unicode scalar, or an ambiguous appended-space configuration.
    pub fn new(mut dictionary: Vec<String>, append_space: bool) -> Result<Self, OcrError> {
        if append_space && dictionary.iter().any(|entry| entry == " ") {
            return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
        }
        if append_space {
            dictionary.push(" ".to_owned());
        }
        let mut unique = BTreeSet::new();
        let valid = !dictionary.is_empty()
            && dictionary
                .iter()
                .all(|entry| entry.chars().count() == 1 && unique.insert(entry.as_str()));
        if !valid {
            return Err(OcrError::for_code(OcrErrorCode::InvalidRequest));
        }
        Ok(Self {
            dictionary,
            dictionary_sha256: None,
        })
    }

    /// Constructs a decoder while retaining the integrity digest of the exact verified dictionary
    /// artifact. The digest is metadata only: entries and class indices are never reordered or
    /// normalized.
    ///
    /// # Errors
    /// Returns `invalid_request` under the same conditions as [`Self::new`].
    pub fn new_verified(
        dictionary: Vec<String>,
        append_space: bool,
        dictionary_sha256: [u8; 32],
    ) -> Result<Self, OcrError> {
        let mut decoder = Self::new(dictionary, append_space)?;
        decoder.dictionary_sha256 = Some(dictionary_sha256);
        Ok(decoder)
    }

    /// Parses a line-oriented embedded dictionary without silently removing spaces.
    ///
    /// # Errors
    /// Returns `invalid_request` if the resulting dictionary violates [`Self::new`].
    pub fn from_dictionary_text(text: &str, append_space: bool) -> Result<Self, OcrError> {
        let dictionary = text.lines().map(str::to_owned).collect();
        Self::new(dictionary, append_space)
    }

    /// Expected model class count, including blank class zero.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.dictionary.len() + 1
    }

    /// Exact dictionary entries in model class order; entry zero maps to model class one.
    #[must_use]
    pub fn dictionary(&self) -> &[String] {
        &self.dictionary
    }

    /// Digest of the exact verified dictionary artifact when supplied by model admission.
    #[must_use]
    pub const fn dictionary_sha256(&self) -> Option<[u8; 32]> {
        self.dictionary_sha256
    }

    /// Decodes row-major per-timestep class probabilities.
    ///
    /// Ties select the lowest class index. Repeated classes collapse unless separated by blank.
    /// Confidence is averaged only across emitted classes, matching Paddle CTC greedy semantics.
    ///
    /// # Errors
    /// Returns a sanitized internal error for class-count/shape mismatch, non-finite values, or
    /// probabilities outside the inclusive range zero through one.
    pub fn decode_probabilities(
        &self,
        probabilities: &[f32],
        time_steps: usize,
        class_count: usize,
    ) -> Result<DecodedSequence, OcrError> {
        let expected = time_steps.checked_mul(class_count).ok_or_else(internal)?;
        if class_count != self.class_count() || probabilities.len() != expected {
            return Err(internal());
        }
        let mut text = String::new();
        let mut confidence_sum = 0.0_f32;
        let mut emitted = 0_u32;
        let mut previous = None;

        for row in probabilities.chunks_exact(class_count) {
            if row
                .iter()
                .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
            {
                return Err(internal());
            }
            let (class, score) = row
                .iter()
                .copied()
                .enumerate()
                .reduce(|best, candidate| {
                    if candidate.1 > best.1 {
                        candidate
                    } else {
                        best
                    }
                })
                .ok_or_else(internal)?;
            let repeated = previous == Some(class);
            previous = Some(class);
            if class == 0 || repeated {
                continue;
            }
            let symbol = self.dictionary.get(class - 1).ok_or_else(internal)?;
            text.push_str(symbol);
            confidence_sum += score;
            emitted = emitted.checked_add(1).ok_or_else(internal)?;
        }
        let confidence = if emitted == 0 {
            Confidence::ZERO
        } else {
            #[allow(clippy::cast_precision_loss)]
            let mean = confidence_sum / emitted as f32;
            Confidence::new(mean).map_err(|_| internal())?
        };
        Ok(DecodedSequence { text, confidence })
    }
}

const fn internal() -> OcrError {
    OcrError::for_code(OcrErrorCode::Internal)
}

#[cfg(test)]
mod tests {
    use impossible_ocr_domain::OcrErrorCode;

    use super::CtcDecoder;

    fn row(class: usize, score: f32, classes: usize) -> Vec<f32> {
        let mut row = vec![0.0; classes];
        row[class] = score;
        row
    }

    #[test]
    fn blanks_repeats_spaces_and_confidence_match_greedy_ctc()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let decoder = CtcDecoder::new(vec!["a".to_owned(), "b".to_owned()], true)?;
        let classes = decoder.class_count();
        let mut probabilities = Vec::new();
        for (class, score) in [(1, 0.9), (1, 0.8), (0, 1.0), (1, 0.7), (3, 0.6), (2, 0.5)] {
            probabilities.extend(row(class, score, classes));
        }
        let sequence = decoder.decode_probabilities(&probabilities, 6, classes)?;
        assert_eq!(sequence.text, "aa b");
        assert!((sequence.confidence.get() - 0.675).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn blank_only_and_zero_timesteps_return_empty_zero_confidence()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let decoder = CtcDecoder::new(vec!["a".to_owned()], false)?;
        let empty = decoder.decode_probabilities(&[], 0, 2)?;
        assert!(empty.text.is_empty());
        assert_eq!(empty.confidence.get().to_bits(), 0.0_f32.to_bits());
        let blank = decoder.decode_probabilities(&[1.0, 0.0], 1, 2)?;
        assert!(blank.text.is_empty());
        assert_eq!(blank.confidence.get().to_bits(), 0.0_f32.to_bits());
        Ok(())
    }

    #[test]
    fn non_finite_range_and_class_mismatches_fail_closed()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let decoder = CtcDecoder::new(vec!["a".to_owned()], false)?;
        for scores in [
            [f32::NAN, 0.0],
            [f32::INFINITY, 0.0],
            [-0.1, 1.0],
            [0.0, 1.1],
        ] {
            assert_eq!(
                decoder
                    .decode_probabilities(&scores, 1, 2)
                    .map_err(impossible_ocr_domain::OcrError::code),
                Err(OcrErrorCode::Internal)
            );
        }
        assert!(decoder.decode_probabilities(&[0.0, 1.0], 1, 3).is_err());
        assert!(decoder.decode_probabilities(&[0.0], 1, 2).is_err());
        assert!(decoder.decode_probabilities(&[], usize::MAX, 2).is_err());
        Ok(())
    }

    #[test]
    fn dictionary_validation_and_tie_breaking_are_deterministic()
    -> Result<(), impossible_ocr_domain::OcrError> {
        assert!(CtcDecoder::new(Vec::new(), false).is_err());
        assert!(CtcDecoder::new(vec!["a".to_owned(), "a".to_owned()], false).is_err());
        assert!(CtcDecoder::new(vec!["ab".to_owned()], false).is_err());
        assert!(CtcDecoder::new(vec![" ".to_owned()], true).is_err());
        let decoder = CtcDecoder::from_dictionary_text("a\nb\n", true)?;
        let sequence = decoder.decode_probabilities(&[0.1, 0.8, 0.8, 0.0], 1, 4)?;
        assert_eq!(sequence.text, "a");
        Ok(())
    }

    #[test]
    fn unicode_scalar_indices_and_verified_digest_are_preserved()
    -> Result<(), impossible_ocr_domain::OcrError> {
        let digest = [0x5a; 32];
        let decoder = CtcDecoder::new_verified(
            vec!["a".to_owned(), "©".to_owned(), "漢".to_owned()],
            true,
            digest,
        )?;
        assert_eq!(decoder.dictionary(), &["a", "©", "漢", " "]);
        assert_eq!(decoder.dictionary_sha256(), Some(digest));
        assert_eq!(decoder.class_count(), 5);
        let sequence = decoder.decode_probabilities(
            &[
                0.0, 0.0, 1.0, 0.0, 0.0, // © remains class two
                0.0, 0.0, 0.0, 1.0, 0.0, // 漢 remains class three
            ],
            2,
            5,
        )?;
        assert_eq!(sequence.text, "©漢");
        assert!(!format!("{sequence:?}").contains("©漢"));
        assert!(!format!("{decoder:?}").contains('©'));
        Ok(())
    }
}
