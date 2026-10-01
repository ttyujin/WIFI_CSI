//! Dataset recording contract for the three-class ESP32 activity study.
//!
//! This module deliberately contains no classifier or inference logic.  It
//! only validates research-session metadata and builds the JSONL records that
//! accompany CSI-derived frames captured by the sensing server.

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::FeatureInfo;

pub(crate) const SCHEMA_VERSION: &str = "activity-csi-v1";
pub(crate) const REQUIRED_SUBCARRIERS: usize = 306;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) enum ActivityLabel {
    Sitting,
    Moving,
    Lying,
}

impl ActivityLabel {
    pub(crate) const ALL: [Self; 3] = [Self::Sitting, Self::Moving, Self::Lying];

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "SITTING" => Some(Self::Sitting),
            "MOVING" => Some(Self::Moving),
            "LYING" => Some(Self::Lying),
            _ => None,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Sitting => "SITTING",
            Self::Moving => "MOVING",
            Self::Lying => "LYING",
        }
    }
}

/// Session IDs become filenames, so keep the accepted alphabet intentionally
/// small. This rejects path traversal, whitespace, control characters, and
/// platform-specific separators while preserving names such as `sitting_001`.
pub(crate) fn valid_session_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}

pub(crate) fn session_path(data_dir: &Path, label: ActivityLabel, session_id: &str) -> PathBuf {
    data_dir
        .join("recordings")
        .join("activity")
        .join(label.as_str())
        .join(format!("{session_id}.jsonl"))
}

pub(crate) fn session_metadata(
    session_id: &str,
    label: ActivityLabel,
    timestamp: f64,
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "session",
        "session_id": session_id,
        "label": label.as_str(),
        "timestamp": timestamp,
        "source": "esp32",
    })
}

/// One accepted raw ADR-018 CSI frame after parsing and feature extraction.
/// This travels on a recording-only in-process channel, never on the public
/// sensing WebSocket, so edge-vitals/demo messages cannot enter the dataset.
#[derive(Debug, Clone)]
pub(crate) struct ActivityCsiSample {
    pub(crate) timestamp: f64,
    pub(crate) server_tick: u64,
    pub(crate) node_id: u8,
    pub(crate) sequence: u32,
    pub(crate) rssi_dbm: f64,
    pub(crate) subcarrier_count: usize,
    pub(crate) amplitude: Vec<f64>,
    pub(crate) features: FeatureInfo,
}

/// The single source of truth for whether an ADR-018 sample may enter an
/// activity dataset. The caller supplies only samples produced by the ESP32
/// ADR-018 receive path, so source/protocol provenance is guaranteed by the
/// type's construction site. Keep readiness and JSONL writing on this same
/// predicate so a fresh but non-recordable ESP32 message cannot arm recording.
pub(crate) fn is_recordable_csi(sample: &ActivityCsiSample) -> bool {
    sample.node_id != 0
        && sample.subcarrier_count == REQUIRED_SUBCARRIERS
        && sample.amplitude.len() == REQUIRED_SUBCARRIERS
        && sample.timestamp.is_finite()
        && sample.rssi_dbm.is_finite()
        && sample.amplitude.iter().all(|value| value.is_finite())
}

/// Attach immutable research metadata to one accepted raw CSI sample. Pose,
/// persons, vitals, classification, and simulator fields are intentionally
/// absent.
pub(crate) fn frame_record(
    sample: &ActivityCsiSample,
    session_id: &str,
    label: ActivityLabel,
) -> Option<serde_json::Value> {
    if !is_recordable_csi(sample) {
        return None;
    }

    Some(serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "frame",
        "session_id": session_id,
        "label": label.as_str(),
        "timestamp": sample.timestamp,
        "server_tick": sample.server_tick,
        "node_id": sample.node_id,
        "sequence": sample.sequence,
        "source": "esp32",
        "rssi_dbm": sample.rssi_dbm,
        "subcarrier_count": sample.subcarrier_count,
        "amplitude_count": sample.amplitude.len(),
        "amplitude": sample.amplitude,
        "features": sample.features,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(subcarrier_count: usize, amplitude: Vec<f64>) -> ActivityCsiSample {
        ActivityCsiSample {
            timestamp: 1_725_000_000.125,
            server_tick: 11,
            node_id: 1,
            sequence: 42,
            rssi_dbm: -57.5,
            subcarrier_count,
            amplitude,
            features: FeatureInfo {
                mean_rssi: -57.5,
                variance: 0.42,
                motion_band_power: 0.31,
                breathing_band_power: 0.07,
                dominant_freq_hz: 0.8,
                change_points: 3,
                spectral_power: 1.25,
            },
        }
    }

    #[test]
    fn sitting_label_is_allowed() {
        assert_eq!(
            ActivityLabel::parse("SITTING"),
            Some(ActivityLabel::Sitting)
        );
    }

    #[test]
    fn moving_label_is_allowed() {
        assert_eq!(ActivityLabel::parse("MOVING"), Some(ActivityLabel::Moving));
    }

    #[test]
    fn lying_label_is_allowed() {
        assert_eq!(ActivityLabel::parse("LYING"), Some(ActivityLabel::Lying));
    }

    #[test]
    fn any_other_label_is_rejected() {
        for invalid in ["STANDING", "sitting", "", "SITTING "] {
            assert_eq!(ActivityLabel::parse(invalid), None, "accepted {invalid:?}");
        }
    }

    #[test]
    fn session_id_rejects_missing_and_unsafe_values() {
        assert!(!valid_session_id(""));
        assert!(!valid_session_id("../sitting_001"));
        assert!(!valid_session_id("sitting/001"));
        assert!(valid_session_id("sitting_001"));
        assert!(valid_session_id("moving-002"));
    }

    #[test]
    fn recorder_and_readiness_share_the_recordable_csi_predicate() {
        let valid = sample(REQUIRED_SUBCARRIERS, vec![1.0; REQUIRED_SUBCARRIERS]);
        let zero_subcarriers = sample(0, vec![]);
        let wrong_declared_count = sample(64, vec![1.0; REQUIRED_SUBCARRIERS]);
        let wrong_amplitude_length = sample(REQUIRED_SUBCARRIERS, vec![1.0; 305]);

        assert!(is_recordable_csi(&valid));
        assert!(frame_record(&valid, "sitting_001", ActivityLabel::Sitting).is_some());

        for candidate in [
            valid,
            zero_subcarriers,
            wrong_declared_count,
            wrong_amplitude_length,
        ] {
            assert_eq!(
                is_recordable_csi(&candidate),
                frame_record(&candidate, "sitting_001", ActivityLabel::Sitting).is_some()
            );
        }
    }
}
