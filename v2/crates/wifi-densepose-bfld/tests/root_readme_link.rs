//! The research fork's root README documents its own service and links to
//! upstream RuView in its acknowledgement. Keep that discovery/attribution
//! contract, while checking BFLD-specific documentation in the crate README
//! rather than requiring upstream-only features in the research overview.

#![cfg(feature = "std")]

const ROOT_README: &str = include_str!("../../../../README.md");
const CRATE_README: &str = include_str!("../README.md");
const ROOT_LICENSE: &str = include_str!("../../../../LICENSE");

#[test]
fn root_readme_preserves_research_scope_and_upstream_attribution() {
    let title = ROOT_README
        .lines()
        .next()
        .expect("root README needs a title");
    assert!(
        title.starts_with("# ") && title.contains("Wi-Fi CSI") && title.contains("고령자"),
        "the research project must remain the root README's primary subject",
    );
    let acknowledgement = ROOT_README
        .split_once("## Based on RuView / Acknowledgement")
        .expect("root README must acknowledge that the project is based on RuView")
        .1
        .split("\n## ")
        .next()
        .unwrap();
    assert!(
        acknowledgement.contains("[ruvnet/RuView](https://github.com/ruvnet/RuView)"),
        "acknowledgement must link to upstream RuView for original functionality and documentation",
    );
    for notice in [
        "MIT License",
        "Copyright (c) 2024 rUv",
        "[LICENSE](LICENSE)",
    ] {
        assert!(
            acknowledgement.contains(notice),
            "acknowledgement must preserve {notice}"
        );
    }
    assert!(
        ROOT_LICENSE.contains("MIT License") && ROOT_LICENSE.contains("Copyright (c) 2024 rUv")
    );
    let scope_disclaimer = acknowledgement
        .split("\n\n")
        .find(|paragraph| paragraph.contains("본 연구의 기능 또는 검증 결과가 아닙니다"))
        .expect(
            "upstream-only features must not be presented as research functionality or results",
        );
    for feature in ["DensePose", "vital signs", "fall detection"] {
        assert!(
            scope_disclaimer.contains(feature),
            "scope disclaimer must cover {feature}"
        );
    }
}

#[test]
fn bfld_crate_readme_mentions_bfld_acronym_and_full_name() {
    assert!(
        CRATE_README.contains("BFLD"),
        "crate README must mention the BFLD acronym",
    );
    assert!(
        CRATE_README.contains("Beamforming Feedback Layer for Detection"),
        "crate README must expand the BFLD acronym at least once",
    );
}

#[test]
fn bfld_crate_readme_cites_all_six_bfld_adrs() {
    for adr in [
        "ADR-118", "ADR-119", "ADR-120", "ADR-121", "ADR-122", "ADR-123",
    ] {
        assert!(
            CRATE_README.contains(adr),
            "crate README must cite {adr} so the discovery path is intact",
        );
    }
}

#[test]
fn bfld_crate_readme_points_at_research_bundle() {
    assert!(
        CRATE_README.contains("docs/research/BFLD/"),
        "crate README must point at the BFLD research dossier",
    );
}

#[test]
fn bfld_crate_readme_documents_three_structural_invariants() {
    assert!(
        CRATE_README.contains("Raw BFI never exits"),
        "crate README must mention invariant I1",
    );
    assert!(
        CRATE_README.contains("in-RAM-only") || CRATE_README.contains("in-RAM only"),
        "crate README must mention invariant I2",
    );
    assert!(
        CRATE_README.contains("Cross-site"),
        "crate README must mention invariant I3",
    );
}
