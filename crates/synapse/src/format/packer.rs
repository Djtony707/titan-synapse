use anyhow::Result;
use std::path::Path;
use super::manifest::SynapseManifest;

/// Pack a specialist into a .synapse file
pub fn pack(manifest: &SynapseManifest, output_dir: &Path, output_path: &Path) -> Result<()> {
    // TODO: Create tar/zip archive with:
    // - manifest.json
    // - base.gguf (or reference)
    // - adapters/*.safetensors
    // - knowledge/graph.sqlite
    // - agent.yaml

    let manifest_json = serde_json::to_string_pretty(manifest)?;
    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest_json)?;

    tracing::info!("Packed specialist '{}' to {}", manifest.name, output_path.display());
    Ok(())
}

/// Unpack a .synapse file
pub fn unpack(synapse_path: &Path, output_dir: &Path) -> Result<SynapseManifest> {
    // TODO: Extract archive and return manifest
    let manifest_path = output_dir.join("manifest.json");
    if manifest_path.exists() {
        let content = std::fs::read_to_string(&manifest_path)?;
        let manifest: SynapseManifest = serde_json::from_str(&content)?;
        return Ok(manifest);
    }

    anyhow::bail!("No manifest.json found in {}", synapse_path.display())
}
