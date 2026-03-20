use anyhow::Result;
use std::path::{Path, PathBuf};
use super::manifest::SynapseManifest;

/// Pack a specialist into a .synapse directory bundle
///
/// Layout:
///   <name>.synapse/
///     manifest.json
///     model.gguf (if found)
///     adapters/*.safetensors
///     knowledge/graph.sqlite
///     agent.yaml
pub fn pack(
    manifest: &SynapseManifest,
    models_dir: &Path,
    adapters_dir: &Path,
    knowledge_db: Option<&Path>,
    output_path: &Path,
) -> Result<()> {
    std::fs::create_dir_all(output_path)?;

    // Write manifest
    let manifest_json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(output_path.join("manifest.json"), manifest_json)?;

    // Copy model GGUF if found
    let model_glob = format!("{}*", manifest.name);
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "gguf") {
                let dest = output_path.join("model.gguf");
                // Symlink instead of copy (saves disk space for multi-GB files)
                #[cfg(unix)]
                {
                    if let Err(_) = std::os::unix::fs::symlink(&path, &dest) {
                        std::fs::copy(&path, &dest)?;
                    }
                }
                #[cfg(not(unix))]
                {
                    std::fs::copy(&path, &dest)?;
                }
                tracing::info!("Linked model: {}", path.display());
                break;
            }
        }
    }

    // Copy adapters
    let adapters_out = output_path.join("adapters");
    std::fs::create_dir_all(&adapters_out)?;
    let mut adapter_count = 0u32;

    if adapters_dir.exists() {
        for entry in std::fs::read_dir(adapters_dir)?.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                let name = path.file_name().unwrap();
                std::fs::copy(&path, adapters_out.join(name))?;
                adapter_count += 1;
            }
        }
    }

    // Copy knowledge graph
    if let Some(db_path) = knowledge_db {
        if db_path.exists() {
            let knowledge_out = output_path.join("knowledge");
            std::fs::create_dir_all(&knowledge_out)?;
            std::fs::copy(db_path, knowledge_out.join("graph.sqlite"))?;
        }
    }

    tracing::info!(
        "Packed specialist '{}': {} adapters, output: {}",
        manifest.name,
        adapter_count,
        output_path.display()
    );

    Ok(())
}

/// Unpack a .synapse directory bundle into the working directories
pub fn unpack(
    synapse_path: &Path,
    models_dir: &Path,
    adapters_dir: &Path,
) -> Result<SynapseManifest> {
    if !synapse_path.exists() {
        anyhow::bail!("Path does not exist: {}", synapse_path.display());
    }

    let manifest_path = synapse_path.join("manifest.json");
    if !manifest_path.exists() {
        anyhow::bail!("No manifest.json found in {}", synapse_path.display());
    }

    let content = std::fs::read_to_string(&manifest_path)?;
    let manifest: SynapseManifest = serde_json::from_str(&content)?;

    // Copy model if present
    let model_file = synapse_path.join("model.gguf");
    if model_file.exists() {
        std::fs::create_dir_all(models_dir)?;
        let dest = models_dir.join(format!("{}.gguf", manifest.name));
        if !dest.exists() {
            std::fs::copy(&model_file, &dest)?;
            tracing::info!("Installed model: {}", dest.display());
        }
    }

    // Copy adapters
    let adapters_src = synapse_path.join("adapters");
    if adapters_src.exists() {
        std::fs::create_dir_all(adapters_dir)?;
        for entry in std::fs::read_dir(&adapters_src)?.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                let name = path.file_name().unwrap();
                let dest = adapters_dir.join(name);
                std::fs::copy(&path, &dest)?;
            }
        }
    }

    // Copy knowledge graph
    let knowledge_src = synapse_path.join("knowledge").join("graph.sqlite");
    if knowledge_src.exists() {
        let knowledge_dir = synapse_path.parent()
            .unwrap_or(Path::new("."))
            .join("knowledge");
        std::fs::create_dir_all(&knowledge_dir)?;
        std::fs::copy(&knowledge_src, knowledge_dir.join("graph.sqlite"))?;
    }

    tracing::info!("Unpacked specialist '{}' (base: {})", manifest.name, manifest.base_model);
    Ok(manifest)
}

/// List all .synapse bundles in a directory
pub fn list_bundles(dir: &Path) -> Result<Vec<(PathBuf, SynapseManifest)>> {
    let mut bundles = Vec::new();

    if !dir.exists() {
        return Ok(bundles);
    }

    for entry in std::fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let manifest_path = path.join("manifest.json");
            if manifest_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&manifest_path) {
                    if let Ok(manifest) = serde_json::from_str::<SynapseManifest>(&content) {
                        bundles.push((path, manifest));
                    }
                }
            }
        }
    }

    Ok(bundles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_and_unpack() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path().join("models");
        let adapters_dir = tmp.path().join("adapters");
        let output = tmp.path().join("test_specialist.synapse");
        std::fs::create_dir_all(&models_dir).unwrap();
        std::fs::create_dir_all(&adapters_dir).unwrap();

        let manifest = SynapseManifest::new("test_specialist", "Qwen3-3B");

        // Pack
        pack(&manifest, &models_dir, &adapters_dir, None, &output).unwrap();
        assert!(output.join("manifest.json").exists());

        // Unpack into new dirs
        let new_models = tmp.path().join("new_models");
        let new_adapters = tmp.path().join("new_adapters");
        let unpacked = unpack(&output, &new_models, &new_adapters).unwrap();
        assert_eq!(unpacked.name, "test_specialist");
        assert_eq!(unpacked.base_model, "Qwen3-3B");
    }

    #[test]
    fn test_list_bundles() {
        let tmp = tempfile::tempdir().unwrap();

        // Create two bundles
        for name in ["alpha", "beta"] {
            let bundle_dir = tmp.path().join(format!("{name}.synapse"));
            std::fs::create_dir_all(&bundle_dir).unwrap();
            let manifest = SynapseManifest::new(name, "Qwen3-3B");
            std::fs::write(
                bundle_dir.join("manifest.json"),
                serde_json::to_string(&manifest).unwrap(),
            ).unwrap();
        }

        let bundles = list_bundles(tmp.path()).unwrap();
        assert_eq!(bundles.len(), 2);
    }
}
