use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub vram_free_mb: u64,
    pub utilization_percent: f32,
    pub temperature_c: Option<u32>,
}

pub struct VramManager {
    pub budget_mb: u64,
}

impl VramManager {
    pub fn new(max_vram_mb: u64) -> Self {
        Self {
            budget_mb: if max_vram_mb > 0 { max_vram_mb } else { 32768 }, // Default 32GB
        }
    }

    /// Get GPU info via nvidia-smi (works on both local and remote)
    pub async fn gpu_info() -> Result<GpuInfo> {
        // Try nvidia-smi first
        let output = tokio::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
            .output()
            .await;

        match output {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let parts: Vec<&str> = stdout.trim().split(", ").collect();
                if parts.len() >= 6 {
                    return Ok(GpuInfo {
                        name: parts[0].to_string(),
                        vram_total_mb: parts[1].parse().unwrap_or(0),
                        vram_used_mb: parts[2].parse().unwrap_or(0),
                        vram_free_mb: parts[3].parse().unwrap_or(0),
                        utilization_percent: parts[4].parse().unwrap_or(0.0),
                        temperature_c: parts[5].parse().ok(),
                    });
                }
            }
            _ => {}
        }

        // No GPU available
        Ok(GpuInfo {
            name: "No GPU detected".into(),
            vram_total_mb: 0,
            vram_used_mb: 0,
            vram_free_mb: 0,
            utilization_percent: 0.0,
            temperature_c: None,
        })
    }

    /// Calculate how much VRAM is available for Synapse
    pub async fn available_vram(&self) -> Result<u64> {
        let info = Self::gpu_info().await?;
        Ok(info.vram_free_mb.min(self.budget_mb))
    }
}
