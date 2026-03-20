use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::config::CloudConfig;
use crate::memory::KnowledgeGraph;

/// Cloud Fallback — when local specialists aren't confident enough, route to a cloud API.
/// The cloud response is captured as training data so the specialist learns to handle
/// similar queries next time. Over time, cloud usage drops to zero.
///
/// This is the key insight: use the cloud as a TEACHER, not a crutch.
/// Every cloud call makes the local system smarter.
pub struct CloudFallback {
    config: CloudConfig,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct CloudRequest {
    model: String,
    messages: Vec<CloudMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CloudMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct CloudResponse {
    choices: Vec<CloudChoice>,
}

#[derive(Debug, Deserialize)]
struct CloudChoice {
    message: CloudMessage,
}

/// Result of a cloud fallback call
#[derive(Debug)]
pub struct FallbackResult {
    pub text: String,
    pub model_used: String,
    pub learned: bool,
}

impl CloudFallback {
    pub fn new(config: &CloudConfig) -> Option<Self> {
        if !config.enabled || config.api_base.is_none() {
            return None;
        }
        Some(Self {
            config: config.clone(),
            client: reqwest::Client::new(),
        })
    }

    /// Check if cloud fallback is available
    pub fn is_available(&self) -> bool {
        self.config.enabled && self.config.api_base.is_some()
    }

    /// Call cloud API and capture response as training data
    pub async fn fallback(
        &self,
        prompt: &str,
        specialist: &str,
        local_response: Option<&str>,
        knowledge: &KnowledgeGraph,
    ) -> Result<FallbackResult> {
        let api_base = self.config.api_base.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No cloud API base configured"))?;
        let model = self.config.model.as_deref().unwrap_or("gpt-4o");

        tracing::info!(
            "☁️ Cloud fallback: specialist '{specialist}' not confident enough, asking {model}"
        );

        let request = CloudRequest {
            model: model.to_string(),
            messages: vec![CloudMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: 0.3, // Lower temp for teaching — we want accurate answers
            max_tokens: 2048,
        };

        let url = format!("{}/v1/chat/completions", api_base.trim_end_matches('/'));
        let mut req = self.client.post(&url)
            .header("Content-Type", "application/json")
            .json(&request);

        if let Some(key) = &self.config.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let resp = req.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Cloud API error {status}: {body}"));
        }

        let cloud_resp: CloudResponse = resp.json().await?;
        let cloud_text = cloud_resp.choices.first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("Empty cloud response"))?;

        // Store as training data — the cloud response is the "preferred" output
        // The local response (if any) is the "rejected" output
        // This creates a DPO preference pair for training
        let learned = if let Some(local) = local_response {
            // DPO pair: cloud answer (preferred) vs local answer (rejected)
            let _ = knowledge.add_preference(
                specialist,
                prompt,
                &cloud_text,  // preferred (cloud)
                local,        // rejected (local)
            );
            tracing::info!(
                "📚 Captured DPO pair from cloud for specialist '{specialist}' — next time will handle locally"
            );
            true
        } else {
            // No local response to compare — still log the cloud response as knowledge
            let _ = knowledge.log_message(
                &format!("cloud-fallback-{}", uuid::Uuid::new_v4()),
                "assistant",
                &cloud_text,
                Some(specialist),
            );
            // Extract facts from the cloud response
            let _ = crate::memory::KnowledgeExtractor::extract_and_store(
                knowledge, &cloud_text, "cloud",
            );
            true
        };

        Ok(FallbackResult {
            text: cloud_text,
            model_used: model.to_string(),
            learned,
        })
    }

    /// Minimum confidence threshold for using local specialist
    /// Below this, we fall back to cloud
    pub fn confidence_threshold() -> f32 {
        0.4 // If specialist confidence < 40%, use cloud
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CloudConfig;

    #[test]
    fn test_cloud_fallback_disabled() {
        let config = CloudConfig::default();
        assert!(CloudFallback::new(&config).is_none());
    }

    #[test]
    fn test_cloud_fallback_enabled() {
        let config = CloudConfig {
            enabled: true,
            api_base: Some("http://localhost:11434".into()),
            api_key: None,
            model: Some("qwen3:30b".into()),
        };
        let fallback = CloudFallback::new(&config);
        assert!(fallback.is_some());
        assert!(fallback.unwrap().is_available());
    }

    #[test]
    fn test_confidence_threshold() {
        assert!(CloudFallback::confidence_threshold() > 0.0);
        assert!(CloudFallback::confidence_threshold() < 1.0);
    }
}
