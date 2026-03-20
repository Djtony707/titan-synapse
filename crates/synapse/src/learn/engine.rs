use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Bridge to the Python learning sidecar
pub struct LearningEngine {
    sidecar_url: String,
    enabled: bool,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
pub struct EvalRequest {
    pub specialist: String,
    pub prompt: String,
    pub response: String,
}

#[derive(Debug, Deserialize)]
pub struct EvalResponse {
    pub score: f32,
    pub improved_response: Option<String>,
    pub feedback: String,
}

#[derive(Debug, Serialize)]
pub struct TrainRequest {
    pub specialist: String,
    pub base_model: String,
}

#[derive(Debug, Deserialize)]
pub struct TrainResponse {
    pub adapter_path: String,
    pub loss: f32,
    pub pairs_used: u32,
}

#[derive(Debug, Deserialize)]
pub struct LearnStatus {
    pub pairs_collected: u32,
    pub training_queue: u32,
    pub last_trained: Option<String>,
    pub adapters_created: u32,
}

impl LearningEngine {
    pub fn new(sidecar_url: &str, enabled: bool) -> Self {
        Self {
            sidecar_url: sidecar_url.to_string(),
            enabled,
            client: reqwest::Client::new(),
        }
    }

    /// Submit a response for evaluation (async, non-blocking)
    pub async fn evaluate(&self, request: EvalRequest) -> Result<EvalResponse> {
        if !self.enabled {
            return Ok(EvalResponse {
                score: 5.0,
                improved_response: None,
                feedback: "Learning disabled".into(),
            });
        }

        let resp = self.client
            .post(format!("{}/evaluate", self.sidecar_url))
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        Ok(resp)
    }

    /// Trigger training immediately
    pub async fn train_now(&self, request: TrainRequest) -> Result<TrainResponse> {
        let resp = self.client
            .post(format!("{}/train", self.sidecar_url))
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        Ok(resp)
    }

    /// Get learning status
    pub async fn status(&self) -> Result<LearnStatus> {
        if !self.enabled {
            return Ok(LearnStatus {
                pairs_collected: 0,
                training_queue: 0,
                last_trained: None,
                adapters_created: 0,
            });
        }

        let resp = self.client
            .get(format!("{}/status", self.sidecar_url))
            .send()
            .await?
            .json()
            .await?;

        Ok(resp)
    }
}
