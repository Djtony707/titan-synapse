use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::server::SharedState;
use crate::streaming;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let is_stream = request.stream.unwrap_or(false);

    if is_stream {
        streaming::stream_response(state, request).await.into_response()
    } else {
        complete_response(state, request).await.into_response()
    }
}

async fn complete_response(
    state: SharedState,
    request: ChatCompletionRequest,
) -> Json<ChatCompletionResponse> {
    let state = state.read().await;
    let model_name = request.model.clone().unwrap_or_else(|| state.config.base_model.clone());

    // Route through orchestrator with Hebbian routing
    let result = state.orchestrator.process(
        &request.messages,
        &state.engine,
        request.max_tokens,
        request.temperature,
        Some(&state.knowledge),
    ).await;

    let (response_text, usage) = match result {
        Ok(result) => (result.text, Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.total_tokens,
        }),
        Err(e) => (format!("Error: {e}"), Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }),
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: chrono::Utc::now().timestamp(),
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: response_text,
            },
            finish_reason: "stop".into(),
        }],
        usage,
    };

    Json(response)
}

pub async fn list_models(
    State(state): State<SharedState>,
) -> Json<ModelList> {
    let state = state.read().await;

    let mut models = vec![
        ModelInfo {
            id: "synapse".into(),
            object: "model".into(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "titan-synapse".into(),
        },
    ];

    // Add each specialist as a model
    for specialist in &state.config.specialists {
        models.push(ModelInfo {
            id: format!("synapse/{}", specialist.name),
            object: "model".into(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "titan-synapse".into(),
        });
    }

    Json(ModelList {
        object: "list".into(),
        data: models,
    })
}
