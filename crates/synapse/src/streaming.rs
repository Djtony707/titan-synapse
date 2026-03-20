use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use futures::stream::{self, Stream};
use std::convert::Infallible;

use crate::openai::{ChatCompletionChunk, ChatCompletionRequest, ChunkChoice, Delta};
use crate::server::SharedState;

pub async fn stream_response(
    state: SharedState,
    request: ChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state_read = state.read().await;
    let model_name = request.model.clone().unwrap_or_else(|| state_read.config.base_model.clone());

    // Generate full response, then stream it token-by-token
    // In production, this will be replaced with true streaming from the inference engine
    let response_text = state_read.orchestrator.process(
        &request.messages,
        &state_read.engine,
        request.max_tokens,
        request.temperature,
    ).await
        .unwrap_or_else(|e| format!("Error: {e}"));
    drop(state_read);

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    // Split into word-level chunks for streaming effect
    let words: Vec<String> = response_text.split_inclusive(' ')
        .map(|s| s.to_string())
        .collect();

    let stream = stream::iter(
        // First chunk: role
        std::iter::once(Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".into()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            }).unwrap()
        )))
        // Content chunks
        .chain(words.into_iter().map(move |word| {
            Ok(Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(word),
                        },
                        finish_reason: None,
                    }],
                }).unwrap()
            ))
        }))
        // Final chunk: [DONE]
        .chain(std::iter::once(Ok(Event::default().data("[DONE]"))))
    );

    Sse::new(stream)
}
