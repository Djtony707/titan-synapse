use std::collections::HashMap;

/// Manages the pool of active specialists and their LoRA adapters
pub struct SpecialistPool {
    /// Currently loaded specialists
    loaded: HashMap<String, SpecialistState>,
    /// Max specialists to keep loaded
    max_loaded: usize,
}

pub struct SpecialistState {
    pub name: String,
    pub adapter_loaded: bool,
    pub last_used: std::time::Instant,
    pub request_count: u64,
}

impl SpecialistPool {
    pub fn new(max_loaded: usize) -> Self {
        Self {
            loaded: HashMap::new(),
            max_loaded,
        }
    }

    /// Ensure a specialist is loaded, evicting LRU if necessary
    pub fn ensure_loaded(&mut self, name: &str) -> bool {
        if self.loaded.contains_key(name) {
            if let Some(state) = self.loaded.get_mut(name) {
                state.last_used = std::time::Instant::now();
                state.request_count += 1;
            }
            return true;
        }

        // Need to load — evict LRU if at capacity
        if self.loaded.len() >= self.max_loaded {
            self.evict_lru();
        }

        self.loaded.insert(name.to_string(), SpecialistState {
            name: name.to_string(),
            adapter_loaded: false,
            last_used: std::time::Instant::now(),
            request_count: 1,
        });

        false // Was not loaded, needs adapter swap
    }

    fn evict_lru(&mut self) {
        if let Some(oldest) = self.loaded.values()
            .min_by_key(|s| s.last_used)
            .map(|s| s.name.clone())
        {
            tracing::info!("Evicting LRU specialist: {oldest}");
            self.loaded.remove(&oldest);
        }
    }

    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
}
