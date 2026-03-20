/// KV Cache management — PagedAttention-style block allocation
/// Each specialist gets its own KV cache partition from the shared pool

pub struct KvCache {
    /// Block size in tokens
    block_size: usize,
    /// Total blocks available
    total_blocks: usize,
    /// Allocated blocks per specialist
    allocations: std::collections::HashMap<String, Vec<usize>>,
    /// Free block indices
    free_blocks: Vec<usize>,
}

impl KvCache {
    pub fn new(total_vram_mb: u64, block_size: usize) -> Self {
        // Estimate blocks from VRAM budget
        // Each block ~= block_size * 2 (K+V) * hidden_dim * 2 bytes (fp16)
        // For a 3B model with hidden_dim=2048: ~8KB per block of 16 tokens
        let bytes_per_block = block_size * 2 * 2048 * 2;
        let total_bytes = total_vram_mb as usize * 1024 * 1024;
        let total_blocks = total_bytes / bytes_per_block;

        Self {
            block_size,
            total_blocks,
            allocations: std::collections::HashMap::new(),
            free_blocks: (0..total_blocks).collect(),
        }
    }

    /// Allocate blocks for a specialist's request
    pub fn allocate(&mut self, specialist: &str, num_tokens: usize) -> Option<Vec<usize>> {
        let blocks_needed = (num_tokens + self.block_size - 1) / self.block_size;
        if blocks_needed > self.free_blocks.len() {
            return None; // Not enough cache space
        }

        let allocated: Vec<usize> = self.free_blocks.drain(..blocks_needed).collect();
        self.allocations
            .entry(specialist.to_string())
            .or_default()
            .extend(&allocated);

        Some(allocated)
    }

    /// Free blocks for a specialist
    pub fn free(&mut self, specialist: &str) {
        if let Some(blocks) = self.allocations.remove(specialist) {
            self.free_blocks.extend(blocks);
        }
    }

    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        let used = self.total_blocks - self.free_blocks.len();
        used as f32 / self.total_blocks as f32
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_blocks: self.total_blocks,
            free_blocks: self.free_blocks.len(),
            specialists_cached: self.allocations.len(),
            utilization: self.utilization(),
        }
    }
}

pub struct CacheStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub specialists_cached: usize,
    pub utilization: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_allocation() {
        let mut cache = KvCache::new(100, 16); // 100MB, 16-token blocks
        assert!(cache.utilization() == 0.0);

        let blocks = cache.allocate("python_expert", 64);
        assert!(blocks.is_some());
        assert!(cache.utilization() > 0.0);

        cache.free("python_expert");
        assert!(cache.utilization() == 0.0);
    }
}
