use anyhow::Result;
use rusqlite::Connection;
use serde_json;
use std::path::Path;
use std::sync::Mutex;

/// SQLite-backed knowledge graph shared across all specialists
pub struct KnowledgeGraph {
    conn: Mutex<Connection>,
}

impl KnowledgeGraph {
    pub fn new(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;

        conn.execute_batch("
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
            CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate);

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                specialist TEXT,
                score REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);

            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                specialist TEXT NOT NULL,
                prompt TEXT NOT NULL,
                chosen TEXT NOT NULL,
                rejected TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS routing_pathways (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pathway TEXT NOT NULL UNIQUE,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0.0,
                last_used TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_pathway ON routing_pathways(pathway);

            CREATE TABLE IF NOT EXISTS specialist_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                specialist TEXT NOT NULL,
                domain TEXT NOT NULL,
                request_count INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0.0,
                avg_tok_per_sec REAL DEFAULT 0.0,
                last_used TEXT DEFAULT (datetime('now')),
                UNIQUE(specialist, domain)
            );
        ")?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Store a fact triple
    pub fn add_fact(&self, subject: &str, predicate: &str, object: &str, source: Option<&str>) -> Result<()> {
        self.conn.lock().unwrap().execute(
            "INSERT INTO facts (subject, predicate, object, source) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![subject, predicate, object, source],
        )?;
        Ok(())
    }

    /// Query facts about a subject
    pub fn query_facts(&self, subject: &str) -> Result<Vec<(String, String, f64)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT predicate, object, confidence FROM facts WHERE subject = ?1 ORDER BY confidence DESC"
        )?;

        let facts = stmt.query_map([subject], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

        Ok(facts)
    }

    /// Store a conversation message
    pub fn log_message(&self, session_id: &str, role: &str, content: &str, specialist: Option<&str>) -> Result<()> {
        self.conn.lock().unwrap().execute(
            "INSERT INTO conversations (session_id, role, content, specialist) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![session_id, role, content, specialist],
        )?;
        Ok(())
    }

    /// Store a preference pair for DPO training
    pub fn add_preference(&self, specialist: &str, prompt: &str, chosen: &str, rejected: &str) -> Result<()> {
        self.conn.lock().unwrap().execute(
            "INSERT INTO preferences (specialist, prompt, chosen, rejected) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![specialist, prompt, chosen, rejected],
        )?;
        Ok(())
    }

    /// Count preference pairs for a specialist
    pub fn preference_count(&self, specialist: &str) -> Result<u32> {
        let count: u32 = self.conn.lock().unwrap().query_row(
            "SELECT COUNT(*) FROM preferences WHERE specialist = ?1",
            [specialist],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Reinforce a routing pathway (Hebbian: pathways that fire together, wire together)
    pub fn reinforce_pathway(&self, specialists: &[String], score: f32) -> Result<()> {
        let pathway = specialists.join("+");
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO routing_pathways (pathway, success_count, avg_score, last_used)
             VALUES (?1, 1, ?2, datetime('now'))
             ON CONFLICT(pathway) DO UPDATE SET
                success_count = success_count + 1,
                avg_score = (avg_score * success_count + ?2) / (success_count + 1),
                last_used = datetime('now')",
            rusqlite::params![pathway, score],
        )?;
        Ok(())
    }

    /// Record a pathway failure
    pub fn weaken_pathway(&self, specialists: &[String]) -> Result<()> {
        let pathway = specialists.join("+");
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO routing_pathways (pathway, failure_count, last_used)
             VALUES (?1, 1, 0.0, datetime('now'))
             ON CONFLICT(pathway) DO UPDATE SET
                failure_count = failure_count + 1,
                last_used = datetime('now')",
            rusqlite::params![pathway],
        )?;
        Ok(())
    }

    /// Get pathway strength (success_count - failure_count, weighted by avg_score)
    pub fn pathway_strength(&self, specialists: &[String]) -> Result<f64> {
        let pathway = specialists.join("+");
        let conn = self.conn.lock().unwrap();
        let result: f64 = conn.query_row(
            "SELECT (success_count - failure_count) * avg_score FROM routing_pathways WHERE pathway = ?1",
            [&pathway],
            |row| row.get(0),
        ).unwrap_or(0.0);
        Ok(result)
    }

    /// Get top routing pathways by strength
    pub fn top_pathways(&self, limit: u32) -> Result<Vec<(String, i64, f64)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT pathway, (success_count - failure_count) as strength, avg_score
             FROM routing_pathways
             ORDER BY strength * avg_score DESC
             LIMIT ?1"
        )?;
        let results = stmt.query_map([limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    }

    /// Update specialist stats for a domain
    pub fn update_specialist_stats(
        &self,
        specialist: &str,
        domain: &str,
        score: f32,
        tok_per_sec: f64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO specialist_stats (specialist, domain, request_count, avg_score, avg_tok_per_sec, last_used)
             VALUES (?1, ?2, 1, ?3, ?4, datetime('now'))
             ON CONFLICT(specialist, domain) DO UPDATE SET
                request_count = request_count + 1,
                avg_score = (avg_score * request_count + ?3) / (request_count + 1),
                avg_tok_per_sec = (avg_tok_per_sec * request_count + ?4) / (request_count + 1),
                last_used = datetime('now')",
            rusqlite::params![specialist, domain, score, tok_per_sec],
        )?;
        Ok(())
    }

    /// Get total facts count
    pub fn fact_count(&self) -> Result<u32> {
        let count: u32 = self.conn.lock().unwrap().query_row(
            "SELECT COUNT(*) FROM facts",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Get total conversation messages count
    pub fn conversation_count(&self) -> Result<u32> {
        let count: u32 = self.conn.lock().unwrap().query_row(
            "SELECT COUNT(*) FROM conversations",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Metacognitive confidence report — per-specialist performance stats
    pub fn specialist_confidence_report(&self) -> Result<Vec<serde_json::Value>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT specialist, domain, request_count, avg_score, avg_tok_per_sec
             FROM specialist_stats ORDER BY avg_score DESC"
        )?;

        let results = stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "specialist": row.get::<_, String>(0)?,
                "domain": row.get::<_, String>(1)?,
                "requests": row.get::<_, i64>(2)?,
                "avg_score": row.get::<_, f64>(3)?,
                "avg_tok_per_sec": row.get::<_, f64>(4)?,
            }))
        })?.filter_map(|r| r.ok()).collect();

        Ok(results)
    }

    /// Get total preference pairs count
    pub fn total_preference_count(&self) -> Result<u32> {
        let count: u32 = self.conn.lock().unwrap().query_row(
            "SELECT COUNT(*) FROM preferences",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let kg = KnowledgeGraph::new(&db_path).unwrap();

        kg.add_fact("Python", "is_a", "programming language", Some("test")).unwrap();
        kg.add_fact("Python", "created_by", "Guido van Rossum", Some("test")).unwrap();

        let facts = kg.query_facts("Python").unwrap();
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_hebbian_routing() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let kg = KnowledgeGraph::new(&db_path).unwrap();

        // Reinforce a pathway multiple times
        let pathway = vec!["python_expert".to_string(), "reviewer".to_string()];
        kg.reinforce_pathway(&pathway, 4.5).unwrap();
        kg.reinforce_pathway(&pathway, 4.8).unwrap();

        let strength = kg.pathway_strength(&pathway).unwrap();
        assert!(strength > 0.0, "Pathway should have positive strength");

        // Check top pathways
        let top = kg.top_pathways(10).unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "python_expert+reviewer");
    }

    #[test]
    fn test_specialist_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let kg = KnowledgeGraph::new(&db_path).unwrap();

        kg.update_specialist_stats("python_expert", "coding", 4.5, 200.0).unwrap();
        kg.update_specialist_stats("python_expert", "coding", 4.8, 220.0).unwrap();

        // Should not error — stats are accumulated
        kg.update_specialist_stats("sql_expert", "database", 4.0, 180.0).unwrap();
    }

    #[test]
    fn test_preferences() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let kg = KnowledgeGraph::new(&db_path).unwrap();

        kg.add_preference("python_expert", "What is a list?", "good answer", "bad answer").unwrap();
        assert_eq!(kg.preference_count("python_expert").unwrap(), 1);
    }
}
