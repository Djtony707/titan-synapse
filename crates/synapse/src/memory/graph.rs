use anyhow::Result;
use rusqlite::Connection;
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

    /// Get total facts count
    pub fn fact_count(&self) -> Result<u32> {
        let count: u32 = self.conn.lock().unwrap().query_row(
            "SELECT COUNT(*) FROM facts",
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
    fn test_preferences() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let kg = KnowledgeGraph::new(&db_path).unwrap();

        kg.add_preference("python_expert", "What is a list?", "good answer", "bad answer").unwrap();
        assert_eq!(kg.preference_count("python_expert").unwrap(), 1);
    }
}
