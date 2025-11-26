use crate::pipeline::{ContextData, ExecutionContext, PipelineStep, StepResult};
use crate::Result;
/// RAG (Retrieval-Augmented Generation) pipeline steps
use std::sync::Arc;

/// Step that retrieves documents based on a query
pub struct RetrievalStep {
    name: String,
    retriever: Arc<dyn DocumentRetriever>,
    query_key: String,
    output_key: String,
    top_k: usize,
}

/// Trait for document retrievers
pub trait DocumentRetriever: Send + Sync {
    /// Retrieve documents for a query
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<Document>>;
}

/// A retrieved document
#[derive(Debug, Clone)]
pub struct Document {
    /// Document ID
    pub id: String,
    /// Document content
    pub content: String,
    /// Relevance score (higher is better)
    pub score: f64,
    /// Optional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl RetrievalStep {
    /// Create a new retrieval step
    pub fn new(
        name: impl Into<String>,
        retriever: Arc<dyn DocumentRetriever>,
        query_key: impl Into<String>,
        output_key: impl Into<String>,
        top_k: usize,
    ) -> Self {
        Self {
            name: name.into(),
            retriever,
            query_key: query_key.into(),
            output_key: output_key.into(),
            top_k,
        }
    }
}

impl PipelineStep for RetrievalStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get query from context
        let query_data = ctx.get(&self.query_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Query key '{}' not found in context",
                self.query_key
            ))
        })?;

        let query = query_data.as_string().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Query key '{}' is not a string",
                self.query_key
            ))
        })?;

        // Retrieve documents
        let documents = self.retriever.retrieve(query, self.top_k)?;

        // Store documents as a vector of strings (content only)
        let contents: Vec<String> = documents.iter().map(|doc| doc.content.clone()).collect();

        ctx.insert(&self.output_key, ContextData::StringVec(contents));

        // Also store full documents in a separate key for potential reranking
        let docs_key = format!("{}_full", self.output_key);
        ctx.insert(docs_key, ContextData::Custom(Arc::new(documents)));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Retrieves documents based on a query")
    }
}

/// Step that ranks/reranks documents
pub struct RankingStep {
    name: String,
    ranker: Arc<dyn DocumentRanker>,
    query_key: String,
    documents_key: String,
    output_key: String,
    top_k: Option<usize>,
}

/// Trait for document rankers
pub trait DocumentRanker: Send + Sync {
    /// Rank documents for a query
    fn rank(&self, query: &str, documents: &[Document]) -> Result<Vec<Document>>;
}

impl RankingStep {
    /// Create a new ranking step
    pub fn new(
        name: impl Into<String>,
        ranker: Arc<dyn DocumentRanker>,
        query_key: impl Into<String>,
        documents_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            ranker,
            query_key: query_key.into(),
            documents_key: documents_key.into(),
            output_key: output_key.into(),
            top_k: None,
        }
    }

    /// Set the number of top documents to keep
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

impl PipelineStep for RankingStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get query from context
        let query_data = ctx.get(&self.query_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Query key '{}' not found in context",
                self.query_key
            ))
        })?;

        let query = query_data.as_string().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Query key '{}' is not a string",
                self.query_key
            ))
        })?;

        // Get documents from context
        let docs_key = format!("{}_full", self.documents_key);
        let docs_data = ctx.get(&docs_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Documents key '{}' not found in context",
                docs_key
            ))
        })?;

        let documents = match docs_data {
            ContextData::Custom(data) => data.downcast_ref::<Vec<Document>>().ok_or_else(|| {
                infer_lib::InferError::InvalidInput(
                    "Documents data is not the correct type".to_string(),
                )
            })?,
            _ => {
                return Err(infer_lib::InferError::InvalidInput(
                    "Documents data is not Custom type".to_string(),
                ))
            }
        };

        // Rank documents
        let mut ranked = self.ranker.rank(query, documents)?;

        // Optionally limit to top_k
        if let Some(k) = self.top_k {
            ranked.truncate(k);
        }

        // Store ranked documents
        let contents: Vec<String> = ranked.iter().map(|doc| doc.content.clone()).collect();

        ctx.insert(&self.output_key, ContextData::StringVec(contents));

        // Store full ranked documents
        let docs_key = format!("{}_full", self.output_key);
        ctx.insert(docs_key, ContextData::Custom(Arc::new(ranked)));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Ranks/reranks documents")
    }
}

/// Step that constructs a prompt from query and documents
pub struct PromptConstructionStep {
    name: String,
    query_key: String,
    documents_key: String,
    output_key: String,
    template: String,
}

impl PromptConstructionStep {
    /// Create a new prompt construction step
    ///
    /// Template can contain {query} and {documents} placeholders
    pub fn new(
        name: impl Into<String>,
        query_key: impl Into<String>,
        documents_key: impl Into<String>,
        output_key: impl Into<String>,
        template: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            query_key: query_key.into(),
            documents_key: documents_key.into(),
            output_key: output_key.into(),
            template: template.into(),
        }
    }
}

impl PipelineStep for PromptConstructionStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get query from context
        let query_data = ctx.get(&self.query_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Query key '{}' not found in context",
                self.query_key
            ))
        })?;

        let query = query_data.as_string().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Query key '{}' is not a string",
                self.query_key
            ))
        })?;

        // Get documents from context
        let docs_data = ctx.get(&self.documents_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Documents key '{}' not found in context",
                self.documents_key
            ))
        })?;

        let documents = docs_data.as_string_vec().ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Documents key '{}' is not a string vector",
                self.documents_key
            ))
        })?;

        // Construct documents string
        let docs_str = documents.join("\n\n");

        // Replace placeholders in template
        let prompt = self
            .template
            .replace("{query}", query)
            .replace("{documents}", &docs_str);

        // Store prompt
        ctx.insert(&self.output_key, ContextData::String(prompt));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Constructs a prompt from query and documents")
    }
}

/// Step that filters documents by score threshold
pub struct ScoreFilterStep {
    name: String,
    documents_key: String,
    output_key: String,
    threshold: f64,
}

impl ScoreFilterStep {
    /// Create a new score filter step
    pub fn new(
        name: impl Into<String>,
        documents_key: impl Into<String>,
        output_key: impl Into<String>,
        threshold: f64,
    ) -> Self {
        Self {
            name: name.into(),
            documents_key: documents_key.into(),
            output_key: output_key.into(),
            threshold,
        }
    }
}

impl PipelineStep for ScoreFilterStep {
    fn execute(&self, ctx: &mut ExecutionContext) -> Result<StepResult> {
        // Get documents from context
        let docs_key = format!("{}_full", self.documents_key);
        let docs_data = ctx.get(&docs_key).ok_or_else(|| {
            infer_lib::InferError::InvalidInput(format!(
                "Documents key '{}' not found in context",
                docs_key
            ))
        })?;

        let documents = match docs_data {
            ContextData::Custom(data) => data.downcast_ref::<Vec<Document>>().ok_or_else(|| {
                infer_lib::InferError::InvalidInput(
                    "Documents data is not the correct type".to_string(),
                )
            })?,
            _ => {
                return Err(infer_lib::InferError::InvalidInput(
                    "Documents data is not Custom type".to_string(),
                ))
            }
        };

        // Filter by threshold
        let filtered: Vec<Document> = documents
            .iter()
            .filter(|doc| doc.score >= self.threshold)
            .cloned()
            .collect();

        // Store filtered documents
        let contents: Vec<String> = filtered.iter().map(|doc| doc.content.clone()).collect();

        ctx.insert(&self.output_key, ContextData::StringVec(contents));

        // Store full filtered documents
        let docs_key = format!("{}_full", self.output_key);
        ctx.insert(docs_key, ContextData::Custom(Arc::new(filtered)));

        Ok(StepResult::Continue)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("Filters documents by score threshold")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ExecutionContext;

    struct MockRetriever;

    impl DocumentRetriever for MockRetriever {
        fn retrieve(&self, _query: &str, top_k: usize) -> Result<Vec<Document>> {
            Ok((0..top_k)
                .map(|i| Document {
                    id: format!("doc{}", i),
                    content: format!("Content {}", i),
                    score: 1.0 - (i as f64 * 0.1),
                    metadata: Default::default(),
                })
                .collect())
        }
    }

    #[test]
    fn test_retrieval_step() {
        let step = RetrievalStep::new("retrieve", Arc::new(MockRetriever), "query", "documents", 3);

        let mut ctx = ExecutionContext::new();
        ctx.insert("query", ContextData::String("test query".to_string()));

        step.execute(&mut ctx).unwrap();

        let docs = ctx.get("documents").unwrap().as_string_vec().unwrap();
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0], "Content 0");
    }

    #[test]
    fn test_prompt_construction_step() {
        let step = PromptConstructionStep::new(
            "construct",
            "query",
            "documents",
            "prompt",
            "Query: {query}\n\nDocuments:\n{documents}",
        );

        let mut ctx = ExecutionContext::new();
        ctx.insert("query", ContextData::String("What is Rust?".to_string()));
        ctx.insert(
            "documents",
            ContextData::StringVec(vec![
                "Rust is a systems programming language.".to_string(),
                "It focuses on safety and performance.".to_string(),
            ]),
        );

        step.execute(&mut ctx).unwrap();

        let prompt = ctx.get("prompt").unwrap().as_string().unwrap();
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Rust is a systems programming language."));
        assert!(prompt.contains("It focuses on safety and performance."));
    }

    #[test]
    fn test_score_filter_step() {
        let documents = vec![
            Document {
                id: "1".to_string(),
                content: "High score doc".to_string(),
                score: 0.9,
                metadata: Default::default(),
            },
            Document {
                id: "2".to_string(),
                content: "Low score doc".to_string(),
                score: 0.3,
                metadata: Default::default(),
            },
        ];

        let mut ctx = ExecutionContext::new();
        ctx.insert("docs_full", ContextData::Custom(Arc::new(documents)));

        let step = ScoreFilterStep::new("filter", "docs", "filtered", 0.5);
        step.execute(&mut ctx).unwrap();

        let filtered = ctx.get("filtered").unwrap().as_string_vec().unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], "High score doc");
    }
}
