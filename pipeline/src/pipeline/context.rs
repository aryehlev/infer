use crate::{ModelInput, ModelOutput};
use polars::prelude::*;
/// Execution context for passing data between pipeline steps
use std::collections::HashMap;
use std::sync::Arc;

/// Data that can be stored in the execution context
#[derive(Clone, Debug)]
pub enum ContextData {
    /// Polars DataFrame
    DataFrame(DataFrame),
    /// Model output
    ModelOutput(ModelOutput),
    /// Model input
    ModelInput(ModelInput),
    /// String data
    String(String),
    /// Float value
    Float(f64),
    /// Integer value
    Int(i64),
    /// Boolean value
    Bool(bool),
    /// Vector of strings (e.g., for document retrieval)
    StringVec(Vec<String>),
    /// Vector of floats
    FloatVec(Vec<f64>),
    /// Custom data (type-erased)
    Custom(Arc<dyn std::any::Any + Send + Sync>),
}

impl ContextData {
    /// Try to extract a DataFrame
    pub fn as_dataframe(&self) -> Option<&DataFrame> {
        match self {
            ContextData::DataFrame(df) => Some(df),
            _ => None,
        }
    }

    /// Try to extract a ModelOutput
    pub fn as_model_output(&self) -> Option<&ModelOutput> {
        match self {
            ContextData::ModelOutput(output) => Some(output),
            _ => None,
        }
    }

    /// Try to extract a ModelInput
    pub fn as_model_input(&self) -> Option<&ModelInput> {
        match self {
            ContextData::ModelInput(input) => Some(input),
            _ => None,
        }
    }

    /// Try to extract a String
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ContextData::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract a Float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ContextData::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Try to extract an Int
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ContextData::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to extract a Bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ContextData::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract a StringVec
    pub fn as_string_vec(&self) -> Option<&Vec<String>> {
        match self {
            ContextData::StringVec(v) => Some(v),
            _ => None,
        }
    }

    /// Try to extract a FloatVec
    pub fn as_float_vec(&self) -> Option<&Vec<f64>> {
        match self {
            ContextData::FloatVec(v) => Some(v),
            _ => None,
        }
    }
}

/// Execution context that holds data shared between pipeline steps
#[derive(Clone, Debug)]
pub struct ExecutionContext {
    data: HashMap<String, ContextData>,
}

impl ExecutionContext {
    /// Create a new empty execution context
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Insert data into the context
    pub fn insert(&mut self, key: impl Into<String>, value: ContextData) {
        self.data.insert(key.into(), value);
    }

    /// Get data from the context
    pub fn get(&self, key: &str) -> Option<&ContextData> {
        self.data.get(key)
    }

    /// Remove data from the context
    pub fn remove(&mut self, key: &str) -> Option<ContextData> {
        self.data.remove(key)
    }

    /// Check if a key exists in the context
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get all keys in the context
    pub fn keys(&self) -> Vec<&String> {
        self.data.keys().collect()
    }

    /// Clear all data from the context
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get the number of items in the context
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the context is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_basic_operations() {
        let mut ctx = ExecutionContext::new();

        ctx.insert("value", ContextData::Float(42.0));
        assert!(ctx.contains("value"));
        assert_eq!(ctx.len(), 1);

        let data = ctx.get("value").unwrap();
        assert_eq!(data.as_float(), Some(42.0));

        ctx.remove("value");
        assert!(!ctx.contains("value"));
        assert_eq!(ctx.len(), 0);
    }

    #[test]
    fn test_context_data_types() {
        let mut ctx = ExecutionContext::new();

        ctx.insert("str", ContextData::String("hello".to_string()));
        ctx.insert("num", ContextData::Int(123));
        ctx.insert("flag", ContextData::Bool(true));

        assert_eq!(ctx.get("str").unwrap().as_string(), Some("hello"));
        assert_eq!(ctx.get("num").unwrap().as_int(), Some(123));
        assert_eq!(ctx.get("flag").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_context_clear() {
        let mut ctx = ExecutionContext::new();
        ctx.insert("a", ContextData::Int(1));
        ctx.insert("b", ContextData::Int(2));

        assert_eq!(ctx.len(), 2);
        ctx.clear();
        assert_eq!(ctx.len(), 0);
        assert!(ctx.is_empty());
    }
}
