#!/usr/bin/env python3
"""
Phoenix Observer for AI Agent ETL Pipeline

This module integrates Arize Phoenix for observability into the ETL pipeline,
allowing for monitoring, evaluating and troubleshooting of the LLM components.
"""
import os
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

try:
    from phoenix import trace, Trace, TraceLogOptions, SpanLogOptions
    from phoenix.session import SpanSessionOptions, TraceSession
    from phoenix.trace.spans import LLMSpanFields, SpanStatusType
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    print("Arize Phoenix not installed. Install with: pip install arize-phoenix")

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phoenix-observer")

class PhoenixObserver:
    """
    Observer class that integrates Arize Phoenix observability into the ETL pipeline.
    """
    def __init__(
        self, 
        enabled: bool = True,
        phoenix_server_url: Optional[str] = None,
        app_name: str = "ai-agent-etl-pipeline",
        env: str = "development"
    ):
        """
        Initialize the Phoenix Observer.
        
        Args:
            enabled: Whether Phoenix observability is enabled
            phoenix_server_url: URL of the Phoenix server, or None for local
            app_name: Name of the application for Phoenix
            env: Environment name (development, staging, production)
        """
        self.enabled = enabled and PHOENIX_AVAILABLE
        
        if not self.enabled:
            if not PHOENIX_AVAILABLE:
                logger.warning("Arize Phoenix not installed. Install with: pip install arize-phoenix")
            return
        
        self.phoenix_server_url = phoenix_server_url
        self.app_name = app_name
        self.env = env
        self.active_traces = {}
        
        # Set up Phoenix configuration
        span_options = SpanLogOptions(
            console=False,
            openinference=True
        )
        
        trace_options = TraceLogOptions(
            console=False,
            openinference=True,
            spans=span_options
        )
        
        session_options = SpanSessionOptions(
            app_name=self.app_name,
            env=self.env,
            server_url=self.phoenix_server_url
        )
        
        self.session = TraceSession(options=session_options)
        self.trace_options = trace_options
        
        logger.info(f"Phoenix Observer initialized for app '{app_name}' in '{env}' environment")
        if phoenix_server_url:
            logger.info(f"Using Phoenix server at {phoenix_server_url}")
        else:
            logger.info("Using local Phoenix data storage")

    def start_trace(
        self, 
        trace_id: Optional[str] = None, 
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new trace for the ETL pipeline process.
        
        Args:
            trace_id: Optional custom trace ID, or auto-generated if None
            user_id: Optional user ID associated with this trace
            metadata: Optional metadata to attach to the trace
            
        Returns:
            The trace ID
        """
        if not self.enabled:
            return str(uuid.uuid4())
        
        if trace_id is None:
            trace_id = str(uuid.uuid4())
            
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Add standard metadata
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "environment": self.env,
        })
        
        if user_id:
            metadata["user_id"] = user_id
            
        # Create the trace
        trace_obj = Trace(id=trace_id, options=self.trace_options)
        self.active_traces[trace_id] = trace_obj
        
        # Start root span for the ETL process
        with trace_obj.span(
            name="etl_pipeline",
            metadata=metadata
        ) as span:
            span.set_status(SpanStatusType.UNSET)
            
        logger.info(f"Started trace {trace_id}")
        return trace_id
        
    def log_extraction_span(
        self, 
        trace_id: str,
        source_type: str,
        source_details: Dict[str, Any],
        num_files: int,
        duration_ms: Optional[float] = None,
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Log an extraction step to Phoenix.
        
        Args:
            trace_id: The trace ID
            source_type: Type of source (local_dir, gdrive, etc.)
            source_details: Details about the source
            num_files: Number of files extracted
            duration_ms: Duration in milliseconds
            status: Status of the extraction (success, error)
            error: Error message if status is error
        """
        if not self.enabled or trace_id not in self.active_traces:
            return
            
        trace_obj = self.active_traces[trace_id]
        
        metadata = {
            "source_type": source_type,
            "source_details": json.dumps(source_details),
            "num_files": num_files
        }
        
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
            
        with trace_obj.span(
            name="extraction",
            parent="etl_pipeline",
            metadata=metadata
        ) as span:
            if status == "success":
                span.set_status(SpanStatusType.OK)
            else:
                span.set_status(SpanStatusType.ERROR)
                if error:
                    span.record_exception(Exception(error))
                    
        logger.info(f"Logged extraction span for trace {trace_id}")
    
    def log_transformation_span(
        self,
        trace_id: str,
        transformation_type: str,
        file_path: str,
        num_elements: int,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        duration_ms: Optional[float] = None,
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Log a transformation step to Phoenix.
        
        Args:
            trace_id: The trace ID
            transformation_type: Type of transformation (text_chunking, embedding, etc.)
            file_path: Path to the file being transformed
            num_elements: Number of elements generated
            chunk_size: Size of chunks if applicable
            chunk_overlap: Overlap size if applicable
            duration_ms: Duration in milliseconds
            status: Status of the transformation (success, error)
            error: Error message if status is error
        """
        if not self.enabled or trace_id not in self.active_traces:
            return
            
        trace_obj = self.active_traces[trace_id]
        
        metadata = {
            "transformation_type": transformation_type,
            "file_path": file_path,
            "num_elements": num_elements
        }
        
        if chunk_size is not None:
            metadata["chunk_size"] = chunk_size
            
        if chunk_overlap is not None:
            metadata["chunk_overlap"] = chunk_overlap
            
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
            
        with trace_obj.span(
            name="transformation",
            parent="etl_pipeline",
            metadata=metadata
        ) as span:
            if status == "success":
                span.set_status(SpanStatusType.OK)
            else:
                span.set_status(SpanStatusType.ERROR)
                if error:
                    span.record_exception(Exception(error))
                    
        logger.info(f"Logged transformation span for trace {trace_id}")
    
    def log_embedding_span(
        self,
        trace_id: str,
        embedding_model: str,
        text_or_image: str,
        embedding_type: str = "text",  # text or image
        embedding_dim: Optional[int] = None,
        duration_ms: Optional[float] = None,
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Log an embedding generation step to Phoenix.
        
        Args:
            trace_id: The trace ID
            embedding_model: Name of the embedding model
            text_or_image: Brief description or preview of content
            embedding_type: Type of embedding (text or image)
            embedding_dim: Dimensionality of embedding
            duration_ms: Duration in milliseconds
            status: Status of the embedding generation (success, error)
            error: Error message if status is error
        """
        if not self.enabled or trace_id not in self.active_traces:
            return
            
        trace_obj = self.active_traces[trace_id]
        
        # Truncate text preview if needed
        if embedding_type == "text" and len(text_or_image) > 100:
            text_preview = text_or_image[:97] + "..."
        else:
            text_preview = text_or_image
            
        metadata = {
            "embedding_model": embedding_model,
            "embedding_type": embedding_type,
            "text_preview": text_preview
        }
        
        if embedding_dim is not None:
            metadata["embedding_dim"] = embedding_dim
            
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
        
        # For embedding generation, use LLM span to better integrate with Phoenix UI
        with trace_obj.span(
            name="embedding_generation",
            parent="transformation",
            metadata=metadata
        ) as span:
            # Set LLM-specific fields
            span.set_llm_fields(
                LLMSpanFields(
                    model=embedding_model,
                    input=text_preview,
                    output="vector embedding" # We don't log actual embeddings due to size
                )
            )
            
            if status == "success":
                span.set_status(SpanStatusType.OK)
            else:
                span.set_status(SpanStatusType.ERROR)
                if error:
                    span.record_exception(Exception(error))
                    
        logger.info(f"Logged embedding span for trace {trace_id}")
    
    def log_loading_span(
        self,
        trace_id: str,
        destination_type: str,
        destination_details: Dict[str, Any],
        num_elements: int,
        duration_ms: Optional[float] = None,
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Log a loading step to Phoenix.
        
        Args:
            trace_id: The trace ID
            destination_type: Type of destination (local_dir, mongodb, weaviate, etc.)
            destination_details: Details about the destination
            num_elements: Number of elements loaded
            duration_ms: Duration in milliseconds
            status: Status of the loading (success, error)
            error: Error message if status is error
        """
        if not self.enabled or trace_id not in self.active_traces:
            return
            
        trace_obj = self.active_traces[trace_id]
        
        metadata = {
            "destination_type": destination_type,
            "destination_details": json.dumps(destination_details),
            "num_elements": num_elements
        }
        
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
            
        with trace_obj.span(
            name="loading",
            parent="etl_pipeline",
            metadata=metadata
        ) as span:
            if status == "success":
                span.set_status(SpanStatusType.OK)
            else:
                span.set_status(SpanStatusType.ERROR)
                if error:
                    span.record_exception(Exception(error))
                    
        logger.info(f"Logged loading span for trace {trace_id}")
    
    def log_search_span(
        self,
        trace_id: str,
        query: str,
        search_type: str,  # text, image, combined
        num_results: int,
        duration_ms: Optional[float] = None,
        results_preview: Optional[List[Dict[str, Any]]] = None,
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Log a search operation to Phoenix.
        
        Args:
            trace_id: The trace ID
            query: The search query
            search_type: Type of search (text, image, combined)
            num_results: Number of results returned
            duration_ms: Duration in milliseconds
            results_preview: Preview of the results (truncated)
            status: Status of the search (success, error)
            error: Error message if status is error
        """
        if not self.enabled or trace_id not in self.active_traces:
            return
            
        trace_obj = self.active_traces[trace_id]
        
        metadata = {
            "query": query,
            "search_type": search_type,
            "num_results": num_results
        }
        
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
            
        if results_preview:
            # Limit the preview to first 3 results
            preview = results_preview[:3]
            # Truncate long text fields to avoid overwhelming the logs
            for result in preview:
                if "text" in result and len(result["text"]) > 100:
                    result["text"] = result["text"][:97] + "..."
            metadata["results_preview"] = json.dumps(preview)
        
        # Log as LLM span since it involves vector similarity
        with trace_obj.span(
            name="vector_search",
            parent="etl_pipeline",
            metadata=metadata
        ) as span:
            # Set LLM-specific fields
            span.set_llm_fields(
                LLMSpanFields(
                    model="vector_search",
                    input=query,
                    output=f"Found {num_results} results" if results_preview is None else json.dumps(preview)
                )
            )
            
            if status == "success":
                span.set_status(SpanStatusType.OK)
            else:
                span.set_status(SpanStatusType.ERROR)
                if error:
                    span.record_exception(Exception(error))
                    
        logger.info(f"Logged search span for trace {trace_id}")
    
    def end_trace(self, trace_id: str, status: str = "success", error: Optional[str] = None):
        """
        End a trace and flush it to Phoenix.
        
        Args:
            trace_id: The trace ID
            status: Final status of the trace (success, error)
            error: Error message if status is error
        """
        if not self.enabled or trace_id not in self.active_traces:
            return
            
        trace_obj = self.active_traces[trace_id]
        
        # Update the root span status
        with trace_obj.span(
            name="etl_pipeline",
            metadata={"status": status}
        ) as span:
            if status == "success":
                span.set_status(SpanStatusType.OK)
            else:
                span.set_status(SpanStatusType.ERROR)
                if error:
                    span.record_exception(Exception(error))
        
        # Finalize the trace
        trace_obj.close()
        
        # Remove from active traces
        del self.active_traces[trace_id]
        
        logger.info(f"Ended trace {trace_id} with status {status}")

# Example usage
def example_usage():
    """Example of how to use the Phoenix Observer with the ETL pipeline."""
    # Create observer
    observer = PhoenixObserver(enabled=True)
    
    # Start a trace for the ETL process
    trace_id = observer.start_trace(user_id="example_user")
    
    try:
        # Log extraction
        start_time = time.time()
        # Simulate extraction process
        time.sleep(0.5)
        observer.log_extraction_span(
            trace_id=trace_id,
            source_type="local_dir",
            source_details={"path": "/app/data"},
            num_files=10,
            duration_ms=(time.time() - start_time) * 1000
        )
        
        # Log transformation for a file
        for i in range(3):
            start_time = time.time()
            # Simulate transformation process
            time.sleep(0.3)
            observer.log_transformation_span(
                trace_id=trace_id,
                transformation_type="text_chunking",
                file_path=f"/app/data/file_{i}.txt",
                num_elements=5,
                chunk_size=1000,
                chunk_overlap=200,
                duration_ms=(time.time() - start_time) * 1000
            )
            
            # Log embedding generation
            start_time = time.time()
            # Simulate embedding generation
            time.sleep(0.2)
            observer.log_embedding_span(
                trace_id=trace_id,
                embedding_model="all-MiniLM-L6-v2",
                text_or_image=f"This is sample text from file_{i}.txt",
                embedding_type="text",
                embedding_dim=384,
                duration_ms=(time.time() - start_time) * 1000
            )
        
        # Log loading
        start_time = time.time()
        # Simulate loading process
        time.sleep(0.5)
        observer.log_loading_span(
            trace_id=trace_id,
            destination_type="weaviate",
            destination_details={"url": "http://localhost:8080"},
            num_elements=15,
            duration_ms=(time.time() - start_time) * 1000
        )
        
        # Log search
        start_time = time.time()
        # Simulate search process
        time.sleep(0.3)
        observer.log_search_span(
            trace_id=trace_id,
            query="How to configure a camera",
            search_type="text",
            num_results=3,
            results_preview=[
                {"text": "Camera configuration guide", "type": "Title"},
                {"text": "To configure your camera, first access the settings menu...", "type": "Text"},
                {"text": "Advanced settings include aperture, ISO and shutter speed...", "type": "Text"}
            ],
            duration_ms=(time.time() - start_time) * 1000
        )
        
        # End trace successfully
        observer.end_trace(trace_id, status="success")
        
    except Exception as e:
        # Log error and end trace with error
        observer.end_trace(trace_id, status="error", error=str(e))
        raise
    
if __name__ == "__main__":
    # Run example if executed directly
    example_usage()