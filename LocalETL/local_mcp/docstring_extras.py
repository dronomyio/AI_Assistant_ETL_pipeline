import functools
from typing import Any, Callable, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


def add_custom_node_examples(func: T) -> T:
    """Decorator to add custom workflow node examples to docstrings.
    
    This is used to enhance the documentation for functions that handle workflow node configurations.
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    return wrapper