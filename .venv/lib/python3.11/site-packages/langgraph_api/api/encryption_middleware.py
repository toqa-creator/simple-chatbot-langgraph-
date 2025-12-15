"""Encryption/decryption middleware for API layer.

This module provides helpers to encrypt data before storing and decrypt
after retrieving, keeping encryption logic at the API layer.
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any

import orjson
import structlog
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request  # noqa: TC002

from langgraph_api.config import LANGGRAPH_ENCRYPTION
from langgraph_api.encryption.context import (
    get_encryption_context,
    set_encryption_context,
)
from langgraph_api.encryption.custom import (
    ModelType,
    get_encryption_instance,
)
from langgraph_api.serde import Fragment, json_loads

if TYPE_CHECKING:
    from langgraph_sdk import Encryption

# Only import EncryptionContext at module load if encryption is configured
# This avoids requiring langgraph-sdk>=0.2.14 for users who don't use encryption
if LANGGRAPH_ENCRYPTION:
    from langgraph_sdk import EncryptionContext

logger = structlog.stdlib.get_logger(__name__)

ENCRYPTION_CONTEXT_KEY = "langchain.dev/encryption_context/v1"


def extract_encryption_context(request: Request) -> dict[str, Any]:
    """Extract encryption context from X-Encryption-Context header.

    Args:
        request: The Starlette request object

    Returns:
        Encryption context dict, or empty dict if header not present

    Raises:
        HTTPException: 422 if header is present but malformed
    """
    header_value = request.headers.get("X-Encryption-Context")
    if not header_value:
        return {}

    try:
        decoded = base64.b64decode(header_value.encode())
        context = orjson.loads(decoded)
        if not isinstance(context, dict):
            raise HTTPException(
                status_code=422,
                detail="Invalid X-Encryption-Context header: expected base64-encoded JSON object",
            )
        return context
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid X-Encryption-Context header: {e}",
        ) from e


class EncryptionContextMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and set encryption context from request headers.

    If a @encryption.context handler is registered, it is called after extracting
    the initial context from the X-Encryption-Context header. The handler receives
    the authenticated user and can derive encryption context from auth (e.g., JWT claims).
    """

    async def dispatch(self, request: Request, call_next):
        context_dict = extract_encryption_context(request)

        # Call context handler if registered (to derive context from auth)
        encryption_instance = get_encryption_instance()
        if encryption_instance and encryption_instance._context_handler:
            user = request.scope.get("user")
            if user:
                initial_ctx = EncryptionContext(
                    model=None, field=None, metadata=context_dict
                )
                try:
                    context_dict = await encryption_instance._context_handler(
                        user, initial_ctx
                    )
                except Exception as e:
                    await logger.aexception(
                        "Error in encryption context handler", exc_info=e
                    )

        set_encryption_context(context_dict)
        request.state.encryption_context = context_dict
        response = await call_next(request)
        return response


async def encrypt_json_if_needed(
    data: dict[str, Any] | None,
    encryption_instance: Encryption | None,
    model_type: ModelType,
    field: str | None = None,
) -> dict[str, Any] | None:
    """Encrypt JSON data dict if encryption is configured.

    Args:
        data: The plaintext data dict
        encryption_instance: The encryption instance (or None if no encryption)
        model_type: The type of model (e.g., "thread", "assistant", "run")
        field: The specific field being encrypted (e.g., "metadata", "context")

    Returns:
        Encrypted data dict with stored context, or original if no encryption configured
    """
    if data is None or encryption_instance is None:
        return data

    encryptor = encryption_instance.get_json_encryptor(model_type)
    if encryptor is None:
        return data

    context_dict = get_encryption_context()

    ctx = EncryptionContext(model=model_type, field=field, metadata=context_dict)
    encrypted = await encryptor(ctx, data)

    # Always store the context marker when encrypting (even if context is empty)
    # This marker is used during decryption to know if data was encrypted
    if encrypted is not None and isinstance(encrypted, dict):
        encrypted[ENCRYPTION_CONTEXT_KEY] = orjson.dumps(context_dict).decode()

    await logger.adebug(
        "Encrypted JSON data",
        model_type=model_type,
        field=field,
        context_stored=bool(context_dict),
    )
    return encrypted


async def decrypt_json_if_needed(
    data: dict[str, Any] | None,
    encryption_instance: Encryption | None,
    model_type: ModelType,
    field: str | None = None,
) -> dict[str, Any] | None:
    """Decrypt JSON data dict if encryption is configured and data was encrypted.

    Only calls the decryptor if the data contains the ENCRYPTION_CONTEXT_KEY marker,
    which indicates it was encrypted. This ensures plaintext data passes through
    unchanged, which is important during mixed-state scenarios where some endpoints
    encrypt on write but others don't yet.

    Args:
        data: The data dict (encrypted or plaintext)
        encryption_instance: The encryption instance (or None if no encryption)
        model_type: The type of model (e.g., "thread", "assistant", "run")
        field: The specific field being decrypted (e.g., "metadata", "context")

    Returns:
        Decrypted data dict (without reserved key), or original if not encrypted
    """
    if data is None or encryption_instance is None:
        return data

    # Only decrypt if data was actually encrypted (has the context marker)
    if ENCRYPTION_CONTEXT_KEY not in data:
        return data

    decryptor = encryption_instance.get_json_decryptor(model_type)
    if decryptor is None:
        return data

    context_dict = {}
    try:
        context_dict = orjson.loads(data[ENCRYPTION_CONTEXT_KEY])
    except Exception as e:
        await logger.awarning(
            "Failed to parse stored encryption context",
            error=str(e),
            model_type=model_type,
            field=field,
        )
    # Remove key before passing to user's decryptor to avoid duplication
    # (context is already passed via ctx.metadata)
    data = {k: v for k, v in data.items() if k != ENCRYPTION_CONTEXT_KEY}

    ctx = EncryptionContext(model=model_type, field=field, metadata=context_dict)
    decrypted = await decryptor(ctx, data)

    # Ensure reserved key is removed from output (in case decryptor didn't handle it)
    if ENCRYPTION_CONTEXT_KEY in decrypted:
        decrypted = {k: v for k, v in decrypted.items() if k != ENCRYPTION_CONTEXT_KEY}

    await logger.adebug(
        "Decrypted JSON data",
        model_type=model_type,
        field=field,
        context_retrieved=bool(context_dict),
    )
    return decrypted


async def _decrypt_field(
    obj: dict[str, Any],
    field_name: str,
    encryption_instance: Encryption,
    model_type: ModelType,
) -> tuple[str, Any]:
    """Decrypt a single field, returning (field_name, decrypted_value).

    Special handling for 'config' field: also decrypts config['configurable']
    (synced from context by the ops layer) and config['metadata'] (merged from
    assistant/thread/run metadata).

    Special handling for 'kwargs' field: also decrypts kwargs['input'],
    kwargs['config'], and kwargs['context'] since these are encrypted
    separately during run creation.

    Returns (field_name, None) if field doesn't exist or is falsy.
    """
    if not obj.get(field_name):
        return (field_name, obj.get(field_name))

    value = obj[field_name]
    # Database fields come back as either:
    # - dict: already parsed JSONB (psycopg JSON adapter)
    # - bytes/bytearray/memoryview/str: raw JSON to parse (psycopg binary mode)
    # - Fragment: wrapper around bytes (used by serde layer)
    if isinstance(value, dict):
        pass  # already parsed
    elif isinstance(value, (bytes, bytearray, memoryview, str, Fragment)):
        value = json_loads(value)
    else:
        raise TypeError(
            f"Cannot decrypt field '{field_name}': expected dict or JSON-serialized "
            f"bytes/str, got {type(value).__name__}"
        )

    decrypted = await decrypt_json_if_needed(
        value, encryption_instance, model_type, field=field_name
    )

    # Special case: config['configurable'] is synced from context by the ops layer,
    # so it may contain encrypted data that needs decryption.
    # config['metadata'] is merged from assistant/thread/run metadata, so also needs decryption.
    if field_name == "config" and isinstance(decrypted, dict):
        configurable = decrypted.get("configurable")
        if configurable and isinstance(configurable, dict):
            decrypted["configurable"] = await decrypt_json_if_needed(
                configurable, encryption_instance, model_type, field="context"
            )
        metadata = decrypted.get("metadata")
        if metadata and isinstance(metadata, dict):
            decrypted["metadata"] = await decrypt_json_if_needed(
                metadata, encryption_instance, model_type, field="metadata"
            )

    # Special case: kwargs contains encrypted subfields (input, config, context, command)
    # that were encrypted separately during run creation.
    if field_name == "kwargs" and isinstance(decrypted, dict):
        for subfield in ["input", "config", "context", "command"]:
            if subfield in decrypted:
                _, decrypted[subfield] = await _decrypt_field(
                    decrypted, subfield, encryption_instance, model_type
                )

    return (field_name, decrypted)


async def _decrypt_object(
    obj: dict[str, Any],
    model_type: ModelType,
    fields: list[str],
    encryption_instance: Encryption,
) -> None:
    """Decrypt all specified fields in a single object (in parallel).

    Only processes fields that exist in the object to avoid adding new fields.
    """
    results = await asyncio.gather(
        *[
            _decrypt_field(obj, f, encryption_instance, model_type)
            for f in fields
            if f in obj
        ]
    )
    for field_name, value in results:
        obj[field_name] = value


async def decrypt_response(
    obj: dict[str, Any],
    model_type: ModelType,
    fields: list[str],
    encryption_instance: Encryption | None = None,
) -> dict[str, Any]:
    """Decrypt specified fields in a response object (from database).

    IMPORTANT: This function only parses and decrypts fields when encryption is
    enabled. When encryption is disabled, fields are returned unchanged (as bytes
    if that's how they came from the DB). This is intentional: some fields can be
    very large, and we want to avoid parsing overhead when the bytes can be passed
    through directly to the response. Callers that need parsed dicts regardless of
    encryption state should use json_loads() on the fields they need to inspect.

    When encryption IS enabled, this parses bytes/memoryview/Fragment to dicts
    before decryption.

    Special handling: When 'config' is in fields, also decrypts config['configurable']
    since the ops layer syncs context into config['configurable'].

    Args:
        obj: Single dict from database (fields may be bytes or already-parsed dicts)
        model_type: Type identifier passed to EncryptionContext.model (e.g., "run", "cron", "thread")
        fields: List of field names to decrypt (e.g., ["metadata", "kwargs"])
        encryption_instance: Optional encryption instance (auto-fetched if None)

    Returns:
        The same dict, with fields decrypted (and parsed) only if encryption is enabled
    """
    if encryption_instance is None:
        encryption_instance = get_encryption_instance()
        if encryption_instance is None:
            return obj

    await _decrypt_object(obj, model_type, fields, encryption_instance)
    return obj


async def decrypt_responses(
    objects: list[dict[str, Any]],
    model_type: ModelType,
    fields: list[str],
    encryption_instance: Encryption | None = None,
) -> list[dict[str, Any]]:
    """Decrypt specified fields in multiple response objects (from database).

    IMPORTANT: This function only parses and decrypts fields when encryption is
    enabled. When encryption is disabled, fields are returned unchanged (as bytes
    if that's how they came from the DB). This is intentional: some fields can be
    very large, and we want to avoid parsing overhead when the bytes can be passed
    through directly to the response. Callers that need parsed dicts regardless of
    encryption state should use json_loads() on the fields they need to inspect.

    When encryption IS enabled, this parses bytes/memoryview/Fragment to dicts
    before decryption.

    Special handling: When 'config' is in fields, also decrypts config['configurable']
    since the ops layer syncs context into config['configurable'].

    Args:
        objects: List of dicts from database (fields may be bytes or already-parsed dicts)
        model_type: Type identifier passed to EncryptionContext.model (e.g., "run", "cron", "thread")
        fields: List of field names to decrypt (e.g., ["metadata", "kwargs"])
        encryption_instance: Optional encryption instance (auto-fetched if None)

    Returns:
        The same list, with fields decrypted (and parsed) only if encryption is enabled
    """
    if encryption_instance is None:
        encryption_instance = get_encryption_instance()
        if encryption_instance is None:
            return objects

    await asyncio.gather(
        *[
            _decrypt_object(obj, model_type, fields, encryption_instance)
            for obj in objects
        ]
    )
    return objects


async def _encrypt_field(
    data: dict[str, Any],
    field_name: str,
    encryption_instance: Encryption,
    model_type: ModelType,
) -> tuple[str, Any]:
    """Encrypt a single field, returning (field_name, encrypted_value).

    Special handling for 'config' field: also encrypts config['configurable']
    since the ops layer syncs config['configurable'] to context.

    Returns (field_name, None) if field doesn't exist or is None.
    """
    if field_name not in data or data[field_name] is None:
        return (field_name, data.get(field_name))

    encrypted = await encrypt_json_if_needed(
        data[field_name],
        encryption_instance,
        model_type,
        field=field_name,
    )

    # Special case: config['configurable'] will be synced to context by the ops layer,
    # so it needs to be encrypted with field="context" to match decryption.
    if field_name == "config" and isinstance(encrypted, dict):
        configurable = encrypted.get("configurable")
        if configurable and isinstance(configurable, dict):
            encrypted["configurable"] = await encrypt_json_if_needed(
                configurable, encryption_instance, model_type, field="context"
            )

    return (field_name, encrypted)


async def encrypt_request(
    data: dict[str, Any],
    model_type: ModelType,
    fields: list[str],
    encryption_instance: Encryption | None = None,
) -> dict[str, Any]:
    """Encrypt specified fields in request data before passing to ops layer (in parallel).

    This is a generic helper that handles encryption for any object type.
    It uses the ContextVar to get encryption context (set by middleware or endpoint).

    Special handling: When 'config' is in fields, also encrypts config['configurable']
    since the ops layer syncs config['configurable'] to context.

    Only processes fields that exist in the data to avoid adding new fields.

    Args:
        data: Request data dict to encrypt
        model_type: Type identifier passed to EncryptionContext.model (e.g., "run", "cron", "thread")
        fields: List of field names to encrypt (e.g., ["metadata", "kwargs"])
        encryption_instance: Optional encryption instance (auto-fetched if None)

    Returns:
        Data dict with encrypted fields

    Example:
        payload = await encrypt_request(
            payload,
            "run",
            ["metadata"]
        )
    """
    if encryption_instance is None:
        encryption_instance = get_encryption_instance()
        if encryption_instance is None:
            return data

    results = await asyncio.gather(
        *[
            _encrypt_field(data, f, encryption_instance, model_type)
            for f in fields
            if f in data
        ]
    )
    for field_name, value in results:
        data[field_name] = value

    return data
