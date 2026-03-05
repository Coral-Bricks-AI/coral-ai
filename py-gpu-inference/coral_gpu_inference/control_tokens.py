#!/usr/bin/env python3
"""
Control tokens for structured product data serialization.

These tokens replace natural language delimiters (like "title:", "=", ";")
with dedicated tokens that won't be confused with actual product content.
During mean pooling, these tokens are excluded so they don't dilute embeddings.
"""

PRODUCT_FIELD_TOKENS = [
    '[product]', '[title]', '[category]', '[brand]', '[color]',
    '[attributes]', '[identifiers]', '[bullets]', '[description]', '[price]',
]

ROLE_TOKENS = [
    '[attr_key]', '[attr_val]', '[id_key]', '[id_val]',
]

SEPARATOR_TOKENS = ['[=]', '[;]', '[|]']

SPECIAL_VALUE_TOKENS = ['[unknown]']

QUERY_TOKENS = ['[query]']

CONTRASTIVE_CONTROL_TOKENS = (
    PRODUCT_FIELD_TOKENS + ROLE_TOKENS + SEPARATOR_TOKENS +
    SPECIAL_VALUE_TOKENS + QUERY_TOKENS
)

CONTRASTIVE_CONTROL_TOKEN_SET = set(CONTRASTIVE_CONTROL_TOKENS)
