"""Static product inventory (source of truth) and lookup helpers."""

from __future__ import annotations

from typing import Optional

from .models import Product

# The brief wrote prices like `$2450.00`; kept here as plain floats.
MOCK_INVENTORY: tuple[Product, ...] = (
    Product(id="LNX-001", name="Classic Silk Trench Coat", brand="Beaumont",      price=2450.00, color="Beige",    in_stock=True),
    Product(id="LNX-002", name="Monogram Leather Tote",     brand="Gucci",         price=1850.00, color="Noir",     in_stock=True),
    Product(id="LNX-003", name="Cashmere Knit Sweater",     brand="VersAI Luxury", price=950.00,  color="Camel",    in_stock=False),
    Product(id="LNX-004", name="Croco-Embossed Loafers",    brand="VersAI Luxury", price=1200.00, color="Bordeaux", in_stock=True),
)

_BY_NAME = {p.name.casefold(): p for p in MOCK_INVENTORY}


def find_by_name(name: str) -> Optional[Product]:
    return _BY_NAME.get(name.strip().casefold()) if name else None


def search(text: str) -> Optional[Product]:
    """Cheap keyword match for the mock shopper: best name/brand token overlap wins."""
    haystack = text.casefold()
    best, best_score = None, 0
    for product in MOCK_INVENTORY:
        tokens = set(product.name.casefold().split()) | {product.brand.casefold()}
        score = sum(1 for token in tokens if token in haystack)
        if score > best_score:
            best, best_score = product, score
    return best if best_score else None
