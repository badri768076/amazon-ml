"""Conservative text normalisation for business names, addresses and countries.

All rules are written once as vectorised Polars expressions (Rust regex engine),
so millions of rows are normalised without Python loops. The scalar helpers
``normalize_name`` / ``normalize_address`` / ``normalize_country`` run the very
same expressions on a one-element Series, so single-record behaviour always
matches the batch pipeline.

Design principle: only transformations that cannot plausibly merge two
different businesses (case, accents on Latin letters, punctuation, legal-form
and street-type spelling variants). Non-Latin scripts are preserved untouched.
"""
from __future__ import annotations

import logging

import polars as pl

from config import Config

log = logging.getLogger("ber.prep")

# --------------------------------------------------------------------------- #
# Dictionaries (spelling variants only - meaning preserved)
# --------------------------------------------------------------------------- #
NAME_ABBREV = {
    # legal forms
    "pvt": "private", "pvte": "private", "prv": "private", "priv": "private",
    "ltd": "limited", "ltda": "limited", "lmtd": "limited", "limted": "limited",
    "co": "company", "cos": "companies", "comp": "company",
    "corp": "corporation", "corpn": "corporation", "corpo": "corporation",
    "inc": "incorporated", "incorp": "incorporated", "incorporation": "incorporated",
    "l.l.c": "llc", "l.l.p": "llp", "pte": "private", "pty": "proprietary",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    # frequent, unambiguous business words
    "intl": "international", "int'l": "international", "natl": "national",
    "mfg": "manufacturing", "mfrs": "manufacturers", "mgmt": "management",
    "svcs": "services", "svc": "services", "srvs": "services", "assoc": "associates",
    "assocs": "associates", "bros": "brothers", "ent": "enterprises",
    "entp": "enterprises", "ents": "enterprises", "indus": "industries", "inds": "industries",
    "dept": "department", "univ": "university", "hosp": "hospital", "govt": "government",
    "tech": "technologies", "techs": "technologies", "technology": "technologies",
    "sys": "systems", "solns": "solutions", "soln": "solutions", "grp": "group",
    "hldgs": "holdings", "hldg": "holdings", "invest": "investments", "mktg": "marketing",
    "dist": "distributors", "distr": "distributors", "pharma": "pharmaceuticals",
    "chem": "chemicals", "engg": "engineering", "engr": "engineering",
    "constr": "construction", "consult": "consultants",
    "&": "and", "et": "and", "und": "and",
}

# Legal-form / filler tokens removed to build the "core" name (blocking + features)
LEGAL_TOKENS = {
    "private", "limited", "company", "companies", "corporation", "incorporated", "llc", "llp",
    "plc", "lp", "proprietary", "gmbh", "ag", "kg", "bv", "nv", "srl", "spa", "sa", "sas",
    "sasu", "sarl", "eurl", "sci", "snc", "opc", "the", "and", "of", "le", "la", "les", "de",
    "des", "du", "m/s", "ms", "messrs",
}

ADDR_ABBREV = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard", "ln": "lane", "dr": "drive",
    "hwy": "highway", "pkwy": "parkway", "sq": "square", "ct": "court", "pl": "place",
    "apt": "apartment", "appt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "blg": "building", "no": "number", "nr": "near", "opp": "opposite",
    "sec": "sector", "sect": "sector", "ngr": "nagar", "mkt": "market", "mg": "mg",
    "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk", "po": "post",
    "ps": "police station", "ind": "industrial", "indl": "industrial", "est": "estate",
    "e": "east", "w": "west", "n": "north", "s": "south", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest", "mt": "mount",
    "ft": "fort", "jn": "junction", "jct": "junction", "chk": "chowk",
    "cedex": "cedex", "fbg": "faubourg", "rte": "route", "imp": "impasse", "pl.": "place",
    "&": "and",
}

COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us", "united states": "us",
    "united states of america": "us", "america": "us", "etats unis": "us",
    "in": "in", "ind": "in", "india": "in", "bharat": "in", "republic of india": "in",
    "fr": "fr", "fra": "fr", "france": "fr", "republique francaise": "fr", "french republic": "fr",
    "uk": "gb", "gb": "gb", "gbr": "gb", "united kingdom": "gb", "great britain": "gb", "england": "gb",
    "de": "de", "deu": "de", "germany": "de", "deutschland": "de",
    "ca": "ca", "can": "ca", "canada": "ca", "es": "es", "spain": "es", "espana": "es",
    "it": "it", "italy": "it", "italia": "it", "jp": "jp", "japan": "jp",
    "cn": "cn", "china": "cn", "br": "br", "brazil": "br", "brasil": "br",
    "mx": "mx", "mexico": "mx", "au": "au", "australia": "au",
}
# Unknown countries pass through as their cleaned lowercase form (open set).


# --------------------------------------------------------------------------- #
# Polars expressions
# --------------------------------------------------------------------------- #
def _basic_clean(e: pl.Expr) -> pl.Expr:
    """Unicode NFKC->NFKD, strip Latin combining accents only, lowercase."""
    return (e.fill_null("").str.normalize("NFKC").str.to_lowercase()
            .str.normalize("NFKD").str.replace_all(r"[̀-ͯ]", "")
            .str.normalize("NFC"))


def _collapse_acronym_dots(e: pl.Expr) -> pl.Expr:
    # "s.a.s." -> "sas", "u.s.a" -> "usa", "j.p. morgan" -> "jp morgan"; "pvt.ltd." untouched here
    return e.str.replace_all(r"\b(\w)\.(\w)\.(\w)\.?", "${1}${2}${3}") \
            .str.replace_all(r"\b(\w)\.(\w)\b\.?", "${1}${2}")


def _punct_to_space(e: pl.Expr) -> pl.Expr:
    return (e.str.replace_all(r"&", " & ")
             .str.replace_all(r"['’`´]", "")               # o'brien -> obrien
             .str.replace_all(r"[^\w&\s]", " ")             # everything else -> space
             .str.replace_all(r"_", " ")
             .str.replace_all(r"\s+", " ").str.strip_chars())


def _map_tokens(e: pl.Expr, mapping: dict) -> pl.Expr:
    return (e.str.split(" ")
             .list.eval(pl.element().replace(mapping))
             .list.join(" ").str.replace_all(r"\s+", " ").str.strip_chars())


def name_expr(col: str = "business_name") -> pl.Expr:
    e = _collapse_acronym_dots(_basic_clean(pl.col(col)))
    e = e.str.replace_all(r"^\s*m\s*/\s*s\b\.?", " ")          # "M/s." honorific prefix (India)
    e = _punct_to_space(e)
    e = _map_tokens(e, NAME_ABBREV)
    return e.alias("normalized_name")


def address_expr(col: str = "business_address") -> pl.Expr:
    e = _collapse_acronym_dots(_basic_clean(pl.col(col)))
    e = (e.str.replace_all(r"\b(pin|pincode|pin code|zip|zipcode|postal code|cp)\s*[:\-]?\s*(\d{3})\s+(\d{3})\b",
                           "${1} ${2}${3}")                  # "pin 560 001" -> "pin 560001"
          .str.replace_all(r"(\d{5})-(\d{4})\b", "${1}")     # ZIP+4 -> ZIP
          .str.replace_all(r"#", " number "))
    e = _punct_to_space(e)
    e = e.str.replace_all(r"(\p{L})(\d)", "${1} ${2}")       # "plot12" -> "plot 12"
    e = _map_tokens(e, ADDR_ABBREV)
    return e.alias("normalized_address")


def country_expr(col: str = "country") -> pl.Expr:
    e = _punct_to_space(_basic_clean(pl.col(col)))
    e = e.replace(COUNTRY_ALIASES)
    return pl.when(e == "").then(None).otherwise(e).alias("normalized_country")


# --------------------------------------------------------------------------- #
# Scalar helpers (same rules as the batch expressions)
# --------------------------------------------------------------------------- #
def _apply_scalar(value: str | None, expr: pl.Expr, col: str) -> str | None:
    return pl.DataFrame({col: [value]}, schema={col: pl.Utf8}).select(expr).item()


def normalize_name(name: str | None) -> str:
    return _apply_scalar(name, name_expr("x"), "x") or ""


def normalize_address(address: str | None) -> str:
    return _apply_scalar(address, address_expr("x"), "x") or ""


def normalize_country(country: str | None) -> str | None:
    return _apply_scalar(country, country_expr("x"), "x")


# --------------------------------------------------------------------------- #
# Derived fields
# --------------------------------------------------------------------------- #
def add_derived(df: pl.LazyFrame | pl.DataFrame, cfg: Config) -> pl.LazyFrame:
    """Add normalised text + reusable token/numeric fields.

    Character n-grams are not materialised per row (that would multiply storage
    by ~20x); they are produced on demand by the blocking index and the sparse
    TF-IDF vectorisers, which *are* the stored n-gram representation.
    """
    legal = list(LEGAL_TOKENS)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    lf = lf.with_columns(name_expr(), address_expr(), country_expr())
    lf = lf.with_columns(
        pl.col("normalized_name").str.split(" ").list.eval(
            pl.element().filter(pl.element().str.len_chars() > 0)).alias("name_tokens"),
        pl.col("normalized_address").str.split(" ").list.eval(
            pl.element().filter(pl.element().str.len_chars() > 0)).alias("address_tokens"),
        pl.col("normalized_address").str.extract_all(r"\d+").alias("numeric_tokens"),
        pl.col("normalized_address").str.extract_all(cfg.postal_regex).list.unique(maintain_order=True)
          .alias("postal_code_candidates"),
    )
    lf = lf.with_columns(
        pl.col("name_tokens").list.eval(pl.element().filter(~pl.element().is_in(legal))).alias("core_tokens"),
    ).with_columns(
        pl.when(pl.col("core_tokens").list.len() > 0).then(pl.col("core_tokens").list.join(" "))
          .otherwise(pl.col("normalized_name")).alias("name_core"),
        pl.col("core_tokens").list.unique().list.sort().list.join(" ").alias("name_signature"),
        pl.col("core_tokens").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("name_acronym"),
        pl.col("address_tokens").list.eval(
            pl.element().filter(pl.element().str.contains(r"^\D{3,}$"))).alias("address_word_tokens"),
        pl.col("normalized_name").str.len_chars().alias("name_length"),
        pl.col("normalized_address").str.len_chars().alias("address_length"),
        pl.col("name_tokens").list.len().alias("name_word_count"),
        pl.col("address_tokens").list.len().alias("address_word_count"),
        pl.col("postal_code_candidates").list.first().alias("primary_postal"),
    ).with_columns(
        # building/house number: first short numeric token that is not a postal code
        pl.col("numeric_tokens").list.eval(
            pl.element().filter(pl.element().str.len_chars() <= 4)).list.first().alias("building_number"),
        pl.col("address_word_tokens").list.unique().list.sort().list.join(" ").alias("address_signature"),
    )
    return lf


NORM_COLUMNS = ["entity_id", "business_name", "business_address", "country",
                "normalized_name", "normalized_address", "normalized_country", "name_core",
                "name_signature", "name_acronym", "name_tokens", "core_tokens", "address_tokens",
                "address_word_tokens", "numeric_tokens", "postal_code_candidates", "primary_postal",
                "building_number", "address_signature", "name_length", "address_length",
                "name_word_count", "address_word_count"]


def preprocess_table(cfg: Config, table: str) -> str:
    """Normalise one raw table -> artifacts/normalized/<table>.parquet (checkpointed)."""
    from data_loader import scan

    out = cfg.norm_dir / f"{table}.parquet"
    if out.exists() and not cfg.force:
        log.info("[checkpoint] %s already normalised", table)
        return str(out)
    cfg.norm_dir.mkdir(parents=True, exist_ok=True)
    lf = add_derived(scan(cfg, table), cfg).select(NORM_COLUMNS)
    lf.sink_parquet(out, compression="zstd") if hasattr(lf, "sink_parquet") else lf.collect().write_parquet(out)
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    log.info("Normalised %-15s rows=%d -> %s", table, n, out.name)
    return str(out)


def load_normalized(cfg: Config, table: str) -> pl.DataFrame:
    p = cfg.norm_dir / f"{table}.parquet"
    if not p.exists():
        preprocess_table(cfg, table)
    return pl.read_parquet(p)
