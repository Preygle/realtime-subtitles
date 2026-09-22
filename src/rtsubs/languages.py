"""Language table shared by the ASR and translation stages.

Each stage wants a different spelling of the same language:

* R2T2 / Qwen3-ASR take an English language *name* ("Chinese", "Japanese").
* NLLB-200 takes a FLORES-200 code ("zho_Hans", "jpn_Jpan").
* The overlay and config just want something human-readable.

``LANGUAGES`` is the single source of truth; the lookup helpers are forgiving
about case, ISO codes and common aliases so the UI and config files can use
whatever spelling is natural.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    #: Name passed to R2T2 / Qwen3-ASR.
    name: str
    #: ISO 639-1 (or 639-3 where no two-letter code exists).
    iso: str
    #: FLORES-200 code used by NLLB-200.
    flores: str
    aliases: tuple[str, ...] = ()


#: Ordered roughly by how well R2T2 supports each language.
LANGUAGES: tuple[Language, ...] = (
    Language("English", "en", "eng_Latn"),
    Language("Chinese", "zh", "zho_Hans", ("mandarin", "chinese (simplified)", "zh-cn")),
    Language("Cantonese", "yue", "yue_Hant", ("zh-hk",)),
    Language("Japanese", "ja", "jpn_Jpan"),
    Language("Korean", "ko", "kor_Hang"),
    Language("Spanish", "es", "spa_Latn"),
    Language("French", "fr", "fra_Latn"),
    Language("German", "de", "deu_Latn"),
    Language("Russian", "ru", "rus_Cyrl"),
    Language("Portuguese", "pt", "por_Latn"),
    Language("Italian", "it", "ita_Latn"),
    Language("Arabic", "ar", "arb_Arab"),
    Language("Hindi", "hi", "hin_Deva"),
    Language("Indonesian", "id", "ind_Latn"),
    Language("Vietnamese", "vi", "vie_Latn"),
    Language("Thai", "th", "tha_Thai"),
    Language("Turkish", "tr", "tur_Latn"),
    Language("Dutch", "nl", "nld_Latn"),
    Language("Polish", "pl", "pol_Latn"),
    Language("Swedish", "sv", "swe_Latn"),
    Language("Danish", "da", "dan_Latn"),
    Language("Finnish", "fi", "fin_Latn"),
    Language("Czech", "cs", "ces_Latn"),
    Language("Greek", "el", "ell_Grek"),
    Language("Romanian", "ro", "ron_Latn"),
    Language("Hungarian", "hu", "hun_Latn"),
    Language("Persian", "fa", "pes_Arab", ("farsi",)),
    Language("Malay", "ms", "zsm_Latn"),
    Language("Filipino", "tl", "tgl_Latn", ("tagalog",)),
    Language("Macedonian", "mk", "mkd_Cyrl"),
    Language("Ukrainian", "uk", "ukr_Cyrl"),
    Language("Hebrew", "he", "heb_Hebr"),
    Language("Norwegian", "no", "nob_Latn"),
)

#: Sentinel used in config and the UI for "let the model decide".
AUTO_DETECT = ""
AUTO_DETECT_LABEL = "Auto-detect"

_INDEX: dict[str, Language] = {}
for _lang in LANGUAGES:
    _INDEX[_lang.name.casefold()] = _lang
    _INDEX[_lang.iso.casefold()] = _lang
    _INDEX[_lang.flores.casefold()] = _lang
    for _alias in _lang.aliases:
        _INDEX[_alias.casefold()] = _lang


def lookup(value: str | None) -> Language | None:
    """Resolve a name, ISO code, FLORES code or alias to a :class:`Language`."""
    if not value:
        return None
    key = value.strip().casefold()
    if key in _INDEX:
        return _INDEX[key]
    # Tolerate regional suffixes like "en-US" or "pt_BR".
    base = key.replace("_", "-").split("-", 1)[0]
    return _INDEX.get(base)


def canonical_name(value: str | None, default: str = "") -> str:
    lang = lookup(value)
    return lang.name if lang else default


def to_flores(value: str | None, default: str = "eng_Latn") -> str:
    lang = lookup(value)
    return lang.flores if lang else default


def same_language(a: str | None, b: str | None) -> bool:
    la, lb = lookup(a), lookup(b)
    return la is not None and lb is not None and la.iso == lb.iso


def ui_choices() -> list[tuple[str, str]]:
    """(label, config value) pairs for the source-language dropdown."""
    return [(AUTO_DETECT_LABEL, AUTO_DETECT)] + [
        (lang.name, lang.name) for lang in LANGUAGES
    ]
