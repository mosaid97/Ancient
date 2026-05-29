"""The canonical 7-step normalization pipeline (plan §2.7).

This is the single public entry point that every consumer (indexer,
translator, citation linker, verifier) imports::

    from apps.backend.normalize import normalize_canonical

    result = normalize_canonical(
        "貞觀十九年，唐太宗李代民征高麗",
        lang="zh",
        era="Tang",
    )
    print(result.canonical)
    print(result.steps)

Per the §2.7 defaults:

============== =========================== =======
step           default                     notes
============== =========================== =======
NFC            on                          always
whitespace     on                          always
T-S            on (s2t)                    always; OpenCC
異體字          on                          always; seed map
避諱            on iff ``era`` is provided  era-conditional; ``unsafe`` flag
通假字          OFF                         translator-only by default
mojimoji       on iff lang∉zh*             ja / mixed only
============== =========================== =======
"""

from __future__ import annotations

from dataclasses import dataclass, field

from apps.backend.normalize import (
    kana,
    loan,
    nfc,
    taboo,
    tsc,
    variants,
    whitespace,
)


@dataclass
class NormalizeStep:
    """One step in the canonical pipeline."""

    name: str
    applied: bool
    before: str
    after: str

    @property
    def changed(self) -> bool:
        return self.applied and self.before != self.after


@dataclass
class CanonicalResult:
    """Result bundle returned by :func:`normalize_canonical`.

    Attributes:
        original: The unmodified input.
        canonical: The fully-normalized output.
        steps: Per-step trace useful for debugging + UI offset maps.
        lang: Resolved language tag.
        era: Resolved era keyword (``Tang`` / ``Song`` / ``Ming`` / ``Qing`` /
            ``None``).
        flags: The flags actually applied (echoes effective config).
    """

    original: str
    canonical: str
    steps: list[NormalizeStep] = field(default_factory=list)
    lang: str = "zh"
    era: str | None = None
    flags: dict[str, bool] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original != self.canonical


def _is_ja_or_mixed(lang: str) -> bool:
    lang = (lang or "").lower()
    return lang.startswith("ja") or lang in {"mixed", "kanbun"}


def normalize_canonical(
    text: str,
    *,
    lang: str = "zh",
    era: str | None = None,
    apply_taboo: bool | None = None,
    apply_loan: bool = False,
    apply_kana: bool | None = None,
    taboo_unsafe: bool = False,
    tsc_config: str = "s2t",
) -> CanonicalResult:
    """Run the 7-step canonical pipeline on ``text``.

    Args:
        text: Input string.
        lang: Language tag (``zh``, ``zh-classical``, ``zh-modern``, ``ja``,
            ``ja-kanbun``, ``ja-modern``, ``mixed``). Default ``zh``.
        era: Era keyword (``Tang`` / ``唐`` / ``Song`` / ``Ming`` / ``Qing``).
            ``None`` skips the 避諱 step.
        apply_taboo: Override; default = ``True`` iff ``era`` is given.
        apply_loan: Apply 通假字 step. Default ``False`` per §2.7.
        apply_kana: Override the auto kana step (default = ``True`` for
            ja / mixed languages).
        taboo_unsafe: Include 避諱 rows marked ``safe: false`` (e.g.
            高頻字 like 世/民). Default ``False``.
        tsc_config: OpenCC config; default ``s2t`` (canonical = traditional).

    Returns:
        A :class:`CanonicalResult` with the full per-step trace.
    """
    original = text or ""
    if apply_taboo is None:
        apply_taboo = era is not None
    if apply_kana is None:
        apply_kana = _is_ja_or_mixed(lang)

    flags = {
        "nfc": True,
        "whitespace": True,
        "tsc": True,
        "variants": True,
        "taboo": bool(apply_taboo and era is not None),
        "loan": bool(apply_loan),
        "kana": bool(apply_kana),
        "taboo_unsafe": bool(taboo_unsafe),
    }

    steps: list[NormalizeStep] = []
    current = original

    after = nfc.normalize(current)
    steps.append(NormalizeStep("nfc", True, current, after))
    current = after

    after = whitespace.normalize(current)
    steps.append(NormalizeStep("whitespace", True, current, after))
    current = after

    after = tsc.normalize(current, config=tsc_config) if flags["tsc"] else current
    steps.append(NormalizeStep("tsc", flags["tsc"], current, after))
    current = after

    after = variants.normalize(current)
    steps.append(NormalizeStep("variants", True, current, after))
    current = after

    if flags["taboo"]:
        after = taboo.normalize(current, era=era, unsafe=taboo_unsafe)
        steps.append(NormalizeStep("taboo", True, current, after))
        current = after
    else:
        steps.append(NormalizeStep("taboo", False, current, current))

    if flags["loan"]:
        after = loan.normalize(current)
        steps.append(NormalizeStep("loan", True, current, after))
        current = after
    else:
        steps.append(NormalizeStep("loan", False, current, current))

    if flags["kana"]:
        after = kana.normalize(current, lang=lang)
        steps.append(NormalizeStep("kana", True, current, after))
        current = after
    else:
        steps.append(NormalizeStep("kana", False, current, current))

    return CanonicalResult(
        original=original,
        canonical=current,
        steps=steps,
        lang=lang,
        era=era,
        flags=flags,
    )
