"""End-to-end Playwright tests for the Ancient Chinese Search Engine SPA.

Requires:
    uv add --dev pytest-playwright playwright
    uv run playwright install chromium

Run:
    uv run pytest apps/backend/tests/e2e/ -v --headed   # visible browser
    uv run pytest apps/backend/tests/e2e/ -v            # headless

The tests target the server already running on APP_URL (default http://localhost:8013).
API calls that require Neo4j data are intercepted via page.route() so the tests
remain deterministic even against an empty database.

Design notes:
- Use wait_until="domcontentloaded" in page.goto() to avoid blocking on CDN scripts
  (Alpine.js, Tailwind, Chart.js, Google Fonts). Those CDNs can be slow on cold load;
  domcontentloaded fires as soon as the HTML is parsed.
- After goto, use wait_for_function to gate on Alpine.js being ready before asserting
  on any x-data / x-show / x-text bound elements.
"""
from __future__ import annotations

import json

import pytest
from playwright.sync_api import Page, Route, expect

APP_URL = "http://localhost:8013"

# ── helpers ──────────────────────────────────────────────────────────────────

def _go(page: Page, path: str = "/") -> None:
    """Navigate to the SPA without blocking on CDN script downloads.

    Uses wait_until="commit" (fires when response headers arrive) rather than
    "domcontentloaded" or "load" to avoid stalling on slow CDN resources
    (Alpine.js, Tailwind, Chart.js, Google Fonts). Alpine initialization is
    then gated separately via wait_for_function.
    """
    page.goto(f"{APP_URL}{path}", wait_until="commit")
    # Wait up to 30 s for Alpine.js to boot (covers cold CDN + slow networks).
    page.wait_for_function("() => typeof window.Alpine !== 'undefined'", timeout=30_000)


def _switch_locale(page: Page, locale: str) -> None:
    """Change the UI locale via the sidebar select."""
    sel = page.locator("nav#sidebar select")
    sel.select_option(locale)
    # Give Alpine time to react to the @change event and fetch the i18n JSON.
    page.wait_for_timeout(1000)


def _mock_json(route: Route, payload: dict, status: int = 200) -> None:
    route.fulfill(
        status=status,
        content_type="application/json",
        body=json.dumps(payload),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. HEALTH / API contract
# ═════════════════════════════════════════════════════════════════════════════

def test_health_endpoint_returns_ok(page: Page) -> None:
    """GET /health must return {"status": "ok"}."""
    resp = page.goto(f"{APP_URL}/health", wait_until="domcontentloaded")
    assert resp is not None and resp.ok
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_spa_index_serves_html(page: Page) -> None:
    """GET / must respond 200 text/html (SPA shell, not a redirect)."""
    # Use domcontentloaded so CDN scripts don't block us.
    resp = page.goto(APP_URL, wait_until="domcontentloaded")
    assert resp is not None and resp.status == 200
    assert "text/html" in (resp.headers.get("content-type", ""))


def test_static_i18n_en_loads(page: Page) -> None:
    """Static English i18n file must be accessible and contain expected keys."""
    resp = page.goto(f"{APP_URL}/static/i18n/en.json", wait_until="domcontentloaded")
    assert resp is not None and resp.ok
    data = resp.json()
    assert "nav.search" in data
    assert "search.placeholder" in data


def test_static_i18n_zh_loads(page: Page) -> None:
    resp = page.goto(f"{APP_URL}/static/i18n/zh.json", wait_until="domcontentloaded")
    assert resp is not None and resp.ok
    data = resp.json()
    assert data.get("nav.search") == "檢索"


def test_404_api_route_returns_json(page: Page) -> None:
    """Unknown /api/* paths must return a JSON 404, not the SPA HTML."""
    resp = page.goto(f"{APP_URL}/api/nonexistent-route-xyz", wait_until="domcontentloaded")
    assert resp is not None and resp.status == 404
    body = resp.json()
    assert "detail" in body


# ═════════════════════════════════════════════════════════════════════════════
# 2. SPA MOUNT & INITIAL STATE
# ═════════════════════════════════════════════════════════════════════════════

def test_spa_mounts_with_sidebar(page: Page) -> None:
    """The sidebar nav must be present after Alpine.js initialises."""
    _go(page)
    sidebar = page.locator("nav#sidebar")
    expect(sidebar).to_be_visible()


def test_spa_default_page_is_search(page: Page) -> None:
    """Search panel must be visible by default (page === 'search')."""
    _go(page)
    search_panel = page.locator("[x-show=\"page === 'search'\"]").first
    expect(search_panel).to_be_visible()


def test_spa_search_input_present(page: Page) -> None:
    """Search input field must be interactable on the default page."""
    _go(page)
    inp = page.locator("input[x-model='query']")
    expect(inp).to_be_visible()
    expect(inp).to_be_enabled()


def test_logo_mark_visible(page: Page) -> None:
    """The 古 logo mark in the sidebar must be visible."""
    _go(page)
    logo = page.locator("nav#sidebar .cjk", has_text="古").first
    expect(logo).to_be_visible()


# ═════════════════════════════════════════════════════════════════════════════
# 3. INTERNATIONALISATION
# ═════════════════════════════════════════════════════════════════════════════

def test_default_locale_is_zh(page: Page) -> None:
    """App boots with Chinese locale; nav shows Chinese labels when expanded."""
    _go(page)
    page.locator("nav#sidebar").hover()
    page.wait_for_timeout(400)
    label = page.locator("nav#sidebar .nav-label", has_text="檢索").first
    expect(label).to_be_visible()


def test_switch_to_english_locale(page: Page) -> None:
    """Selecting English in the locale switcher changes nav labels to English."""
    _go(page)
    _switch_locale(page, "en")
    page.locator("nav#sidebar").hover()
    page.wait_for_timeout(400)
    label = page.locator("nav#sidebar .nav-label", has_text="Search").first
    expect(label).to_be_visible()


def test_switch_to_arabic_sets_rtl(page: Page) -> None:
    """Switching to Arabic must set dir=rtl on the root element."""
    _go(page)
    _switch_locale(page, "ar")
    # setLocale() runs async in Alpine; wait for the attribute to actually be set.
    page.wait_for_function(
        "() => document.documentElement.getAttribute('dir') === 'rtl'",
        timeout=8_000,
    )
    direction = page.evaluate("document.documentElement.getAttribute('dir')")
    assert direction == "rtl"


def test_switch_to_japanese_locale(page: Page) -> None:
    """Japanese locale must update the HTML lang attribute."""
    _go(page)
    _switch_locale(page, "ja")
    page.wait_for_function(
        "() => document.documentElement.getAttribute('lang') === 'ja'",
        timeout=8_000,
    )
    lang = page.evaluate("document.documentElement.getAttribute('lang')")
    assert lang == "ja"


def test_locale_persisted_to_localstorage(page: Page) -> None:
    """Selected locale must be written to localStorage."""
    _go(page)
    _switch_locale(page, "en")
    stored = page.evaluate("localStorage.getItem('locale')")
    assert stored == "en"


# ═════════════════════════════════════════════════════════════════════════════
# 4. SIDEBAR NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("nav_target,expected_text", [
    ("upload",      "upload"),
    ("dashboard",   "dashboard"),
    ("pipeline",    "pipeline"),
    ("interactive", "interactive"),
])
def test_sidebar_nav_switches_page(page: Page, nav_target: str, expected_text: str) -> None:
    """Clicking sidebar buttons must switch the active page panel."""
    page.route("**/api/admin/stats", lambda r: _mock_json(r, {
        "documents": {"primary": 0, "secondary": 0, "total": 0},
        "pages": {"total": 0, "by_mode": {}, "by_fusion": {}},
        "chunks": {"total": 0, "primary": 0, "secondary": 0,
                   "translated": 0, "failed": 0, "untranslated": 0},
        "embeddings": {"embedded": 0, "total": 0},
        "ocr_eval": {}, "communities": 0, "relationships": 0,
        "pipeline_runs": [], "translations": {},
    }))
    page.route("**/api/admin/pipeline", lambda r: _mock_json(r, {"jobs": []}))
    page.route("**/api/upload/jobs*", lambda r: _mock_json(r, []))
    page.route("**/api/documents*", lambda r: _mock_json(r, []))

    _go(page)
    page.locator(f"button[\\@click=\"navigateTo('{nav_target}')\"]").click()
    page.wait_for_timeout(500)

    panel = page.locator(f"[x-show=\"page === '{nav_target}'\"]").first
    expect(panel).to_be_visible()


def test_nav_search_returns_to_search(page: Page) -> None:
    """Navigating away then back to search restores the search panel."""
    _go(page)
    page.locator("button[\\@click=\"navigateTo('upload')\"]").click()
    page.wait_for_timeout(300)
    page.locator("button[\\@click=\"navigateTo('search')\"]").click()
    page.wait_for_timeout(300)
    panel = page.locator("[x-show=\"page === 'search'\"]").first
    expect(panel).to_be_visible()


# ═════════════════════════════════════════════════════════════════════════════
# 5. SEARCH
# ═════════════════════════════════════════════════════════════════════════════

_MOCK_SEARCH_RESPONSE = {
    "query": "唐律",
    "mode": "hybrid",
    "intent": "factual",
    "bm25_ready": True,
    "total": 2,
    "primary_count": 2,
    "secondary_count": 0,
    "duration_ms": 42,
    "results": [
        {
            "chunk_id": "c001",
            "text": "名例律第一條",
            "citation": "《唐律疏議》卷一",
            "document_tier": "primary",
            "score": 0.92,
            "rerank_score": 0.95,
            "trust_score": 1.0,
            "keywords": ["律", "名例"],
            "editorial_layers": None,
            "translation_canonical": None,
            "translation_vernacular": None,
            "page_mode": "text",
            "rank": 1,
            "verified": False,
            "evidence_strength": None,
            "verifier_outcome": None,
            "intent": "factual",
        },
        {
            "chunk_id": "c002",
            "text": "十惡尤切",
            "citation": "《唐律疏議》卷一",
            "document_tier": "primary",
            "score": 0.85,
            "rerank_score": 0.88,
            "trust_score": 0.9,
            "keywords": ["十惡"],
            "editorial_layers": None,
            "translation_canonical": None,
            "translation_vernacular": None,
            "page_mode": "text",
            "rank": 2,
            "verified": True,
            "evidence_strength": "strong",
            "verifier_outcome": "supports",
            "intent": "factual",
        },
    ],
    "primaryRibbon": [],
    "secondaryRibbon": [],
}


def test_search_input_accepts_cjk_text(page: Page) -> None:
    """Typing CJK text into the search box must reflect in the input value."""
    _go(page)
    inp = page.locator("input[x-model='query']")
    inp.fill("唐律疏議")
    assert inp.input_value() == "唐律疏議"


def test_search_button_triggers_api_call(page: Page) -> None:
    """Clicking the search button must issue a GET /api/search request."""
    requests_seen: list[str] = []

    def _handle(route: Route) -> None:
        requests_seen.append(route.request.url)
        _mock_json(route, _MOCK_SEARCH_RESPONSE)

    page.route("**/api/search*", _handle)
    _go(page)
    page.locator("input[x-model='query']").fill("唐律")
    page.locator("button[\\@click='doSearch()']").click()
    page.wait_for_timeout(1500)

    assert any("/api/search" in u for u in requests_seen), "No search request fired"


def test_search_results_render(page: Page) -> None:
    """After a successful search the result cards must appear in the DOM."""
    page.route("**/api/search*", lambda r: _mock_json(r, _MOCK_SEARCH_RESPONSE))
    _go(page)
    page.locator("input[x-model='query']").fill("唐律")
    page.locator("button[\\@click='doSearch()']").click()
    primary_section = page.locator("[x-show='primaryResults.length > 0']").first
    expect(primary_section).to_be_visible(timeout=8_000)


def test_search_enter_key_submits(page: Page) -> None:
    """Pressing Enter in the search field must fire doSearch()."""
    fired: list[bool] = []
    page.route("**/api/search*", lambda r: (fired.append(True), _mock_json(r, _MOCK_SEARCH_RESPONSE)))
    _go(page)
    inp = page.locator("input[x-model='query']")
    inp.fill("律令")
    inp.press("Enter")
    page.wait_for_timeout(1500)
    assert fired, "Search was not triggered by Enter key"


def test_search_filter_tier_select_present(page: Page) -> None:
    """Tier filter select must be present and have the expected options."""
    _go(page)
    sel = page.locator("select[x-model='filters.tier']")
    expect(sel).to_be_visible()
    options = sel.locator("option").all()
    values = [o.get_attribute("value") for o in options]
    assert "both" in values
    assert "primary" in values
    assert "secondary" in values


def test_search_mode_select_has_hybrid_and_dense(page: Page) -> None:
    """Mode filter must expose both hybrid and dense options."""
    _go(page)
    sel = page.locator("select[x-model='searchMode']")
    expect(sel).to_be_visible()
    options = sel.locator("option").all()
    values = [o.get_attribute("value") for o in options]
    assert "hybrid" in values
    assert "dense" in values


def test_search_error_state_shown_on_api_failure(page: Page) -> None:
    """A 500 from /api/search must surface an error message in the UI."""
    page.route("**/api/search*", lambda r: _mock_json(r, {"detail": "Neo4j down"}, status=500))
    _go(page)
    page.locator("input[x-model='query']").fill("anything")
    page.locator("button[\\@click='doSearch()']").click()
    error_el = page.locator("[x-show='searchError']").first
    expect(error_el).to_be_visible(timeout=6_000)


# ═════════════════════════════════════════════════════════════════════════════
# 6. UPLOAD PAGE
# ═════════════════════════════════════════════════════════════════════════════

def test_upload_page_has_file_input(page: Page) -> None:
    """Upload page must contain a file input element."""
    page.route("**/api/upload/jobs*", lambda r: _mock_json(r, []))
    _go(page)
    page.locator("button[\\@click=\"navigateTo('upload')\"]").click()
    page.wait_for_timeout(500)
    file_input = page.locator("[x-show=\"page === 'upload'\"] input[type='file']")
    expect(file_input).to_be_attached()


def test_upload_page_has_ocr_model_select(page: Page) -> None:
    """Upload page must expose an OCR model selector."""
    page.route("**/api/upload/jobs*", lambda r: _mock_json(r, []))
    _go(page)
    page.locator("button[\\@click=\"navigateTo('upload')\"]").click()
    page.wait_for_timeout(500)
    upload_panel = page.locator("[x-show=\"page === 'upload'\"]").first
    ocr_sel = upload_panel.locator("select[x-model='upload.ocrModel']")
    expect(ocr_sel).to_be_visible()


# ═════════════════════════════════════════════════════════════════════════════
# 7. DASHBOARD (admin stats)
# ═════════════════════════════════════════════════════════════════════════════

# The dashboard template uses stats.documents.total for the headline number.
_MOCK_STATS = {
    "documents": {"primary": 5, "secondary": 12, "total": 17},
    "pages": {"total": 300, "by_mode": {"text": 200, "ocr": 100}, "by_fusion": {}},
    "chunks": {
        "total": 4000, "primary": 2000, "secondary": 2000,
        "translated": 1800, "failed": 50, "untranslated": 150,
    },
    "embeddings": {"embedded": 3800, "total": 4000},
    "ocr_eval": {"approved": 90, "flagged": 5, "unevaluated": 5},
    "communities": 42,
    "relationships": 1500,
    "pipeline_runs": [],
    "translations": {},
}


def test_dashboard_loads_stats(page: Page) -> None:
    """Navigating to dashboard must fire /api/admin/stats and render the panel."""
    page.route("**/api/admin/stats", lambda r: _mock_json(r, _MOCK_STATS))
    _go(page)
    page.locator("button[\\@click=\"navigateTo('dashboard')\"]").click()
    page.wait_for_timeout(500)
    dashboard_panel = page.locator("[x-show=\"page === 'dashboard'\"]").first
    expect(dashboard_panel).to_be_visible(timeout=5_000)


def test_dashboard_shows_document_count(page: Page) -> None:
    """Dashboard stats panel must display the correct total document count (17)."""
    page.route("**/api/admin/stats", lambda r: _mock_json(r, _MOCK_STATS))
    _go(page)
    page.locator("button[\\@click=\"navigateTo('dashboard')\"]").click()
    page.wait_for_timeout(500)
    # stats.documents.total = 17; template renders it via .toLocaleString()
    panel = page.locator("[x-show=\"page === 'dashboard'\"]").first
    expect(panel).to_contain_text("17", timeout=8_000)


# ═════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE PAGE
# ═════════════════════════════════════════════════════════════════════════════

def test_pipeline_page_renders(page: Page) -> None:
    """Pipeline page must render without JS errors."""
    page.route("**/api/admin/pipeline", lambda r: _mock_json(r, {"jobs": []}))
    page.route("**/api/admin/health", lambda r: _mock_json(r, {
        "neo4j": "ok", "minio": "ok", "silra": "ok",
    }))
    _go(page)
    page.locator("button[\\@click=\"navigateTo('pipeline')\"]").click()
    page.wait_for_timeout(500)
    panel = page.locator("[x-show=\"page === 'pipeline'\"]").first
    expect(panel).to_be_visible(timeout=5_000)


# ═════════════════════════════════════════════════════════════════════════════
# 9. AUTH / LOGIN MODAL
# ═════════════════════════════════════════════════════════════════════════════

def test_sign_in_button_opens_modal(page: Page) -> None:
    """Clicking 'Sign in' in the sidebar must open the login modal."""
    _go(page)
    page.locator("nav#sidebar").hover()
    page.wait_for_timeout(400)
    sign_in_btn = page.locator("button[\\@click='showLoginModal = true']")
    expect(sign_in_btn).to_be_visible()
    sign_in_btn.click()
    modal = page.locator("[x-show='showLoginModal']").first
    expect(modal).to_be_visible(timeout=3_000)


def test_login_modal_has_email_and_password_fields(page: Page) -> None:
    """Login modal must expose email and password inputs."""
    _go(page)
    page.locator("nav#sidebar").hover()
    page.wait_for_timeout(400)
    page.locator("button[\\@click='showLoginModal = true']").click()
    modal = page.locator("[x-show='showLoginModal']").first
    expect(modal.locator("input[x-model='loginEmail']")).to_be_visible(timeout=3_000)
    expect(modal.locator("input[x-model='loginPassword']")).to_be_visible()


def test_login_modal_close_button_dismisses(page: Page) -> None:
    """The × button in the login modal must close it."""
    _go(page)
    page.locator("nav#sidebar").hover()
    page.wait_for_timeout(400)
    page.locator("button[\\@click='showLoginModal = true']").click()
    modal = page.locator("[x-show='showLoginModal']").first
    expect(modal).to_be_visible(timeout=3_000)
    modal.locator("button[\\@click='showLoginModal = false; authError = null']").first.click()
    page.wait_for_timeout(400)
    expect(modal).to_be_hidden(timeout=3_000)


def test_login_api_error_shown_in_modal(page: Page) -> None:
    """A failed /api/auth/login must display the error inside the modal."""
    page.route("**/api/auth/login", lambda r: _mock_json(
        r, {"detail": "Invalid credentials"}, status=401
    ))
    _go(page)
    page.locator("nav#sidebar").hover()
    page.wait_for_timeout(400)
    page.locator("button[\\@click='showLoginModal = true']").click()
    modal = page.locator("[x-show='showLoginModal']").first
    expect(modal).to_be_visible(timeout=3_000)
    modal.locator("input[x-model='loginEmail']").fill("bad@example.com")
    modal.locator("input[x-model='loginPassword']").fill("wrong")
    modal.locator("button[\\@click='login()']").click()
    error_el = modal.locator("[x-show='authError']").first
    expect(error_el).to_be_visible(timeout=5_000)


# ═════════════════════════════════════════════════════════════════════════════
# 10. NO JS ERRORS ON ANY PAGE
# ═════════════════════════════════════════════════════════════════════════════

def test_no_console_errors_on_load(page: Page) -> None:
    """The SPA must not emit any console errors during initial load."""
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.route("**/api/auth/me", lambda r: _mock_json(r, {}, status=404))
    page.route("**/api/admin/stats", lambda r: _mock_json(r, _MOCK_STATS))
    page.route("**/api/admin/health", lambda r: _mock_json(r, {
        "neo4j": "ok", "minio": "ok", "silra": "ok",
    }))
    _go(page)
    # Exclude noise from external CDNs/fonts which are outside our control.
    relevant = [e for e in errors if "Failed to load resource" not in e
                and "ERR_FAILED" not in e
                and "fonts.googleapis" not in e
                and "cdn.tailwindcss" not in e
                and "cdn.jsdelivr" not in e
                and "unpkg.com" not in e]
    assert relevant == [], f"Console errors on load: {relevant}"
