"""End-to-end browser tests against a running server (API + built SPA).

  uvicorn app.main:app --port 8000   (with seeded data: server/scripts/seed_demo.py)
  E2E_BASE=http://127.0.0.1:8000 python -m pytest e2e -q
"""
import os
import re

import pytest

pw = pytest.importorskip("playwright.sync_api")
BASE = os.environ.get("E2E_BASE", "http://127.0.0.1:8000")
SHOTS = os.environ.get("E2E_SHOTS")


def _up():
    import urllib.request
    try:
        return urllib.request.urlopen(f"{BASE}/api/health", timeout=2).status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="server not running")


@pytest.fixture(scope="module")
def page():
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 950})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.errors = errors
        yield pg
        b.close()


def shot(page, name):
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/{name}.png", full_page=True)


def login(page, email, pw_):
    page.goto(f"{BASE}/#/login")
    page.get_by_label("이메일").fill(email)
    page.get_by_label("비밀번호").fill(pw_)
    page.get_by_role("button", name="로그인").click()


def test_rejects_wrong_password(page):
    login(page, "admin@trajai.dev", "wrong-password")
    pw.expect(page.get_by_role("alert")).to_contain_text("incorrect")


def test_dashboard_after_login(page):
    login(page, "admin@trajai.dev", "admin1234")
    pw.expect(page.get_by_role("heading", name="대시보드")).to_be_visible()
    pw.expect(page.get_by_text("최근 시나리오")).to_be_visible()
    page.wait_for_timeout(600)
    shot(page, "dashboard")


def test_scenario_detail_and_live_replay(page):
    page.goto(f"{BASE}/#/scenarios")
    page.get_by_role("cell", name=re.compile("제트기류")).click()
    pw.expect(page.get_by_text("추정 결과")).to_be_visible()
    pw.expect(page.get_by_text("EKF · 일자 기상").first).to_be_visible()
    page.get_by_role("button", name="재생").click()
    pw.expect(page.get_by_text("완료")).to_be_visible(timeout=30_000)
    page.wait_for_timeout(500)
    shot(page, "scenario_detail")


def test_new_scenario_form_validation(page):
    page.goto(f"{BASE}/#/scenarios/new")
    page.get_by_label("선택 일자 기상 사용").uncheck()
    page.get_by_label("탄도계수 β (kg/m²)").fill("50")
    page.get_by_role("button", name="시뮬레이션 실행").click()
    pw.expect(page.get_by_text(re.compile("greater than or equal to 300"))).to_be_visible()
    shot(page, "new_scenario_validation")


def test_viewer_is_read_only(page):
    page.get_by_role("button", name="로그아웃").click()
    login(page, "viewer@trajai.dev", "password123")
    pw.expect(page.get_by_role("heading", name="대시보드")).to_be_visible()
    assert page.get_by_role("link", name="새 시나리오").count() == 0
    page.goto(f"{BASE}/#/admin")
    pw.expect(page.get_by_text("admin 이상 권한")).to_be_visible()


def test_viewer_can_export_csv(page):
    page.goto(f"{BASE}/#/scenarios")
    with page.expect_download() as dl:
        page.get_by_role("button", name="추정 결과 CSV").click()
    d = dl.value
    assert d.suggested_filename == "trajai_estimates.csv"
    head = open(d.path(), encoding="utf-8").readline()
    assert head.startswith("scenario_id,scenario,") and "threat_level" in head


def test_no_uncaught_errors(page):
    assert page.errors == []
