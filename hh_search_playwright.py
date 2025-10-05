"""Automation helpers for the hh.ru advanced vacancy search form."""
import argparse
import asyncio
import json
import re
from urllib.parse import urlencode
from urllib.request import urlopen

from playwright.async_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

ADVANCED_SEARCH_URL = "https://hh.ru/search/vacancy/advanced"
KEYWORDS_INPUT = "[data-qa=\"vacancysearch__keywords-input\"]"
SALARY_INPUT = "[data-qa=\"advanced-search-salary\"]"
REGION_INPUT = "[data-qa=\"advanced-search-region-add\"]"
REGION_CHIP_DELETE = "[data-qa=\"selected-regions\"] [data-qa=\"chip-delete-action\"]"
SUBMIT_BUTTON = "[data-qa=\"advanced-search-submit-button\"]"
COOKIES_ACCEPT_BUTTON = "[data-qa=\"cookies-policy-informer-accept\"]"


def resolve_area_id(region: str) -> str:
    """Return the hh.ru area id for the provided region name."""
    params = urlencode({"text": region})
    with urlopen(f"https://api.hh.ru/suggests/area_leaves?{params}") as response:
        payload = json.loads(response.read().decode("utf-8"))
    items = payload.get("items", [])
    if not items:
        raise ValueError(f"No area suggestions returned for '{region}'")
    normalized = region.lower()
    for item in items:
        if item.get("text", "").lower() == normalized:
            return item["id"]
    return items[0]["id"]


async def accept_cookies(page: Page) -> None:
    """Dismiss the cookie banner if it appears."""
    try:
        await page.locator(COOKIES_ACCEPT_BUTTON).click(timeout=2000)
    except PlaywrightTimeoutError:
        pass

async def clear_selected_regions(page: Page) -> None:
    """Remove all currently selected region chips."""
    delete_button = page.locator(REGION_CHIP_DELETE)
    while await delete_button.count():
        await delete_button.first.click()
        await page.wait_for_timeout(100)


async def select_region(page: Page, region: str) -> None:
    """Populate hidden area inputs with the resolved region id."""
    area_id = await asyncio.to_thread(resolve_area_id, region)
    await clear_selected_regions(page)

    input_box = page.locator(REGION_INPUT).first
    await input_box.fill(region)

    await page.evaluate(
        """
        (value) => {
            const form = document.querySelector('form');
            if (!form) {
                return;
            }
            form.querySelectorAll('input[name=\"area\"]').forEach((el) => el.remove());
            form.querySelectorAll('input[name=\"L_save_area\"]').forEach((el) => el.remove());

            const areaInput = document.createElement('input');
            areaInput.type = 'hidden';
            areaInput.name = 'area';
            areaInput.value = value;
            form.appendChild(areaInput);

            const saveAreaInput = document.createElement('input');
            saveAreaInput.type = 'hidden';
            saveAreaInput.name = 'L_save_area';
            saveAreaInput.value = 'true';
            form.appendChild(saveAreaInput);
        }
        """,
        area_id,
    )

async def fill_search_form(
    page: Page,
    profession: str,
    salary_from: str,
    region: str,
) -> None:
    """Populate the main fields on the advanced search form."""
    await page.wait_for_load_state("networkidle")
    await accept_cookies(page)

    if profession:
        await page.locator(KEYWORDS_INPUT).first.fill(profession)

    if salary_from is not None:
        await page.locator(SALARY_INPUT).first.fill(str(salary_from))

    if region:
        await select_region(page, region)

    await page.locator(SUBMIT_BUTTON).first.click()
    await page.wait_for_load_state("networkidle")
    await page.wait_for_url(re.compile(r"/search/(vacancy|job)"), timeout=30000)


async def run_search(
    profession: str,
    salary_from: str,
    region: str,
    headless: bool,
) -> None:
    """Launch Chromium, populate the form and output the resulting URL."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            await page.goto(ADVANCED_SEARCH_URL, wait_until="networkidle")
            await fill_search_form(page, profession, salary_from, region)
            print("Results page URL:", page.url)
        finally:
            await browser.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate the hh.ru advanced vacancy search form with Playwright",
    )
    parser.add_argument(
        "--profession",
        default="QA Engineer",
        help="Profession, position or company name to search for.",
    )
    parser.add_argument(
        "--salary-from",
        dest="salary_from",
        default="100000",
        help="Minimum salary value.",
    )
    parser.add_argument(
        "--region",
        default="Москва",
        help="Region name to search within (e.g. 'Москва').",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser without UI.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_search(
            profession=args.profession,
            salary_from=args.salary_from,
            region=args.region,
            headless=args.headless,
        )
    )
