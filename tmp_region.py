import asyncio
from pprint import pprint
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://hh.ru/search/vacancy/advanced", wait_until="networkidle")
        try:
            await page.locator('[data-qa="cookies-policy-informer-accept"]').click(timeout=3000)
        except Exception:
            pass
        await page.locator('[data-qa="advanced-search-region-selectFromList"]').click()
        await page.wait_for_timeout(500)
        # collect all elements with data-qa in modal container
        elements = await page.query_selector_all('[data-qa]')
        qa_names = set()
        for el in elements:
            attr = await el.get_attribute('data-qa')
            if attr and ('region' in attr or 'clarification' in attr):
                qa_names.add(attr)
        pprint(sorted(qa_names))
        await browser.close()

asyncio.run(main())
