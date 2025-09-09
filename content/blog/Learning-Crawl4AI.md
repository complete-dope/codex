---
title : Learning Crawl 4 AI 
date : 2025-02-20
layout : post
---

# Crawl-4-AI 

## Getting relevant data from Crawl4AI

Its good at generating markdown format but seems to be a a bit off getting structured data out from it !!

So it provides 2 inbuilt strategies :

### Using Content Selection (https://docs.crawl4ai.com/core/content-selection/)



#### Using CSS based selection

```python
import asyncio
import json 
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonXPathExtractionStrategy
from bs4 import BeautifulSoup

browser_config = BrowserConfig(
    headless=False,  # Changed to headless mode
    user_agent_mode="random"
)

crawler_config = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    magic=True,
    verbose=True,
    log_console=True,
    wait_until="networkidle",
    simulate_user=True,
    exclude_external_images=True,
    css_selector="div.yuRUbf a",
)

query = "https://www.google.com/search?q=health"

async def main():
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            urls = [query+"&start=0" , query+"&start=10"]
            result = await crawler.arun(
                url=query,
                config=crawler_config
            )
            
            print("Extraction completed successfully")
            # Add inside your main function
            with open("raw_html.html", "w", encoding="utf-8") as f:
                f.write(result.cleaned_html if hasattr(result, 'html') else "No HTML available")

            print("Cleaned HTML is  :" , result.cleaned_html) # print not helpful content from this !!
            
    except Exception as e:
        print(f"Crawling failed: {e}")

asyncio.run(main())
```




#### Using Schema based extraction : 



```python
import asyncio
import json 
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonXPathExtractionStrategy
from bs4 import BeautifulSoup

# Updated schema to be more resilient to Google's layout changes
schema = {
  "name": "Link-Title-Schema",
  "baseSelector": "//div[contains(@class, 'yuRUbf')]",  # More flexible match
  "fields": [
    {
      "name": "link",
      "selector": ".//a",  # Direct child anchor tag
      "type": "attribute",
      "attribute": "href"
    }
  ]
}

browser_config = BrowserConfig(
    headless=False,  # Changed to headless mode
    user_agent_mode="random"
)

crawler_config = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    magic=True,
    verbose=True,
    log_console=True,
    wait_until="networkidle",
    simulate_user=True,
    exclude_external_images=True,
    extraction_strategy=JsonXPathExtractionStrategy(schema=schema, verbose=True) 
)

query = "https://www.google.com/search?q=health"


async def main():
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            urls = [query+"&start=0" , query+"&start=10"]
            result = await crawler.arun(
                url=query,
                config=crawler_config
            )
            
            print("Extraction completed successfully")
            print("Cleaned HTML is  :" , result.cleaned_html)

            soup = BeautifulSoup(result.html, "html.parser") 
            links = [a["href"] for a in soup.select("div.yuRUbf a") if a.has_attr("href")]

            print("Extracted Links:", links)

            with open("extracted_links.txt", "w") as f:
                f.write("\n".join(links))
            
            # Check if extracted content exists
            if result.extracted_content:
                print(f"Length of extracted content: {len(result.extracted_content)}")
                
                try:
                    # Try to parse as JSON
                    data = json.loads(result.extracted_content)
                    print(f"Successfully parsed JSON with {len(data)} entries")
                except json.JSONDecodeError as e:
                    print(f"Could not parse JSON: {e}")
                    print(f"Raw content: {result.extracted_content[:200]}...")
            else:
                print("No content was extracted")
            
            # Save markdown and media
            try:
                with open("result_markdown.md", "w") as f:
                    f.write(result.markdown)
                print("Saved markdown file")
                
                with open("result_media.md", "w") as f:
                    f.write(str(result.media))
                print("Saved media file")
            except IOError as e:
                print(f"Error saving files: {e}")
                
    except Exception as e:
        print(f"Crawling failed: {e}")

asyncio.run(main())

```


### Custom Methods to get relevant data: 

Get the full HTML from the page and then pass this to the beautiful soup parser ( to get the HTML ) and then use the logic there to get the desired output !!

```python
soup = BeautifulSoup(result.html, "html.parser") 
links = [a["href"] for a in soup.select("div.yuRUbf a") if a.has_attr("href")]
print("Extracted Links:", links)
with open("extracted_links.txt", "w") as f:
    f.write("\n".join(links))
```

## Adding args to crawler and browser configs

`--disable-blink-features=AutomationControlled` : very important features , without this the browser sends it as a webdriver request which is probably bad !

1. extra-args : [--disable-blink-features=AutomationControlled] in browser config 
2. text-only modes 


Removing Popups :
*remove_overlay_elements = True
*magic = True
*exclude_external_links = True
*exclude_tags = [ "" , "" , "" ]


The log saying `Console error: Cannot redefine property: webdriver` means that the code has some property in the config file which is trying to change the existing property. So basically the browsers have prevented this property from getting changed ... !



### Strange
`arun_many` seems to work fine in headless = False mode !! debug what happen when using just arun  


#### Nice links to visit : 
1. https://docs.crawl4ai.com/api/parameters/























