import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ? gets urls from a json file
def insert_urls_not_allowed_by_bs4(file_path):
    f = open(file_path)
    urls = json.load(f)
    for url in urls:
        try:
            chrome_options = Options()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-dev-shm-usage')

            driver = webdriver.Chrome(options=chrome_options)
            driver.maximize_window()
            driver.get(url)

            website_text = driver.find_element(By.XPATH, "/html/body").text
            print(website_text)
            timeout = 10
        except Exception as e:
            print(url, " ", type(e))
        finally:
            driver.quit()


# insert_urls_not_allowed_by_bs4(
    # r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\not_allowed.json")

# ? Scraping with selenium
# chrome_options = Options()
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--disable-dev-shm-usage')
# driver = webdriver.Chrome(options=chrome_options)

# driver.maximize_window()

# # url = "https://akispetretzikis.com/blog/articles/ti-mageireyoyme-stis-diakopes"
# url = "https://www.kathimerini.gr/k/travel/"
# # print(url)

# driver.get(url)

# # website_text = driver.find_element(By.XPATH, "/html/body").text
# # print(website_text)

# timeout = 10

# # Wait for the presence of a specific element on the page
# try:
#     # ? check if website has a body
#     element_present = EC.presence_of_element_located(
#         (By.XPATH, "/html/body"))
#     WebDriverWait(driver, timeout).until(element_present)
#     print("Page loaded successfully!")
# except TimeoutException:
#     print("Timed out waiting for page to load.")
# finally:
#     driver.quit()

# driver.quit()

# ? for websites throwing FLOC error
# options = uc.ChromeOptions()
# options.add_argument("--headless=new")
# driver = uc.Chrome(options=options)

# url = "https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/qatar2022"
# driver.get(url)
# time.sleep(5)

# website_text = driver.find_element(By.XPATH, "/html/body").text
# print(website_text)
# driver.quit()
