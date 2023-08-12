import json
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import mongoDB_connection

mongoDB_connection.connect_to_mongodb()
db = mongoDB_connection.db
db.command("collMod", "url", validator=mongoDB_connection.validator())


# ? gets urls from a json file
def insert_urls_not_allowed_by_bs4(file_path):
    f = open(file_path)
    urls = json.load(f)
    for url in urls:
        # ? using undetected selenium
        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        driver = uc.Chrome(options=options)

        try:
            driver.get(url)
            time.sleep(5)

            timeout = 10
            wait = WebDriverWait(driver, timeout)
            # ? check if website has a body
            element_present = EC.presence_of_element_located(
                (By.XPATH, "/html/body"))

            wait.until(element_present)
            print(url, " loaded successfully!")
            website_text = driver.find_element(By.XPATH, "/html/body").text
            language = driver.find_element(
                By.XPATH, "//html").get_attribute('lang')

            print(url, ": ", language)

            # todo: insert into mongodb
            if not db.get_collection('url').find_one({"url": url}):
                # ? url does not exist in the collection
                new_link = {"url": url, "text": website_text,
                            "language": language}
                db.get_collection("url").insert_one(new_link)
                print("inserted ", url)

        except Exception as e:
            print("error" + str(e))
        finally:
            print("quitting driver")
            driver.quit()


insert_urls_not_allowed_by_bs4(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\not_allowed.json")

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
