
# ? Scraping with selenium
# chrome_options = Options()
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--disable-dev-shm-usage')
# driver = webdriver.Chrome(options=chrome_options)

# driver.maximize_window()

# # url = "https://el.wikipedia.org/wiki/%CE%9B%CE%B9%CE%BF%CE%BD%CE%AD%CE%BB_%CE%9C%CE%AD%CF%83%CE%B9"
# url = "https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/qatar2022"
# driver.get(url)
# website_text = driver.find_element(By.XPATH, "/html/body").text
# print(website_text)

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
