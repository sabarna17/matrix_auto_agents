# # Python program Continued
# # Webdriver For Firefox
# from selenium import webdriver
# driver = webdriver.Chrome()
# driver.get("https://mbasic.facebook.com")
# html = driver.page_source # Getting Source of Current URL / Web-Page Loaded
# print(html)
# # End


# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import time

# # Setup Chrome webdriver
# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')
# chrome_options.add_argument('--headless')
# driver = webdriver.Chrome(options=chrome_options)

# # Open Youtube
# url = 'https://www.youtube.com'
# driver.get(url)
# time.sleep(2)

# # Search for the channel
# search_box = driver.find_element_by_name('search_query')
# search_box.send_keys('Sarupyo Chatterjee')
# search_box.send_keys(Keys.RETURN)
# time.sleep(2)

# # Click on the channel from the search results
# channel_link = driver.find_element_by_link_text('Sarupyo Chatterjee')
# channel_link.click()
# time.sleep(2)

# # Get list of video URLs
# videos = driver.find_elements_by_xpath('//*[@id="video-title"]')
# video_list = [{'title': video.get_attribute('title'), 'url': video.get_attribute('href')} for video in videos]

# print(video_list)

# driver.quit() 