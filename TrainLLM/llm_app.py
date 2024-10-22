import argparse
import logging
import requests
import sys

from bs4 import BeautifulSoup
from GoogleNews import GoogleNews
from urllib.parse import urlparse, parse_qs
from transformers import T5ForConditionalGeneration, T5Tokenizer

log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM application for summarizing news from the web")
    parser.add_argument("--topic", type=str, required=True, help="News topic")
    return parser.parse_args()

def clean_url(url):
    """Extract meaningfull part from a URL"""

    # First, extract the part of the URL that contains the actual article link
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # Check if the URL contains a 'url' or 'q' query parameter (common in Google News links)
    if 'url' in query_params:
        cleaned_url = query_params['url'][0]
    elif 'q' in query_params:
        cleaned_url = query_params['q'][0]
    else:
        # If no such query parameters, use the original URL
        cleaned_url = url

    # Now, truncate the URL at the first '&' symbol, if present
    cleaned_url = cleaned_url.split('&')[0]  # Truncate at the first '&'

    return cleaned_url

def search_google_news(query, language='en', pages=1):
    """Function to search for Google News"""
    googlenews = GoogleNews(lang=language)
    googlenews.search(query)

    all_articles = []
    for page in range(1, pages + 1):
        googlenews.getpage(page)
        all_articles.extend(googlenews.result())

    return all_articles

def get_link_text(url):
    return clean_url(url)
    
# Function to fetch article text from a URL
def get_article_text(url):
    url = get_link_text(url)  # Clean the URL before making the request
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the main content (this can vary by website, so we target common tags)
        paragraphs = soup.find_all('p')  # Get all <p> tags, common for article text
        article_text = ' '.join([para.get_text() for para in paragraphs])

        return article_text
    except Exception as e:
        print(f"Failed to fetch the article text: {e}")
        return None
    
def summarize_article(article_text, model, tokenizer):
    """Summarizes the given article text using the fine-tuned model."""
    inputs = tokenizer(article_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    args = parse_args()
    query = args.topic

    log.info(f"Searching for news related to: {query}")

    # Load the fine-tuned model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_model")
    tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_model")

    news_articles = search_google_news(query)

    # Loop through URLs, fetch and summarize each article
    for article in news_articles:
        try:
            url = article['link']
            print(f"Fetching news from: {url}")
            print("---------------------------------")
            article_text = get_article_text(url)
            print(f"Article:\n{article['title']}\n")
            print(f"Published: {article['date']}")
            print("---------------------------------")
            summary = summarize_article(article_text, model, tokenizer)
            print(f"Summary:\n{summary}\n")
            print("----------------------------------------------------------------------------------------------")
        except Exception as e:
            print(f"Error processing {url}: {e}")
        print("---------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------")

if __name__ == "__main__":
    log.info(f"Program start...")
    main()
