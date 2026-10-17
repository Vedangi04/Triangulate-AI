import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# DEBUG: Let's see if the script even sees the key
key = os.getenv("TAVILY_API_KEY")

if not key:
    print("❌ Script cannot find the .env file or the key inside it.")
else:
    # Print the first 8 characters of the key to verify it's the right one
    # without showing the whole secret.
    print(f"Checking key starting with: {key[:8]}...")

    try:
        tavily = TavilyClient(api_key=key)
        response = tavily.search(query="Current CEO of NVIDIA")
        print("✅ Success! Result:", response['results'][0]['content'][:100], "...")
    except Exception as e:
        print(f"❌ Error from Tavily: {e}")
