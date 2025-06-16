from langchain_community.tools import TavilySearchResults

search = TavilySearchResults(max_results=2)
tools_list = [search]