from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["TAVILY_API_KEY"]=os.getenv('TAVILY_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')