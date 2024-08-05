from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["TAVILY_API_KEY"]=os.getenv('TAVILY_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError

from linebot.models import MessageEvent, TextMessage, TextSendMessage

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.vectorstores import Chroma
import json
import myPackage.GetRaw 
import myPackage.PartDataFrame
import myPackage.APIToDF